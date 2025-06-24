"""
Description (essay_writer_agent.py):
This script defines the EssayWriterAgent class (analogous to ewriter in helper.py), which orchestrates a multi-step process for generating an essay. 
It utilizes a LangGraph StateGraph with a detailed AgentState to manage various components like the task, plan, research content, draft, critique, and revision tracking. Key nodes in the graph include:
planner: Creates an initial essay outline.
research_plan: Gathers initial research based on the task and plan using Tavily.
generate: Writes or revises the essay using the plan, accumulated research, and any critique.
reflect: Critiques the current draft.
research_critique: Performs further research based on the critique to improve the essay.
The graph facilitates an iterative refinement loop: generate -> reflect -> research_critique -> generate, until a maximum number of revisions is met. The LLM's structured output capability is used for generating search queries. State persistence is managed via SqliteSaver, and nodes are configured to be interruptible, aligning with the UI interactions defined in helper.py.
"""

from dotenv import load_dotenv
_ = load_dotenv()

from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, List
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langchain_core.pydantic_v1 import BaseModel
from tavily import TavilyClient
import os
import sqlite3 # for SqliteSaver connection

# --- State Definition ---
class AgentState(TypedDict):
    task: str
    plan: str
    draft: str
    critique: str
    content: List[str]
    revision_number: int
    max_revisions: int
    # Fields from helper.py that might be used internally or for UI state management
    lnode: str # To track the last executed node name
    queries: List[str] # To store generated search queries
    count: Annotated[int, lambda x, y: x + y] # Example for a counter if needed, matches helper

# --- Model and Tool Initialization ---
# These are moved into ewriter, but can be global if ewriter is not the primary way to run
# model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
# tavily = TavilyClient(api_key=os.environ.get("TAVILY_API_KEY"))

# --- Pydantic Model for Structured Output (Queries) ---
class Queries(BaseModel):
    queries: List[str]

# --- Prompts are now part of ewriter class ---

# Node functions will be methods of the ewriter class if we follow helper.py structure
# For standalone script, they'd be top-level or part of a different class.
# The helper.py defines ewriter class that encapsulates graph logic and nodes.
# This script will define the `ewriter` class to align with `helper.py`.

class EssayWriterAgent: # Renamed for clarity from ewriter to match file intent
    def __init__(self):
        self.model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
        self.tavily = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])

        # Prompts (as defined in helper.py's ewriter)
        self.PLAN_PROMPT = ("You are an expert writer tasked with writing a high level outline of a short 3 paragraph essay. "
                            "Write such an outline for the user provided topic. Give the three main headers of an outline of "
                             "the essay along with any relevant notes or instructions for the sections. ")
        self.WRITER_PROMPT = ("You are an essay assistant tasked with writing excellent 3 paragraph essays. "
                              "Generate the best essay possible for the user's request and the initial outline. "
                              "If the user provides critique, respond with a revised version of your previous attempts. "
                              "Utilize all the information below as needed: \n"
                              "------\n"
                              "Research Content:\n{content}") # Added placeholder
        self.RESEARCH_PLAN_PROMPT = ("You are a researcher charged with providing information that can "
                                     "be used when writing the following essay. Generate a list of search "
                                     "queries that will gather "
                                     "any relevant information. Only generate 3 queries max.")
        self.REFLECTION_PROMPT = ("You are a teacher grading an 3 paragraph essay submission. "
                                  "Generate critique and recommendations for the user's submission. "
                                  "Provide detailed recommendations, including requests for length, depth, style, etc.")
        self.RESEARCH_CRITIQUE_PROMPT = ("You are a researcher charged with providing information that can "
                                         "be used when making any requested revisions (as outlined below). "
                                         "Generate a list of search queries that will gather any relevant information. "
                                         "Only generate 2 queries max.")
        
        builder = StateGraph(AgentState)
        builder.add_node("planner", self.plan_node)
        builder.add_node("research_plan", self.research_plan_node)
        builder.add_node("generate", self.generation_node)
        builder.add_node("reflect", self.reflection_node)
        builder.add_node("research_critique", self.research_critique_node)
        
        builder.set_entry_point("planner")
        
        builder.add_edge("planner", "research_plan")
        builder.add_edge("research_plan", "generate")
        builder.add_conditional_edges(
            "generate", 
            self.should_continue, 
            {END: END, "reflect": "reflect"}
        )
        builder.add_edge("reflect", "research_critique")
        builder.add_edge("research_critique", "generate")
        
        # In-memory SQLite for checkpointer, ensuring thread safety for Gradio
        self.memory = SqliteSaver(conn=sqlite3.connect(":memory:", check_same_thread=False))
        self.graph = builder.compile(
            checkpointer=self.memory,
            # Interrupts as defined in helper.py if needed for UI control
            interrupt_after=['planner', 'generate', 'reflect', 'research_plan', 'research_critique']
        )

    def plan_node(self, state: AgentState):
        print("\n>> Planner Node executing...")
        messages = [
            SystemMessage(content=self.PLAN_PROMPT), 
            HumanMessage(content=state['task'])
        ]
        response = self.model.invoke(messages)
        return {
            "plan": response.content,
            "lnode": "planner", # For UI state tracking
            "count": state.get("count", 0) + 1 # For UI state tracking
        }

    def research_plan_node(self, state: AgentState):
        print("\n>> Research Plan Node executing...")
        structured_llm_for_queries = self.model.with_structured_output(Queries)
        queries_response = structured_llm_for_queries.invoke([
            SystemMessage(content=self.RESEARCH_PLAN_PROMPT),
            HumanMessage(content=state['task']) # Research based on the task
        ])
        
        current_content = state.get('content', [])
        generated_queries = []
        if queries_response.queries:
            generated_queries = queries_response.queries
            for q in generated_queries:
                print(f"  -- Conducting research for query: {q}")
                try:
                    response = self.tavily.search(query=q, max_results=2)
                    for r in response['results']:
                        current_content.append(r['content'])
                except Exception as e:
                    print(f"    !! Error during Tavily search for query '{q}': {e}")
                    current_content.append(f"Error fetching results for: {q}")
        else:
            print("  -- No queries generated by LLM for initial research.")
            
        return {
            "content": current_content,
            "queries": generated_queries, # Store the queries
            "lnode": "research_plan",
            "count": state.get("count", 0) + 1
        }

    def generation_node(self, state: AgentState):
        print("\n>> Generation Node executing...")
        research_summary = "\n\n".join(state.get('content', []) or [])
        
        formatted_writer_prompt = self.WRITER_PROMPT.format(content=research_summary)
        
        messages_for_writer = [
            SystemMessage(content=formatted_writer_prompt),
            HumanMessage(content=f"Essay Task: {state['task']}\n\nInitial Plan:\n{state['plan']}")
        ]
        
        if state.get('critique') and state.get('draft'): # If revising
            messages_for_writer.append(
                HumanMessage(content=f"\nPrevious Draft:\n{state['draft']}\n\nCritique for Revision:\n{state['critique']}")
            )

        response = self.model.invoke(messages_for_writer)
        current_revision_number = state.get("revision_number", 0)
        return {
            "draft": response.content, 
            "revision_number": current_revision_number + 1,
            "lnode": "generate",
            "count": state.get("count", 0) + 1
        }

    def reflection_node(self, state: AgentState):
        print("\n>> Reflection Node executing...")
        messages = [
            SystemMessage(content=self.REFLECTION_PROMPT), 
            HumanMessage(content=state['draft'])
        ]
        response = self.model.invoke(messages)
        return {
            "critique": response.content,
            "lnode": "reflect",
            "count": state.get("count", 0) + 1
        }

    def research_critique_node(self, state: AgentState):
        print("\n>> Research Critique Node executing...")
        if not state.get('critique'):
            print("  -- No critique found, skipping research for critique.")
            return {
                "lnode": "research_critique",
                "count": state.get("count", 0) + 1,
                "content": state.get('content', []) # Return existing content
            }

        structured_llm_for_queries = self.model.with_structured_output(Queries)
        queries_response = structured_llm_for_queries.invoke([
            SystemMessage(content=self.RESEARCH_CRITIQUE_PROMPT),
            HumanMessage(content=state['critique']) # Research based on the critique
        ])
        
        current_content = state.get('content', []) # Append to existing content
        generated_queries = state.get('queries', []) # Append to existing queries list
        
        if queries_response.queries:
            for q in queries_response.queries:
                generated_queries.append(q) # Add new queries
                print(f"  -- Conducting research for critique query: {q}")
                try:
                    response = self.tavily.search(query=q, max_results=2)
                    for r in response['results']:
                        current_content.append(r['content'])
                except Exception as e:
                    print(f"    !! Error during Tavily search for query '{q}': {e}")
                    current_content.append(f"Error fetching results for: {q}")
        else:
            print("  -- No queries generated by LLM for critique-based research.")
            
        return {
            "content": current_content,
            "queries": generated_queries,
            "lnode": "research_critique",
            "count": state.get("count", 0) + 1
        }
        
    def should_continue(self, state: AgentState):
        current_revision = state.get("revision_number", 0)
        max_rev = state.get("max_revisions", 0)
        print(f"\n>> Should Continue Node? Revision: {current_revision}, Max: {max_rev}")
        if current_revision > max_rev:
            print("   -- Max revisions reached. Ending.")
            return END
        print("   -- Continuing to reflection.")
        return "reflect"

if __name__ == "__main__":
    # This script primarily defines the agent logic.
    # The Gradio UI and invocation would typically be in helper.py or a separate main script.
    print("EssayWriterAgent class defined. To run with UI, use helper.py.")
    
    # Example of direct invocation for testing (without Gradio)
    print("\n--- Direct Invocation Test ---")
    agent_instance = EssayWriterAgent()
    
    test_task = "The impact of AI on modern education"
    initial_state_test = {
        "task": test_task,
        "max_revisions": 1, # 1 draft, 1 revision cycle
        "revision_number": 0,
        "content": [],
        "queries": [],
        "lnode": "",
        "count": 0
    }
    thread_config_test = {"configurable": {"thread_id": "direct_test_thread"}}

    # Simulating the flow manually for one cycle for demonstration
    # In practice, graph.stream or graph.invoke would handle this.
    current_state = initial_state_test
    
    # Planner
    print("\n-- Invoking Planner --")
    current_state.update(agent_instance.plan_node(current_state))
    print(f"Plan: {current_state['plan'][:100]}...")

    # Research Plan
    print("\n-- Invoking Research Plan --")
    current_state.update(agent_instance.research_plan_node(current_state))
    print(f"Research Content Snippets: {len(current_state['content'])}")
    
    # Generate
    print("\n-- Invoking Generate --")
    current_state.update(agent_instance.generation_node(current_state))
    print(f"Draft (Rev {current_state['revision_number']}): {current_state['draft'][:100]}...")

    # Should Continue
    if agent_instance.should_continue(current_state) == "reflect":
        # Reflect
        print("\n-- Invoking Reflect --")
        current_state.update(agent_instance.reflection_node(current_state))
        print(f"Critique: {current_state['critique'][:100]}...")

        # Research Critique
        print("\n-- Invoking Research Critique --")
        current_state.update(agent_instance.research_critique_node(current_state))
        print(f"Updated Research Content Snippets: {len(current_state['content'])}")
        
        # Generate (Revision)
        print("\n-- Invoking Generate (Revision) --")
        current_state.update(agent_instance.generation_node(current_state))
        print(f"Revised Draft (Rev {current_state['revision_number']}): {current_state['draft'][:100]}...")
    
    print("\n--- Direct Invocation Test Complete ---")
    print(f"Final Draft (Rev {current_state['revision_number']}):\n{current_state['draft']}")