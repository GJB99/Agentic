"""
Description (langgraph_basic_agent.py):
This script introduces LangGraph for building a research agent. 
It defines an AgentState to manage the conversation messages. 
The Agent class constructs a StateGraph with two primary nodes: "llm" (for calling the OpenAI model, which is bound to the Tavily search tool) and "action" (for executing the requested Tavily search). 
Conditional edges route the flow: if the LLM requests a tool, the graph transitions to the "action" node; otherwise, it ends. 
After an action, the flow returns to the LLM node with the tool's results. 
This demonstrates a more structured, declarative approach to defining agent behavior compared to the manual loop in the first part.

"""

from dotenv import load_dotenv
_ = load_dotenv()

from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, ToolMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults

# Initialize the tool
tool = TavilySearchResults(max_results=4)

# Define the state for our agent
class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]

class Agent:
    def __init__(self, model, tools, system=""):
        self.system = system
        graph = StateGraph(AgentState)
        graph.add_node("llm", self.call_openai)
        graph.add_node("action", self.take_action)
        graph.add_conditional_edges(
            "llm",
            self.exists_action,
            {True: "action", False: END} # Route to 'action' if tool call exists, else to END
        )
        graph.add_edge("action", "llm") # After action, go back to llm
        graph.set_entry_point("llm")
        self.graph = graph.compile()
        self.tools = {t.name: t for t in tools}
        self.model = model.bind_tools(tools) # Bind tools for function calling

    def exists_action(self, state: AgentState):
        # Check the last message from the LLM for tool calls
        result = state['messages'][-1]
        if isinstance(result, AIMessage) and hasattr(result, 'tool_calls') and result.tool_calls:
            return True # There are tool calls
        return False # No tool calls

    def call_openai(self, state: AgentState):
        messages = state['messages']
        if self.system:
            # Prepend system message if provided
            messages_for_llm = [SystemMessage(content=self.system)] + messages
        else:
            messages_for_llm = messages
        
        print("\n>> Calling LLM...")
        message = self.model.invoke(messages_for_llm)
        return {'messages': [message]}

    def take_action(self, state: AgentState):
        tool_calls = state['messages'][-1].tool_calls
        results = []
        for t in tool_calls:
            print(f"\n>> Calling Tool: {t['name']} with args {t['args']}")
            if not t['name'] in self.tools:
                print("\n ....bad tool name....")
                results.append(ToolMessage(tool_call_id=t['id'], name=t['name'], content="Error: Tool not found. Please retry with a valid tool name."))
            else:
                try:
                    result_content = self.tools[t['name']].invoke(t['args'])
                except Exception as e:
                    print(f"Error invoking tool {t['name']}: {e}")
                    result_content = f"Error: Failed to execute tool {t['name']}. Reason: {str(e)}"
                results.append(ToolMessage(tool_call_id=t['id'], name=t['name'], content=str(result_content)))
        
        print("\n>> Action Results sent back to LLM.")
        return {'messages': results}

if __name__ == "__main__":
    prompt_template = """You are a smart research assistant. Use the search engine to look up information. \
You are allowed to make multiple calls (either together or in sequence). \
Only look up information when you are sure of what you want. \
If you need to look up some information before asking a follow up question, you are allowed to do that!
"""
    model = ChatOpenAI(model="gpt-4o")
    abot = Agent(model, [tool], system=prompt_template)

    print("\n--- Test 1: Weather in SF ---")
    messages_sf = [HumanMessage(content="What is the weather in sf?")]
    result_sf = abot.graph.invoke({"messages": messages_sf})
    if result_sf['messages'] and isinstance(result_sf['messages'][-1], AIMessage):
        print(f"Final Answer (SF): {result_sf['messages'][-1].content}")

    print("\n--- Test 2: Weather in SF and LA (parallel tool calls) ---")
    messages_sf_la = [HumanMessage(content="What is the weather in SF and LA?")]
    result_sf_la = abot.graph.invoke({"messages": messages_sf_la})
    if result_sf_la['messages'] and isinstance(result_sf_la['messages'][-1], AIMessage):
        print(f"Final Answer (SF & LA): {result_sf_la['messages'][-1].content}")

    print("\n--- Test 3: Multi-hop question (Super Bowl) ---")
    query_super_bowl = "Who won the super bowl in 2024? In what state is the winning team headquarters located? \
What is the GDP of that state? Answer each question."
    messages_super_bowl = [HumanMessage(content=query_super_bowl)]
    result_super_bowl = abot.graph.invoke({"messages": messages_super_bowl})
    if result_super_bowl['messages'] and isinstance(result_super_bowl['messages'][-1], AIMessage):
         print(f"Final Answer (Super Bowl): {result_super_bowl['messages'][-1].content}")