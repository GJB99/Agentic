"""
Description (langgraph_human_in_loop.py):
This script focuses on incorporating human oversight into LangGraph agents. 
It defines a custom reduce_messages function to allow updating existing messages in the state by their ID, which is key for modifying past agent decisions. 
The HumanApprovalAgent is configured to interrupt_before its "action" node, pausing execution to allow human review. 
Users can then approve, deny, or modify the proposed tool call. 
Modifications involve updating the agent's state with a revised AIMessage (containing the altered tool call) that shares the ID of the original. 
The script also demonstrates "time travel" by retrieving historical states and "branching" by updating a past state with modified information and resuming execution from that point, creating an alternative conversation history.
"""

from dotenv import load_dotenv
_ = load_dotenv()

from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, List
import operator
from uuid import uuid4

from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, ToolMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.checkpoint.sqlite import SqliteSaver

def reduce_messages(left: List[AnyMessage], right: List[AnyMessage]) -> List[AnyMessage]:
    for message in right:
        if not message.id:
            message.id = str(uuid4())
    merged = left.copy()
    for message in right:
        for i, existing in enumerate(merged):
            if existing.id == message.id:
                merged[i] = message
                break
        else:
            merged.append(message)
    return merged

class AgentState(TypedDict):
    messages: Annotated[List[AnyMessage], reduce_messages]

tool = TavilySearchResults(max_results=2)

class HumanApprovalAgent:
    def __init__(self, model, tools, system="", checkpointer=None):
        self.system = system
        graph = StateGraph(AgentState)
        graph.add_node("llm", self.call_openai)
        graph.add_node("action", self.take_action)
        graph.add_conditional_edges("llm", self.exists_action, {True: "action", False: END})
        graph.add_edge("action", "llm")
        graph.set_entry_point("llm")
        self.graph = graph.compile(
            checkpointer=checkpointer,
            interrupt_before=["action"]
        )
        self.tools = {t.name: t for t in tools}
        self.model = model.bind_tools(tools)

    def call_openai(self, state: AgentState):
        messages = state['messages']
        if self.system:
            messages_for_llm = [SystemMessage(content=self.system)] + messages
        else:
            messages_for_llm = messages
        print("\n>> Calling LLM...")
        message = self.model.invoke(messages_for_llm)
        return {'messages': [message]}

    def exists_action(self, state: AgentState):
        result = state['messages'][-1]
        if isinstance(result, AIMessage) and hasattr(result, 'tool_calls') and result.tool_calls:
            print("LLM requested an action. Interrupting for approval.")
            return True
        print("LLM did not request an action. Ending.")
        return False

    def take_action(self, state: AgentState):
        tool_calls = state['messages'][-1].tool_calls
        results = []
        for t in tool_calls:
            print(f"\n>> Approved! Calling Tool: {t['name']} with args {t['args']}")
            try:
                result_content = self.tools[t['name']].invoke(t['args'])
            except Exception as e:
                result_content = f"Error: Failed to execute tool {t['name']}. Reason: {str(e)}"
            results.append(ToolMessage(tool_call_id=t['id'], name=t['name'], content=str(result_content)))
        print("\n>> Action Results sent back to LLM.")
        return {'messages': results}

if __name__ == "__main__":
    memory = SqliteSaver.from_conn_string(":memory:")
    prompt_template = """You are a smart research assistant. Use the search engine to look up information. \
You are allowed to make multiple calls (either together or in sequence). \
Only look up information when you are sure of what you want. \
If you need to look up some information before asking a follow up question, you are allowed to do that!
"""
    model = ChatOpenAI(model="gpt-3.5-turbo")
    abot = HumanApprovalAgent(model, [tool], system=prompt_template, checkpointer=memory)

    print("\n--- Manual Human Approval Example ---")
    thread_id_approval = "thread_human_approval_1"
    thread_config_approval = {"configurable": {"thread_id": thread_id_approval}}
    initial_messages = [HumanMessage(content="Whats the weather in SF?")]
    
    for event in abot.graph.stream({"messages": initial_messages}, thread_config_approval):
        for node_name, output in event.items():
            if node_name != "__end__":
                print(f"Output from node '{node_name}':")
                if 'messages' in output and output['messages']:
                    for m in output['messages']:
                        if not m.id: m.id = str(uuid4())
                    print(f"  Last message added: {output['messages'][-1].pretty_repr(html=False)[:300]}...")
    
    current_snapshot = abot.graph.get_state(thread_config_approval)
    print(f"\nGraph interrupted. Next node: {current_snapshot.next}")

    if current_snapshot.next == ('action',):
        last_ai_message = current_snapshot.values['messages'][-1]
        if isinstance(last_ai_message, AIMessage) and last_ai_message.tool_calls:
            print(f"AI wants to call tool: {last_ai_message.tool_calls[0]['name']} with args: {last_ai_message.tool_calls[0]['args']}")
            proceed = input("Proceed with this action? (y/n/modify): ").lower()
            if proceed == 'y':
                for event in abot.graph.stream(None, thread_config_approval): # Continue
                    # Process event
                    pass
            elif proceed == 'modify':
                new_query = input("Enter new query for tavily_search_results_json: ")
                modified_tool_calls = last_ai_message.tool_calls.copy()
                modified_tool_calls[0]['args'] = {'query': new_query}
                modified_ai_message = AIMessage(content=last_ai_message.content, tool_calls=modified_tool_calls, id=last_ai_message.id)
                abot.graph.update_state(thread_config_approval, {"messages": [modified_ai_message]})
                for event in abot.graph.stream(None, thread_config_approval): # Continue from modified state
                    # Process event
                    pass
    # ... (rest of the example interaction and time travel logic) ...
    # (Simplified for brevity as the core mechanism is shown)

    # Time Travel / Branching demonstration part
    print("\n--- Time Travel and Branching Example ---")
    abot_no_interrupt_for_history = HumanApprovalAgent(model, [tool], system=prompt_template, checkpointer=memory)
    abot_no_interrupt_for_history.graph = StateGraph(AgentState).add_node("llm", abot_no_interrupt_for_history.call_openai) \
        .add_node("action", abot_no_interrupt_for_history.take_action) \
        .add_conditional_edges("llm", abot_no_interrupt_for_history.exists_action, {True: "action", False: END}) \
        .add_edge("action", "llm").set_entry_point("llm").compile(checkpointer=memory) # No interrupt

    thread_id_timetravel = "thread_timetravel_1"
    config_timetravel = {"configurable": {"thread_id": thread_id_timetravel}}
    abot_no_interrupt_for_history.graph.invoke({"messages": [HumanMessage(content="Weather in London?")]}, config_timetravel)
    abot_no_interrupt_for_history.graph.invoke({"messages": [HumanMessage(content="And in Paris?")]}, config_timetravel)
    history = list(abot_no_interrupt_for_history.graph.get_state_history(config_timetravel))

    if len(history) > 3: # Need enough states to go back
        # Example: Branch from state before the last action was taken (e.g., after LLM call for Paris)
        # This is a heuristic; robust selection would inspect state contents.
        # history[0] is latest. history[-1] is oldest.
        # Let's target a state that was an LLM output proposing a tool call.
        state_to_branch_from = None
        for s in reversed(history): # Search from oldest
            if s.next == ('action',): # State where LLM proposed an action
                state_to_branch_from = s
                break
        
        if state_to_branch_from:
            print(f"\nBranching from state: {state_to_branch_from.config['configurable']['thread_ts']}")
            ai_message_to_modify = state_to_branch_from.values['messages'][-1]
            if isinstance(ai_message_to_modify, AIMessage) and ai_message_to_modify.tool_calls:
                tool_call_id = ai_message_to_modify.tool_calls[0]['id']
                modified_message_for_branch = AIMessage(
                    content=ai_message_to_modify.content,
                    tool_calls=[{'name': 'tavily_search_results_json', 'args': {'query': 'current weather in Berlin'}, 'id': tool_call_id}],
                    id=ai_message_to_modify.id
                )
                branch_config = abot_no_interrupt_for_history.graph.update_state(
                    state_to_branch_from.config, # Branch from this specific point in time
                    {"messages": [modified_message_for_branch]}
                )
                print("\nStreaming from the new branch (weather in Berlin):")
                for event in abot_no_interrupt_for_history.graph.stream(None, branch_config):
                    # Process event
                    pass
                final_branched_state = abot_no_interrupt_for_history.graph.get_state(branch_config)
                print(f"Final AI response (branched): {final_branched_state.values['messages'][-1].content}")