"""
Description (langgraph_persistence_streaming.py):
This script demonstrates state persistence and response streaming in LangGraph agents. 
The AgentWithPersistence class compiles its graph with a checkpointer (e.g., SqliteSaver or AsyncSqliteSaver), enabling agent state to be saved and retrieved. 
The run_sync_persistence_example function showcases multi-turn conversations where the agent maintains context across interactions within the same thread_id, and demonstrates conversation isolation using different thread IDs. 
The run_async_streaming_example function uses astream_events to stream LLM response tokens in real-time and also logs tool execution events, enhancing responsiveness and observability.
"""

from dotenv import load_dotenv
_ = load_dotenv()

from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, ToolMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.aiosqlite import AsyncSqliteSaver
import asyncio

tool = TavilySearchResults(max_results=2)

class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]

class AgentWithPersistence:
    def __init__(self, model, tools, checkpointer, system=""):
        self.system = system
        graph = StateGraph(AgentState)
        graph.add_node("llm", self.call_openai)
        graph.add_node("action", self.take_action)
        graph.add_conditional_edges("llm", self.exists_action, {True: "action", False: END})
        graph.add_edge("action", "llm")
        graph.set_entry_point("llm")
        self.graph = graph.compile(checkpointer=checkpointer)
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
            return True
        return False

    def take_action(self, state: AgentState):
        tool_calls = state['messages'][-1].tool_calls
        results = []
        for t in tool_calls:
            print(f"\n>> Calling Tool: {t['name']} with args {t['args']}")
            try:
                result_content = self.tools[t['name']].invoke(t['args'])
            except Exception as e:
                result_content = f"Error: Failed to execute tool {t['name']}. Reason: {str(e)}"
            results.append(ToolMessage(tool_call_id=t['id'], name=t['name'], content=str(result_content)))
        print("\n>> Action Results sent back to LLM.")
        return {'messages': results}

async def run_sync_persistence_example():
    print("\n--- Synchronous Persistence Example ---")
    memory_sync = SqliteSaver.from_conn_string(":memory:")
    prompt_template = """You are a smart research assistant. Use the search engine to look up information. \
You are allowed to make multiple calls (either together or in sequence). \
Only look up information when you are sure of what you want. \
If you need to look up some information before asking a follow up question, you are allowed to do that!
"""
    model = ChatOpenAI(model="gpt-4o")
    abot_sync = AgentWithPersistence(model, [tool], system=prompt_template, checkpointer=memory_sync)

    thread1_id = "thread_1_sync"
    thread2_id = "thread_2_sync"

    print(f"\n--- Conversation with Thread ID: {thread1_id} ---")
    messages_sf = [HumanMessage(content="What is the weather in sf?")]
    thread_config_1 = {"configurable": {"thread_id": thread1_id}}
    
    print("\nInitial call for SF weather:")
    for event in abot_sync.graph.stream({"messages": messages_sf}, thread_config_1):
        for node_name, output in event.items():
            if node_name != "__end__":
                print(f"Output from node '{node_name}':")
                if 'messages' in output and output['messages']:
                    print(f"  {output['messages'][-1].pretty_repr(html=False)[:500]}...")
    
    final_state_1_part1 = abot_sync.graph.get_state(thread_config_1)
    if final_state_1_part1 and final_state_1_part1.values['messages']:
        print(f"\nAI Response (SF): {final_state_1_part1.values['messages'][-1].content}")

    print(f"\n--- Continuing Conversation with Thread ID: {thread1_id} ---")
    messages_la = [HumanMessage(content="What about in la?")]
    for event in abot_sync.graph.stream({"messages": messages_la}, thread_config_1): # Stream events
        for node_name, output in event.items():
            if node_name != "__end__":
                print(f"Output from node '{node_name}':")
                if 'messages' in output and output['messages']:
                    print(f"  {output['messages'][-1].pretty_repr(html=False)[:500]}...")

    final_state_1_part2 = abot_sync.graph.get_state(thread_config_1)
    if final_state_1_part2 and final_state_1_part2.values['messages']:
        print(f"\nAI Response (LA): {final_state_1_part2.values['messages'][-1].content}")

    print(f"\n--- Final Question with Thread ID: {thread1_id} ---")
    messages_warmer = [HumanMessage(content="Which one is warmer?")]
    for event in abot_sync.graph.stream({"messages": messages_warmer}, thread_config_1):
         for node_name, output in event.items():
            if node_name != "__end__":
                print(f"Output from node '{node_name}':")
                if 'messages' in output and output['messages']:
                    print(f"  {output['messages'][-1].pretty_repr(html=False)[:500]}...")
    
    final_state_1_part3 = abot_sync.graph.get_state(thread_config_1)
    if final_state_1_part3 and final_state_1_part3.values['messages']:
        print(f"\nAI Response (Comparison): {final_state_1_part3.values['messages'][-1].content}")

    print(f"\n--- New Conversation with Thread ID: {thread2_id} (to show isolation) ---")
    messages_warmer_new_thread = [HumanMessage(content="Which one is warmer?")]
    thread_config_2 = {"configurable": {"thread_id": thread2_id}}
    for event in abot_sync.graph.stream({"messages": messages_warmer_new_thread}, thread_config_2):
        for node_name, output in event.items():
            if node_name != "__end__":
                print(f"Output from node '{node_name}':")
                if 'messages' in output and output['messages']:
                    print(f"  {output['messages'][-1].pretty_repr(html=False)[:500]}...")
    
    final_state_2 = abot_sync.graph.get_state(thread_config_2)
    if final_state_2 and final_state_2.values['messages']:
        print(f"\nAI Response (New Thread): {final_state_2.values['messages'][-1].content}")

async def run_async_streaming_example():
    print("\n\n--- Asynchronous Token Streaming Example ---")
    memory_async = AsyncSqliteSaver.from_conn_string(":memory:")
    prompt_template = """You are a smart research assistant. Use the search engine to look up information. \
You are allowed to make multiple calls (either together or in sequence). \
Only look up information when you are sure of what you want. \
If you need to look up some information before asking a follow up question, you are allowed to do that!
"""
    model = ChatOpenAI(model="gpt-4o")
    abot_async = AgentWithPersistence(model, [tool], system=prompt_template, checkpointer=memory_async)

    thread_id_streaming = "thread_streaming_async"
    messages_streaming = [HumanMessage(content="What is the weather in SF?")]
    thread_config_streaming = {"configurable": {"thread_id": thread_id_streaming}}

    print(f"\nStreaming response for weather in SF (Thread: {thread_id_streaming}):")
    full_response_content = ""
    async for event in abot_async.graph.astream_events({"messages": messages_streaming}, thread_config_streaming, version="v1"):
        kind = event["event"]
        if kind == "on_chat_model_stream":
            content = event["data"]["chunk"].content
            if content:
                print(content, end="", flush=True)
                full_response_content += content
        elif kind == "on_tool_start":
            print(f"\n>> Starting tool: {event['name']} with input {event['data'].get('input')}", flush=True)
        elif kind == "on_tool_end":
            print(f"\n<< Finished tool: {event['name']}", flush=True)
    print("\n--- End of streaming ---")
    print(f"Full streamed response content: {full_response_content}")

if __name__ == "__main__":
    asyncio.run(run_sync_persistence_example())
    asyncio.run(run_async_streaming_example())