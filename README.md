# Project: AI Agents with LangGraph
This repository contains a collection of Python scripts demonstrating the construction of AI agents, from basic implementations to more sophisticated versions leveraging the LangGraph framework. The project explores concepts such as the ReAct pattern, agentic search, state persistence, streaming, human-in-the-loop interactions, and culminates in a multi-step essay writing agent.

## Explanation project
This project comprehensively explores AI agent development, commencing with a foundational Python-based ReAct agent to illustrate core reasoning and action loops using basic tools and OpenAI's LLM. It then transitions to LangGraph, demonstrating how to construct more controllable agents capable of leveraging Tavily for efficient agentic search, contrasting this with traditional web scraping techniques using DuckDuckGo, Requests, and BeautifulSoup. Subsequent parts enhance these LangGraph agents by incorporating SQLite-backed persistence for stateful, multi-turn conversations and real-time streaming for improved user interaction. The project further delves into advanced control mechanisms, enabling human-in-the-loop oversight for action approval, state modification, and "time-travel" debugging by revisiting and branching from past execution states. Finally, these concepts culminate in the development of a sophisticated, multi-step essay writing agent that iteratively plans, researches, drafts, and refines content, all orchestrated by LangGraph and presented through an interactive Gradio user interface.

## .env.example
OPENAI_API_KEY="your_openai_api_key_here"
TAVILY_API_KEY="your_tavily_api_key_here"

Purpose: This acts as a template for setting up necessary environment variables. To use the project, create a copy named .env in the project root and populate it with your actual API keys for OpenAI and Tavily. This method ensures that sensitive credentials are not hardcoded into the source code.

## requirements.txt
openai
python-dotenv
httpx
langgraph
langchain
langchain-core
langchain-openai
langchain-community
tavily-python
requests
beautifulsoup4
duckduckgo_search
pygments
gradio
# Ensure you have a C compiler and SQLite development libraries for aiosqlite/sqlite persistence
# e.g., on Debian/Ubuntu: sudo apt-get install build-essential libsqlite3-dev

Purpose: This lists all Python package dependencies required to execute the scripts within this project. Users can install these dependencies by running pip install -r requirements.txt in their Python environment.
