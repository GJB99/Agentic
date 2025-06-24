"""
Description (tavily_agentic_search_example.py):
This script showcases the use of Tavily for "agentic search." 
It connects to the Tavily API and performs two types of searches. 
The first example fetches a direct answer for a query about Nvidia's Blackwell GPU. 
The second example attempts to retrieve structured weather data for San Francisco, demonstrating how Tavily can return JSON-like string content that can be parsed and utilized directly by an agent. 
The script uses pygments to pretty-print the resulting JSON for enhanced readability, highlighting the agent-friendly nature of such search results.
"""

from dotenv import load_dotenv
import os
from tavily import TavilyClient
import json
from pygments import highlight, lexers, formatters

_ = load_dotenv()

client = TavilyClient(api_key=os.environ.get("TAVILY_API_KEY"))

print("\n--- Example 1: Nvidia Blackwell GPU ---")
result_nvidia = client.search("What is in Nvidia's new Blackwell GPU?",
                              include_answer=True)
print("Tavily Answer:")
print(result_nvidia.get("answer", "No direct answer found."))

print("\n--- Example 2: Weather in San Francisco (Agentic Search) ---")
city = "San Francisco"
query_weather = f"what is the current weather in {city}? provide structured data."

weather_search_results = client.search(query_weather, search_depth="advanced", max_results=1, include_raw_content=False, include_domains=[])

if weather_search_results and weather_search_results.get("results"):
    data_string = weather_search_results["results"][0]["content"]
    print("\nRaw Content from Tavily (first result):")
    print(data_string)
    try:
        parsed_json = json.loads(data_string.replace("'", "\""))
        formatted_json = json.dumps(parsed_json, indent=4)
        colorful_json = highlight(formatted_json,
                                  lexers.JsonLexer(),
                                  formatters.TerminalFormatter())
        print("\nPretty Printed JSON (Weather):")
        print(colorful_json)
    except json.JSONDecodeError as e:
        print(f"\nCould not parse the content as JSON: {e}")
    except Exception as e:
        print(f"\nAn error occurred during JSON processing: {e}")
else:
    print("\nNo results found for the weather query or an error occurred.")