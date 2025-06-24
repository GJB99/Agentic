"""
Description (traditional_search_example.py):
This script illustrates the conventional approach to web information retrieval: searching and scraping. 
It uses the duckduckgo_search library to obtain a list of URLs for a given query. 
Subsequently, it employs requests to fetch the HTML content of a selected URL and BeautifulSoup to parse this HTML. 
Text is extracted from common HTML elements and cleaned using regular expressions. 
This process is contrasted with agentic search to emphasize the additional steps and potential fragility involved in traditional web scraping for agent data gathering.
"""

import requests
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS
import re
from dotenv import load_dotenv

_ = load_dotenv()

ddg = DDGS()

def search_ddg(query, max_results=3):
    print(f"\n>> Searching DDG for: {query}")
    try:
        results = ddg.text(query, max_results=max_results)
        return [i["href"] for i in results if "href" in i]
    except Exception as e:
        print(f"Error during DuckDuckGo search: {e}")
        print("Returning fallback results due to DDG search exception.")
        return [
            "https://weather.com/weather/today/l/USCA0987:1:US",
            "https://www.accuweather.com/en/us/san-francisco-ca/94103/weather-forecast/347629"
        ]

def scrape_webpage_content(url):
    if not url:
        return "URL was empty, cannot scrape."
    print(f"\n>> Scraping URL: {url}")
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        return f"Failed to retrieve the webpage {url}. Error: {e}"

    soup = BeautifulSoup(response.text, 'html.parser')
    texts = []
    for tag in soup.find_all(['h1', 'h2', 'h3', 'p', 'title']):
        text_content = tag.get_text(" ", strip=True)
        if text_content:
            texts.append(text_content)
    
    combined_text = "\n".join(texts)
    cleaned_text = re.sub(r'\s*\n\s*', '\n', combined_text).strip()
    cleaned_text = re.sub(r' +', ' ', cleaned_text)
    return cleaned_text[:5000] if cleaned_text else "No relevant text found or extracted."

if __name__ == "__main__":
    city = "San Francisco"
    query_traditional = f"current weather in {city} site:weather.com OR site:accuweather.com"

    print("\n--- Traditional Search and Scrape Example ---")
    urls = search_ddg(query_traditional, max_results=1)

    if urls:
        print("\nFound URLs:")
        for u in urls:
            print(u)
        first_url = urls[0]
        scraped_content = scrape_webpage_content(first_url)
        print(f"\nScraped Content from {first_url} (first 500 chars):\n")
        print(scraped_content[:500] + "..." if len(scraped_content) > 500 else scraped_content)
    else:
        print("No URLs found by traditional search.")