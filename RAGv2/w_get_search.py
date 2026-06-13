import requests
from bs4 import BeautifulSoup
import os
from dotenv import load_dotenv
load_dotenv()

#Google Search
#API_KEY- https://developers.google.com/custom-search/v1/overview
#CSE_ID- https://programmablesearchengine.google.com/controlpanel/all
def google_search(query, num_results=5):
    # Set GOOGLE_API_KEY and GOOGLE_CSE_ID in your environment or a .env file.
    API_KEY = os.getenv("GOOGLE_API_KEY")
    CSE_ID = os.getenv("GOOGLE_CSE_ID")

    """Fetch search results from Google Custom Search API."""
    url = f"https://www.googleapis.com/customsearch/v1"
    params = {
        "key": API_KEY,
        "cx": CSE_ID,
        "q": query,
        "num": num_results
    }
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, params=params, headers=headers)
    results = response.json()
    
    search_results = []
    if "items" in results: 
        for item in results["items"]:
            search_results.append(item["link"])
    
    return search_results

def duckduckgo_search(query,num_results=5):
    url = f"https://duckduckgo.com/html/?q={query.replace(' ', '+')}"
    headers = {"User-Agent": "Mozilla/5.0"}
    
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, "html.parser")
    
    results = soup.select("a.result__url")
    #print(results)
    search_results = [f"https://{result.get_text().strip()}" for result in results[:num_results]]
    
    return search_results

def brave_search(query,num_results=5):
    url = f"https://search.brave.com/search?q={query.replace(' ', '+')}"
    headers = {"User-Agent": "Mozilla/5.0", "Accept-Encoding": "gzip, deflate"}

    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, "html.parser")

    results = soup.select("a.result-header")
    return [result["href"] for result in results[:5]]


def bing_search(query,num_results=5):
    url = f"https://www.bing.com/search?q={query.replace(' ', '+')}"
    headers = {"User-Agent": "Mozilla/5.0"}

    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, "html.parser")

    results = soup.select("li.b_algo h2 a")
    return [result["href"] for result in results[:5]]


def yahoo_search(query,num_results=5):
    url = f"https://search.yahoo.com/search?p={query.replace(' ', '+')}"
    headers = {"User-Agent": "Mozilla/5.0"}

    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, "html.parser")

    results = soup.select("h3.title a")
    return [result["href"] for result in results[:5]]


def startpage_search(query,num_results=5):
    url = f"https://www.startpage.com/sp/search?q={query.replace(' ', '+')}"
    headers = {"User-Agent": "Mozilla/5.0"}

    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, "html.parser")

    results = soup.select("a.result-link")
    return [result["href"] for result in results[:5]]

def get_search(query,searcher="duckDuckGo",num_results=5):
    if(searcher=="google"):
        return(google_search(query,num_results=num_results))
    elif(searcher=="duckDuckGo"):
        return(duckduckgo_search(query,num_results=num_results))
    elif(searcher=="brave"):
        return(brave_search(query,num_results=num_results))
    elif(searcher=="bing"):
        return(bing_search(query,num_results=num_results))
    elif(searcher=="yahoo"):
        return(yahoo_search(query,num_results=num_results))
    elif(searcher=="startpage"):
        return(startpage_search(query,num_results=num_results))
    else:
        return(duckduckgo_search(query,num_results=num_results))
# Example Usage
#query = "Who are the main competitors of eCADSTAR?"
#results = google_search(query)

#print(results)

# Example usage:
#query = "Today's news in india"
#links = get_duckduckgo_results(query)

#print(links)