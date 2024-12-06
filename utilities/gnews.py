import json
import urllib.request
import os
import urllib.parse

def search_articles(tags, apikey):
    if not apikey:
        raise ValueError("API key is required.")
    
    query = " ".join(tags)
    
    encoded_query = urllib.parse.quote(query)
    
    url = f"https://gnews.io/api/v4/search?q={encoded_query}&lang=en&country=us&sortBy=relevance&max=10&apikey={apikey}"

    with urllib.request.urlopen(url) as response:
        data = json.loads(response.read().decode("utf-8"))
        return data  


apikey = "641f22d599ff946aa3dacd57b7d3ee09"  
tags = ["Bitcoin"]  

response_data = search_articles(tags, apikey)

if "articles" in response_data:
    for article in response_data["articles"]:
        print(f"Title: {article['title']}")
        print(f"Description: {article['description']}")
        print(f"URL: {article['url']}")

else:
    print("No articles found.")
