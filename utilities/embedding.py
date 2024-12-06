import os
import random
from qdrant_client import QdrantClient, models
import requests
import json
import urllib.request
import urllib.parse
from dotenv import load_dotenv
load_dotenv()

QDRANT_API_KEY = os.environ.get('QDRANT_API_KEY')
HUGGINGFACE_TOKEN = os.environ.get('HUGGINGFACE_TOKEN')
apikey = os.environ.get('GNEWS_API_KEY')


qdrant_client = QdrantClient(
    url="https://291d16b5-08bb-42a4-830a-d27ecc0a4544.us-west-2-0.aws.cloud.qdrant.io:6333", 
    api_key=QDRANT_API_KEY,
)
collection_name="check_embeddings"

from qdrant_client.models import VectorParams, Distance

VECTOR_DIM = 384


try:
    qdrant_client.get_collection(collection_name=collection_name)
    print("Collection already exists.")
except Exception as e:
    print("Creating new collection...")
    qdrant_client.create_collection(
        collection_name=collection_name,  
        vectors_config=VectorParams(
                size=VECTOR_DIM,  
                distance=Distance.COSINE  
        )
    )


API_URL = "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2"
headers = {"Authorization": HUGGINGFACE_TOKEN}

def query(texts):
    response = requests.post(API_URL, headers=headers, json={"inputs": texts, "options":{"wait_for_model":True}})
    return response.json()

def store_embedding(texts, qdrant_client, collection_name=collection_name):
    embedding_response = query(texts)
    if not isinstance(embedding_response, list) or len(embedding_response) != len(texts):
        raise Exception("Mismatch between input texts and embeddings received.")

    random_multiplier = random.randint(100000, 999999)  
    point_ids = [i * random_multiplier for i in range(len(texts))]
    payloads = [{"string": text} for text in texts] 

    response = qdrant_client.upsert(
        collection_name=collection_name,
        points=models.Batch(
            ids=point_ids,
            vectors=embedding_response,
            payloads=payloads,
        ),
    )
    return response


def search_articles(tags, apikey):
    
    
    query = " ".join(tags)
    encoded_query = urllib.parse.quote(query)
    url = f"https://gnews.io/api/v4/search?q={encoded_query}&lang=en&country=us&sortBy=relevance&max=10&apikey={apikey}"

    with urllib.request.urlopen(url) as response:
        data = json.loads(response.read().decode("utf-8"))
        return data  


def search_and_store_embeddings(tags, apikey=apikey, qdrant_client=qdrant_client, collection_name=collection_name):
    response_data = search_articles(tags, apikey)
    
    if "articles" in response_data:
        titles = [article['title'] for article in response_data['articles']]

        store_embedding(titles, qdrant_client, collection_name)
        print(f"Stored {len(titles)} articles in the collection.")

    else:
        print("No articles found.")




