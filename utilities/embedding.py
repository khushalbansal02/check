import os
import random
from qdrant_client import QdrantClient, models
import requests
import json
import urllib.request
import urllib.parse
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
load_dotenv()

QDRANT_API_KEY = os.environ.get('QDRANT_API_KEY')
apikey = os.environ.get('GNEWS_API_KEY')


# Load model locally (this will download the model the first time)
model = SentenceTransformer('all-MiniLM-L6-v2')


qdrant_client = QdrantClient(
    url="https://3b5215d5-2312-44fa-92c4-6ced8863758a.us-east4-0.gcp.cloud.qdrant.io:6333", 
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

def query(texts):
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    return embeddings.tolist()


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



# points, _ = qdrant_client.scroll(
#     collection_name="check_embeddings",
#     scroll_filter=None,
#     limit=100  # Fetch up to 100 points at a time
# )

# # Check if points are retrieved
# if not points:
#     print("No points found in the collection.")
# else:
#     for point in points:
#         print(f"ID: {point.id}, Payload: {point.payload}")

# qdrant_client.delete(
#     collection_name=collection_name,
#     # No filter means delete all points
#     filter=None
# )

# print(f"All points deleted from collection '{collection_name}'.")





