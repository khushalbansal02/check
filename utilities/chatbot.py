import os
from langchain.agents import initialize_agent
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain_google_genai import ChatGoogleGenerativeAI
from qdrant_client import QdrantClient, models
from dotenv import load_dotenv
import requests
import numpy as np


load_dotenv()

GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')
SERPAPI_API_KEY = os.environ.get('SERPAPI_API_KEY')
QDRANT_API_KEY = os.environ.get('QDRANT_API_KEY')
HUGGINGFACE_TOKEN = os.environ.get('HUGGINGFACE_TOKEN')


llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    temperature=0,
)

tool_names = ["serpapi"]
tools = load_tools(tool_names)
agent = initialize_agent(tools, llm, agent="zero-shot-react-description")


qdrant_client = QdrantClient(
    url="https://291d16b5-08bb-42a4-830a-d27ecc0a4544.us-west-2-0.aws.cloud.qdrant.io:6333", 
    api_key=QDRANT_API_KEY,
)

collection_name = "check_embeddings"
VECTOR_DIM = 384  

def get_embedding(text, model_url="https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2"):
    headers = {"Authorization": f"Bearer {HUGGINGFACE_TOKEN}"}
    response = requests.post(model_url, headers=headers, json={"inputs": text, "options": {"wait_for_model": True}})
    if response.status_code == 200:
        return response.json()[0]  
    else:
        raise Exception(f"Error fetching embedding: {response.status_code}, {response.text}")

def search_in_qdrant(query_embedding, top_k=5):
    search_result = qdrant_client.search(
        collection_name=collection_name,
        query_vector=query_embedding,
        limit=top_k
    )
    return search_result


def get_response(post_title, post_content, user_query, conversation_history):
    
    title_embedding = get_embedding(post_title)
    search_result = search_in_qdrant(title_embedding)  
    relevant_context = post_content  
    
    if search_result:
        relevant_context = "\n".join([hit.payload["string"] for hit in search_result])  
    
    prompt = (f"You are a misinformation combating chatbot on a forum website. "
              f"People will make posts and viewers of the post will ask you questions (queries) regarding it. "
              f"Use your own knowledge base or the tools at your disposal to answer.\n\n"
              f"Post Title: {post_title}\n"
              f"Post content: {relevant_context}\n\n"
              f"Conversation History: {conversation_history}\n\n"
              f"User Query: {user_query}")
    
    
    return agent.run(prompt)
