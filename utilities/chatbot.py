import os
from langchain.agents import initialize_agent
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain_google_genai import ChatGoogleGenerativeAI
from qdrant_client import QdrantClient, models 
from dotenv import load_dotenv
import numpy as np
from sentence_transformers import SentenceTransformer  # new import
load_dotenv()

GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')
SERPAPI_API_KEY = os.environ.get('SERPAPI_API_KEY')
QDRANT_API_KEY = os.environ.get('QDRANT_API_KEY')

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0,
)

tool_names = ["serpapi"]
tools = load_tools(tool_names)
agent = initialize_agent(tools, llm, agent="zero-shot-react-description")

qdrant_client = QdrantClient(
    url="https://3b5215d5-2312-44fa-92c4-6ced8863758a.us-east4-0.gcp.cloud.qdrant.io:6333", 
    api_key=QDRANT_API_KEY,
)

collection_name = "check_embeddings"
VECTOR_DIM = 384

model = SentenceTransformer('all-MiniLM-L6-v2')

def get_embedding(text):
    embedding = model.encode(text, convert_to_numpy=True)
    return embedding.tolist()  

def search_in_qdrant(query_embedding, top_k=5):
    try:
        # Log the query embedding for debugging
        print("Query embedding length:", len(query_embedding))
        print("Query embedding (first 10 values):", query_embedding[:10])

        # Perform the search
        search_result = qdrant_client.search(
            collection_name=collection_name,
            query_vector=query_embedding,
            limit=top_k
        )
        
        # Log the search result
        print("Search results:", search_result)
        return search_result

    except Exception as e:
        # Log any errors
        print(f"Error during Qdrant search: {e}")
        return []


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
    # print(prompt)
    # return "Check"
    return agent.run(prompt)

