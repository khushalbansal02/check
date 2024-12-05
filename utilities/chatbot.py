import os
from langchain.agents import initialize_agent
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()
GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')
SERPAPI_API_KEY = os.environ.get('SERPAPI_API_KEY')

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    temperature=0,
)

tool_names = ["serpapi"]
tools = load_tools(tool_names)
agent = initialize_agent(tools, llm, agent="zero-shot-react-description")

def get_response(post_title, post_content, user_query, conversation_history):
    prompt = (f"You are a misinformation combating chatbot on a forum website. "
              f"People will make posts and viewers of the post will ask you questions (queries) regarding it. "
              f"Use your own knowledge base or the tools at your disposal to answer.\n\n"
              f"Post Title: {post_title}\n"
              f"Post content: {post_content}\n\n"
              f"Conversation History: {conversation_history}\n\n"
              f"User Query: {user_query}")
    
    return agent.run(prompt)
