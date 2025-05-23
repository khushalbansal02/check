from transformers import pipeline
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
import torch
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

QDRANT_API_KEY = os.environ.get('QDRANT_API_KEY')

# Initialize Qdrant Client
qdrant_client = QdrantClient(
    url="https://3b5215d5-2312-44fa-92c4-6ced8863758a.us-east4-0.gcp.cloud.qdrant.io:6333",
    api_key=QDRANT_API_KEY,
)
collection_name = "check_embeddings"
VECTOR_DIM = 384

# Initialize SentenceTransformer
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Load NLI Model
device = 0 if torch.cuda.is_available() else -1
nli_pipeline = pipeline(
    "text-classification",
    model="facebook/bart-large-mnli",
    tokenizer="facebook/bart-large-mnli",
    device=device,
    return_all_scores=True,
    padding=True,
    truncation=True,
)

def get_embedding(text):
    """
    Generate an embedding for the given text using SentenceTransformer.
    """
    embedding = embedding_model.encode(text, convert_to_numpy=True)
    return embedding.tolist()

def search_in_qdrant(query_embedding, top_k=5):
    """
    Search articles in the Qdrant collection using a query embedding.
    """
    try:
        search_result = qdrant_client.search(
            collection_name=collection_name,
            query_vector=query_embedding,
            limit=top_k
        )
        return search_result
    except Exception as e:
        print(f"Error during Qdrant search: {e}")
        return []

def check_post_authenticity(articles, post):
    """
    Compare a social media post with related articles to check its authenticity.
    Normalize entailment and contradiction scores such that their sum equals 1.
    """
    results = []
    for article in articles:
        outputs = nli_pipeline([(article, post)])[0]  # list of dicts with label and score

        # Extract scores for contradiction and entailment only
        contradiction_score = next(entry['score'] for entry in outputs if entry['label'] == "CONTRADICTION")
        entailment_score = next(entry['score'] for entry in outputs if entry['label'] == "ENTAILMENT")

        # Normalize scores to ensure their sum is 1
        total = contradiction_score + entailment_score
        scaled_contradiction = contradiction_score / total
        scaled_entailment = entailment_score / total

        # Decide the label based on the higher score
        if scaled_entailment > scaled_contradiction:
            label = "Likely Real"
        else:
            label = "Likely Fake"

        results.append({
            'predicted_label': label,
            'probabilities': {
                'class_0': round(scaled_contradiction, 4),
                'class_1': round(scaled_entailment, 4)
            },
            'article': article
        })

    return results


def predict_post_authenticity(post_title, post_content):
    """
    Predict the authenticity of a social media post using related articles.
    """
    post_text = post_title + " " + post_content

    # Generate embedding for the post title
    title_embedding = get_embedding(post_title)

    # Search for related articles in Qdrant
    search_results = search_in_qdrant(title_embedding, top_k=5)
    related_articles = [result.payload["string"] for result in search_results if "string" in result.payload]

    if not related_articles:
        return {
            'error': 'No related articles found for the given post.'
        }

    # Check post authenticity
    authenticity_results = check_post_authenticity(related_articles, post_text)

    return {
        'post_title': post_title,
        'post_content': post_content,
        'authenticity_results': authenticity_results
    }
