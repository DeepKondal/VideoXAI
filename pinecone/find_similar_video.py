import os
import json
import numpy as np
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer

# ✅ Load Pre-trained Embedding Model (512D output)
# embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
# embedding_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")  # ✅ 512D Model
embedding_model = SentenceTransformer("sentence-transformers/paraphrase-mpnet-base-v2")

# ✅ Pinecone API Key (Replace with your actual API key)
PINECONE_API_KEY = "Secret_Key"

# ✅ Create a Pinecone instance
pc = Pinecone(api_key=PINECONE_API_KEY)

# ✅ Define Pinecone index names
original_index_name = "original-video-db"
adversarial_index_name = "adversarial-video-db"

# ✅ Create Pinecone indexes if they don't exist
for index_name in [original_index_name, adversarial_index_name]:
    existing_indexes = pc.list_indexes().names()
    if index_name not in existing_indexes:
        pc.create_index(
            name=index_name,
            dimension=768,  # ✅ Using 512D vectors for embeddings
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
        print(f"✅ Created Pinecone index: {index_name}")
    else:
        print(f"⚡ Pinecone index '{index_name}' already exists.")

# ✅ Connect to Pinecone indexes
original_index = pc.Index(original_index_name)
adversarial_index = pc.Index(adversarial_index_name)
print("✅ Successfully connected to Pinecone!")

# ✅ Define the base directories
BASE_DIR = "output"
CLEAN_DIR = os.path.join(BASE_DIR, "clean")
ADVERSARIAL_DIR = os.path.join(BASE_DIR, "adversarial")

# ✅ Function to generate embeddings from JSON attention data
def generate_embedding(attention_data):
    """ Converts attention JSON data into a 512D embedding vector. """
    text_representation = json.dumps(attention_data)  # Convert JSON to a string
    embedding = embedding_model.encode(text_representation)  # Generate embedding (512D)
    return embedding.tolist()  # Convert to list for Pinecone


def find_similar_videos(json_path, data_type, top_k=5):
    """
    Finds the most similar videos based on attention embeddings.
    
    - json_path: Path to the new video's JSON attention data.
    - data_type: 'original' or 'adversarial' (which database to search in).
    - top_k: Number of similar videos to return.
    """
    if not os.path.exists(json_path):
        print(f"❌ JSON file not found: {json_path}")
        return None

    # Load JSON attention data
    with open(json_path, 'r') as f:
        attention_data = json.load(f)

    # ✅ Generate embedding for the new video
    query_vector = generate_embedding(attention_data)

    # ✅ Select the correct index (original or adversarial)
    index = original_index if data_type == "original" else adversarial_index

    # ✅ Corrected Pinecone query format (using keyword arguments)
    search_results = index.query(vector=query_vector, top_k=top_k, include_metadata=True)

    print(f"🔍 Found {len(search_results['matches'])} similar videos")
    
    # Print search results
    for match in search_results["matches"]:
        print(f"📌 Video ID: {match['id']} | Similarity Score: {match['score']:.4f}")

    return search_results["matches"]

# Example Usage:
find_similar_videos("output/clean/zSLn0oYp7lg.mp4/zSLn0oYp7lg_rs.json", "original", top_k=5)
