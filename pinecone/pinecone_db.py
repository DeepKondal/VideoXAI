import os
import json
import numpy as np
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer

# ✅ Load Pre-trained Embedding Model (512D output)
# embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
# embedding_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")  # ✅ 512D Model
embedding_model = SentenceTransformer("sentence-transformers/paraphrase-mpnet-base-v2")  # ✅ 768D Model


# ✅ Pinecone API Key (Replace with your actual API key)
PINECONE_API_KEY = "Secret_key"

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
            dimension=768,  # ✅ Using 768D vectors for embeddings
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
    embedding = embedding_model.encode(text_representation)  # Generate embedding (768)
    return embedding.tolist()  # Convert to list for Pinecone

# ✅ Function to upload JSON data with embeddings to Pinecone
def process_json_and_upload(json_path, video_name, data_type, version="1"):
    """ Reads JSON data, generates embeddings, and uploads to Pinecone with versioning. """
    if not os.path.exists(json_path):
        print(f"❌ JSON file not found: {json_path}")
        return

    # Load JSON attention data
    with open(json_path, 'r') as f:
        attention_data = json.load(f)

    # ✅ Generate embedding (512D vector)
    embedding_vector = generate_embedding(attention_data)

    # ✅ Ensure vector dimension is exactly 512D (Padding if needed)
    required_dim = 768  # Update to 768 for the new model
    if len(embedding_vector) < required_dim:
        embedding_vector.extend([0.0] * (required_dim - len(embedding_vector)))  # Pad with zeros
    elif len(embedding_vector) > required_dim:
        embedding_vector = embedding_vector[:required_dim]  # Trim excess dimensions

    # ✅ Create a unique ID with versioning
    pinecone_id = f"{video_name}_v{version}"

    # ✅ Select the correct index (original or adversarial)
    index = original_index if data_type == "original" else adversarial_index

    # ✅ Upload to Pinecone
    index.upsert([(pinecone_id, embedding_vector)])
    print(f"✅ Uploaded {video_name} (version {version}) to {data_type} database with embeddings.")

# ✅ Function to batch upload all JSON files in a directory
def batch_upload(data_dir, data_type, version="1"):
    """ Finds all JSON files in a directory and uploads them to Pinecone. """
    print(f"🔍 Searching for JSON files in: {data_dir}")

    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith("_rs.json"):  # Only process attention JSON files
                video_name = file.replace("_rs.json", "")  # Extract video ID
                json_path = os.path.join(root, file)

                print(f"📂 Found JSON: {json_path}")  # Debugging print
                process_json_and_upload(json_path, video_name, data_type, version)

if __name__ == "__main__":
    # ✅ Upload Original Videos
    print("🚀 Uploading Original Videos...")
    batch_upload(CLEAN_DIR, "original", version="1")

    # ✅ Upload Adversarial Videos
    print("🚀 Uploading Adversarial Videos...")
    batch_upload(ADVERSARIAL_DIR, "adversarial", version="1")

    print("✅ All videos have been uploaded to Pinecone!")
