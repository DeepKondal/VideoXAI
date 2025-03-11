import os
import json
import numpy as np
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer


load_dotenv()
# ------------------------------------------------------
# 1) LOAD MODEL
# ------------------------------------------------------
embedding_model = SentenceTransformer("sentence-transformers/paraphrase-mpnet-base-v2")  # 768D

# ------------------------------------------------------
# 2) PINECONE SETUP
# ------------------------------------------------------

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")  # Use env variable or paste key

pc = Pinecone(api_key=PINECONE_API_KEY)

original_index_name = "original-video-db"
adversarial_index_name = "adversarial-video-db"

# Create indexes if they don't exist
for index_name in [original_index_name, adversarial_index_name]:
    existing_indexes = pc.list_indexes().names()
    if index_name not in existing_indexes:
        pc.create_index(
            name=index_name,
            dimension=768,  # 768D model
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
        print(f"✅ Created Pinecone index: {index_name}")
    else:
        print(f"⚡ Pinecone index '{index_name}' already exists.")

original_index = pc.Index(original_index_name)
adversarial_index = pc.Index(adversarial_index_name)
print("✅ Successfully connected to Pinecone!")

# ------------------------------------------------------
# 3) DIRECTORIES
# ------------------------------------------------------
BASE_DIR = "output"
CLEAN_DIR = os.path.join(BASE_DIR, "clean")
ADVERSARIAL_DIR = os.path.join(BASE_DIR, "adversarial")

# ------------------------------------------------------
# 4) EMBEDDING FUNCTION
# ------------------------------------------------------
def generate_embedding(frame_data: dict):
    """
    Converts a *single frame's* data into a text string and
    generates a 768D embedding using the model.
    
    e.g. frame_data = {
         "frame_index": 4,
         "max_attention": ...,
         "min_attention": ...,
         "mean_attention": ...
    }
    """
    # Convert the frame dictionary to a string (JSON or any textual format)
    text_representation = json.dumps(frame_data)  
    embedding = embedding_model.encode(text_representation)
    return embedding.tolist()

# ------------------------------------------------------
# 5) PROCESS AND UPLOAD FRAME-WISE
# ------------------------------------------------------
def process_json_and_upload(json_path, video_name, data_type, version="1"):
    """
    Reads JSON data (a list of frames). For each frame,
    - Generate a separate embedding
    - Upload (upsert) to Pinecone with metadata
    """
    if not os.path.exists(json_path):
        print(f"❌ JSON file not found: {json_path}")
        return

    # Load all frames
    with open(json_path, 'r') as f:
        frames_data = json.load(f)
    
    # Decide which Pinecone index to use
    index = original_index if data_type == "original" else adversarial_index

    # We'll batch upsert all frames together for speed
    to_upsert = []

    for frame in frames_data:
        # frame looks like: {"frame_index": 4, "max_attention": 0.014..., "min_attention": ...}

        # 1) Create a unique ID for each frame
        frame_id = frame["frame_index"]
        pinecone_id = f"{video_name}_frame_{frame_id}_v{version}"

        # 2) Generate embedding for this single frame
        embedding_vector = generate_embedding(frame)

        # 3) (Optional) Ensure vector dimension is 768 (pad or trim if needed)
        required_dim = 768
        if len(embedding_vector) < required_dim:
            embedding_vector.extend([0.0] * (required_dim - len(embedding_vector)))
        elif len(embedding_vector) > required_dim:
            embedding_vector = embedding_vector[:required_dim]

        # 4) Prepare metadata
        # You can store frame_index, video_name, and the raw attention values for filtering or reference
        metadata = {
            "video_name": video_name,
            "frame_index": frame_id,
            "max_attention": frame.get("max_attention", 0),
            "min_attention": frame.get("min_attention", 0),
            "mean_attention": frame.get("mean_attention", 0)
        }

        # 5) Add to the upsert list
        to_upsert.append((pinecone_id, embedding_vector, metadata))

    # 6) Upsert all frames in a single batch
    index.upsert(vectors=to_upsert)
    print(f"✅ Uploaded {len(to_upsert)} frames for '{video_name}' (version {version}) to {data_type} DB.")

# ------------------------------------------------------
# 6) BATCH UPLOAD HELPER
# ------------------------------------------------------
def batch_upload(data_dir, data_type, version="1"):
    """Finds all JSON files in a directory and uploads them to Pinecone."""
    print(f"🔍 Searching for JSON files in: {data_dir}")

    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith("_rs.json"):  # or any pattern you want
                video_name = file.replace("_rs.json", "")  # Extract the video ID
                json_path = os.path.join(root, file)

                print(f"📂 Found JSON: {json_path}")
                process_json_and_upload(json_path, video_name, data_type, version)

# ------------------------------------------------------
# 7) MAIN
# ------------------------------------------------------
if __name__ == "__main__":
    # Upload Original Videos
    print("🚀 Uploading Original Videos...")
    batch_upload(CLEAN_DIR, "original", version="1")

    # Upload Adversarial Videos
    print("🚀 Uploading Adversarial Videos...")
    batch_upload(ADVERSARIAL_DIR, "adversarial", version="1")

    print("✅ All videos have been uploaded to Pinecone!")
