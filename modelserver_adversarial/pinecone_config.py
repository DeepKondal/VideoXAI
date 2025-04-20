import os
from pinecone import Pinecone, ServerlessSpec

# ✅ Load Pinecone API key from environment variables
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "pcsk_39ZDd9_RXpg6kTnaZ8eH6qdGoqqyu1216ytURYVaJHhJDz9qkdRSU4jP82wDk6mQZyCEdj")

# ✅ Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "adversarial-processed-videos"

# ✅ Check if the index exists, otherwise create it
if index_name not in pc.list_indexes().names():
    print(f"Creating Pinecone index: {index_name}")
    pc.create_index(
        name=index_name,
        dimension=512,  # Adjust based on your model output
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

# ✅ Connect to the existing index
index = pc.Index(index_name)

# ✅ Function to store vectors in Pinecone
def store_vector(video_name, vector):
    """Stores a processed vector in Pinecone, ensuring the correct dimension."""
    required_dim = 512

    # ✅ Pad if vector is too small
    if len(vector) < required_dim:
        vector += [0.0] * (required_dim - len(vector))  # Fill remaining dimensions

    # ✅ Trim if vector is too large
    vector = vector[:required_dim]

    print(f"✅ Storing {video_name} with vector length: {len(vector)}")  # Debugging print
    index.upsert(vectors=[(video_name, vector)])


# ✅ Function to retrieve vectors from Pinecone
def fetch_vector(video_name):
    """Fetches a stored vector from Pinecone."""
    return index.fetch(ids=[video_name])


print("✅ Pinecone setup complete.")
print(fetch_vector("__NrybzYzUg.mp4"))