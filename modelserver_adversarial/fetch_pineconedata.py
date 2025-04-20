import os
from pinecone_config import index  # ✅ Import Pinecone index

# ✅ Function to fetch stored vectors
def fetch_vector(video_name):
    """Fetches a stored vector from Pinecone."""
    try:
        result = index.fetch(ids=[video_name])  # Returns a FetchResponse object

        # ✅ Access the fetched vectors correctly
        vectors = result.vectors  # Use .vectors instead of dict indexing

        if video_name in vectors:
            print(f"✅ Fetched vector for {video_name}:")
            print(vectors[video_name].values)  # ✅ Access vector values
        else:
            print(f"❌ No data found for {video_name}")

    except Exception as e:
        print(f"❌ Error fetching vector: {e}")

# ✅ Replace with your stored video filename
video_id = "__NrybzYzUg.mp4"  # Example video ID
fetch_vector(video_id)
