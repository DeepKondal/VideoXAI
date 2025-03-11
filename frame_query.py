import os
from pinecone import Pinecone
import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

# ------------------------------------------------------
# 1) SETUP PINECONE CONNECTION
# ------------------------------------------------------
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

if not PINECONE_API_KEY:
    raise ValueError("❌ Pinecone API key is missing! Check your .env file.")

pc = Pinecone(api_key=PINECONE_API_KEY)

# Connect to both indexes
original_index = pc.Index("original-video-db")
adversarial_index = pc.Index("adversarial-video-db")

# Load embedding model
embedding_model = SentenceTransformer("sentence-transformers/paraphrase-mpnet-base-v2")

# ------------------------------------------------------
# 2) QUERY FUNCTION TO FIND SIMILAR FRAMES
# ------------------------------------------------------
def query_similar_frames(frame_data, top_k=5, video_filter=None, index=original_index):
    query_embedding = embedding_model.encode(str(frame_data)).tolist()
    query_embedding = query_embedding[:768] + [0.0] * (768 - len(query_embedding))

    metadata_filter = {"video_name": {"$eq": video_filter}} if video_filter else None

    query_results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True,
        filter=metadata_filter
    )

    matches = [
        {"frame_id": match.id, "score": match.score, "metadata": match.metadata}
        for match in query_results.matches
    ]

    print(f"\n🔍 Similar Frames in {'Original' if index == original_index else 'Adversarial'} DB:")
    for match in matches:
        print(f"🖼️ Frame: {match['frame_id']}, Score: {match['score']:.4f}, Metadata: {match['metadata']}")

    return matches

# ------------------------------------------------------
# 3) QUERY FUNCTION TO FILTER BY ATTENTION THRESHOLD
# ------------------------------------------------------
def query_by_attention_threshold(min_max_attention=0.02, top_k=10, index=original_index):
    query_results = index.query(
        vector=[0.0] * 768,
        top_k=top_k,
        include_metadata=True,
        filter={"max_attention": {"$gt": min_max_attention}}
    )

    matches = [
        {"frame_id": match.id, "score": match.score, "metadata": match.metadata}
        for match in query_results.matches
    ]

    print(f"\n📊 Frames with Max Attention > {min_max_attention} in {'Original' if index == original_index else 'Adversarial'} DB:")
    for match in matches:
        print(f"📊 Frame: {match['frame_id']}, Score: {match['score']:.4f}, Metadata: {match['metadata']}")

    return matches

# ------------------------------------------------------
# 4) QUERY FUNCTION TO FIND SPECIFIC VIDEO FRAMES
# ------------------------------------------------------
def get_frames_for_video(video_name, top_k=20, index=original_index):
    query_results = index.query(
        vector=[0.0] * 768,
        top_k=top_k,
        include_metadata=True,
        filter={"video_name": {"$eq": video_name}}
    )

    frames = [
        {"frame_id": match.id, "score": match.score, "metadata": match.metadata}
        for match in query_results.matches
    ]

    print(f"\n🎬 Frames for Video '{video_name}' in {'Original' if index == original_index else 'Adversarial'} DB:")
    for frame in frames:
        print(f"🎬 Frame: {frame['frame_id']}, Score: {frame['score']:.4f}, Metadata: {frame['metadata']}")

    return frames

# def query_top_key_frames(video_name, top_k=5, index=original_index):
#     """
#     Queries Pinecone for the top `top_k` key frames in a video based on temporal saliency.

#     :param video_name: Name of the video in the database.
#     :param top_k: Number of top key frames to retrieve.
#     :param index: Pinecone index to query (original/adversarial).
#     :return: Retrieved frames with metadata.
#     """
#     query_results = index.query(
#         vector=[0.0] * 768,  # Dummy vector to fetch frames by metadata
#         top_k=top_k,
#         include_metadata=True,
#         filter={"video_name": {"$eq": video_name}}
#     )

#     key_frames = sorted(
#         [
#             {
#                 "frame_id": match.id,
#                 "score": match.score,
#                 "metadata": match.metadata
#             }
#             for match in query_results.matches
#         ],
#         key=lambda x: x["metadata"]["max_attention"],  # Sort by max_attention (highest saliency)
#         reverse=True
#     )[:top_k]  # Get the top_k frames

#     print(f"\n🔥 Top {top_k} Key Frames with Temporal Saliency for Video: {video_name}")
#     for frame in key_frames:
#         print(f"🖼️ Frame: {frame['frame_id']}, Score: {frame['score']:.4f}, Max Attention: {frame['metadata']['max_attention']}")

#     return key_frames
def query_top_key_frames(video_name, top_k=5, index=original_index):
    """
    Queries Pinecone for the top `top_k` key frames in a video based on temporal saliency.

    :param video_name: Name of the video in the database.
    :param top_k: Number of top key frames to retrieve.
    :param index: Pinecone index to query (original/adversarial).
    :return: Retrieved frames with metadata.
    """
    query_results = index.query(
        vector=[0.0] * 768,  # Dummy vector to fetch frames by metadata
        top_k=top_k,
        include_metadata=True,
        filter={"video_name": {"$eq": video_name}}
    )

    key_frames = sorted(
        [
            {
                "frame_id": match.id,
                "score": match.score,
                "metadata": match.metadata
            }
            for match in query_results.matches
        ],
        key=lambda x: x["metadata"]["mean_attention"],  # Sort by mean_attention (highest saliency)
        reverse=True
    )[:top_k]  # Get the top_k frames

    print(f"\n🔥 Top {top_k} Key Frames with Temporal Saliency for Video: {video_name}")
    for frame in key_frames:
        print(f"🖼️ Frame: {frame['frame_id']}, Score: {frame['score']:.4f}, Mean Attention: {frame['metadata']['mean_attention']}")

    return key_frames

# ------------------------------------------------------
# 5) MAIN FUNCTION FOR TESTING
# ------------------------------------------------------
if __name__ == "__main__":
    print("\n🔎 Running Pinecone Queries for Both Original and Adversarial Videos...")

    # Sample frame data for similarity search
    sample_frame = {
        "frame_index": 112,
        "max_attention": 0.0293,
        "min_attention": 0.0019,
        "mean_attention": 0.0055
    }

    # 🔍 Finding Similar Frames
    query_similar_frames(sample_frame, top_k=5, video_filter="4ZRld8o3BiM", index=original_index)
    query_similar_frames(sample_frame, top_k=5, video_filter="4ZRld8o3BiM", index=adversarial_index)

    # 📊 Finding Frames with High Max Attention
    query_by_attention_threshold(min_max_attention=0.02, top_k=5, index=original_index)
    query_by_attention_threshold(min_max_attention=0.02, top_k=5, index=adversarial_index)

    # 🎬 Getting Frames for a Specific Video
    get_frames_for_video("4ZRld8o3BiM", top_k=10, index=original_index)
    get_frames_for_video("4ZRld8o3BiM", top_k=10, index=adversarial_index)

    print("\n✅ All queries completed successfully!")
    # Query for top 5 frames with high temporal saliency in a specific video
    video_name = "_hShLat2FSY"  # Replace with your video ID
    top_key_frames = query_top_key_frames(video_name, top_k=5, index=original_index)
