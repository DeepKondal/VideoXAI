import os
import requests
import numpy as np
import pandas as pd  # <-- Added for CSV saving
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

# 1. Setup
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
if not PINECONE_API_KEY:
    raise ValueError("❌ Missing Pinecone API Key!")

embedder = SentenceTransformer('sentence-transformers/paraphrase-mpnet-base-v2')
pc = Pinecone(api_key=PINECONE_API_KEY)

# Load indexes
indexes = {
    "original": pc.Index("original-video-db"),
    "adversarial": pc.Index("adversarial-video-db"),
    "targeted": pc.Index("targeted-video-db"),
    "untargeted": pc.Index("untargeted-video-db")
}

# 2. Query full frame data per video
def get_unique_videos(index, top_k=2000):
    result = index.query(vector=[0.0] * 768, top_k=top_k, include_metadata=True)
    return list({m.metadata["video_name"] for m in result.matches if "video_name" in m.metadata})

def get_video_attention(index, video_name):
    result = index.query(
        vector=[0.0] * 768,
        top_k=100,
        include_metadata=True,
        filter={"video_name": {"$eq": video_name}}
    )
    return {
        m.metadata["frame_index"]: m.metadata.get("mean_attention", 0.0)
        for m in result.matches if "frame_index" in m.metadata
    }

# 3. LLM caller
def ask_ollama(prompt, model="llama3.2"):
    res = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": model, "prompt": prompt, "stream": False}
    )
    return res.json()["response"]

# 4. RAG logic
def rag_compare_model_drops():
    videos = get_unique_videos(indexes["original"], top_k=2000)
    variants = ["adversarial", "targeted", "untargeted"]
    all_rows = []
    variant_model_drops = {v: [] for v in variants}

    for video in videos:
        original = get_video_attention(indexes["original"], video)
        if not original:
            continue

        for variant in variants:
            variant_data = get_video_attention(indexes[variant], video)
            if not variant_data:
                continue

            shared_frames = original.keys() & variant_data.keys()
            if not shared_frames:
                continue

            drops = [abs(original[f] - variant_data[f]) for f in shared_frames]
            max_diff = max(drops)
            max_frame = list(shared_frames)[np.argmax(drops)]
            avg_drop = sum(drops) / len(drops)

            all_rows.append({
                "video": video,
                "variant": variant,
                "avg_diff": avg_drop,
                "max_diff": max_diff,
                "max_frame": max_frame,
                "frames_compared": len(shared_frames)
            })

            variant_model_drops[variant].append(avg_drop)

    # ✅ Save raw RAG results to CSV
    df = pd.DataFrame(all_rows)
    df.to_csv("rag_comparison_results.csv", index=False)
    print("📁 Saved detailed results to 'rag_comparison_results.csv'")

    # Sort to find top 300 by highest max drop
    all_rows.sort(key=lambda x: x["max_diff"], reverse=True)
    top_context = all_rows[:300]

    print("\n📊 MODEL DROP AVERAGES:")
    for variant, drops in variant_model_drops.items():
        avg = np.mean(drops) if drops else 0.0
        print(f"🧪 {variant.upper():<12}: {avg:.6f}")

    # Construct context string for LLM
    context = "\n".join(
        f"Video={row['video']} Variant={row['variant']} "
        f"Avg_Drop={row['avg_diff']:.6f} Max_Drop={row['max_diff']:.6f} "
        f"Max_Frame={row['max_frame']} Frames_Compared={row['frames_compared']}"
        for row in top_context
    )

    # Final LLM prompt
    prompt = f"""
You are an expert AI analyst evaluating video classification robustness.

Each line below shows frame-level degradation stats between ORIGINAL vs adversarial variants:
Format:  
Video=<video_id> Variant=<variant> Avg_Drop=<float> Max_Drop=<float> Max_Frame=<int> Frames_Compared=<int>

Your tasks:
1. From the dataset, return the TOP 10 videos ranked by Max_Drop. Do not guess. Sort exactly from highest to lowest Max_Drop:
   - For each, print:  Video ID,  Variant,  Avg_Drop,  Max_Drop,  Max_Frame,  Frames_Compared
   
2. Then, compute and print the **average Avg_Drop** per variant across all entries.

3. Report which variant is the most severe overall based on **highest average drop**.

 IMPORTANT: Only use the info provided in the data below. Do not assume other fields like frame rate, genre, etc.

--- BEGIN DATA ---
{context}
--- END DATA ---

Respond strictly using this format:

 TOP 10 MOST AFFECTED VIDEO FRAMES
1.  Video ID: ...   Variant: ...   Avg_Drop: ...   Max_Drop: ...   Max_Frame: ...   Frames_Compared: ...
...

 MODEL WITH MOST SEVERE DROPS (avg across all videos)  
 ADVERSARIAL        : ...  
 TARGETED           : ...  
 UNTARGETED         : ...  
 MOST SEVERE MODEL  : ...
"""

    print("\n🚀 Sending prompt to Ollama...")
    answer = ask_ollama(prompt)
    print("\n📊 Ollama's Result:\n")
    print(answer)


if __name__ == "__main__":
    rag_compare_model_drops()
