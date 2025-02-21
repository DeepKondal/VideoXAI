
from fastapi import FastAPI, BackgroundTasks, HTTPException ,File
from pydantic import BaseModel
import Model_ResNet
import os
import torch
from attention_extractor import AttentionExtractor as TimesformerAttentionExtractor
from fastapi import UploadFile
from generateTemporalSpatial import process_video, create_sample_frames_visualization
from generateTemporalSpatial import AttentionExtractor as TemporalSpatialAttentionExtractor

app = FastAPI()
# Initialize STAA
attention_extractor = TimesformerAttentionExtractor(model_name="facebook/timesformer-base-finetuned-k400", device="cuda" if torch.cuda.is_available() else "cpu")

# Extractor for Temporal-Spatial XAI (generateTemporalSpatial)
temporal_spatial_extractor = TemporalSpatialAttentionExtractor(model_name="facebook/timesformer-base-finetuned-k400",device="cuda" if torch.cuda.is_available() else "cpu")
# Output directory
OUTPUT_DIR = "video_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)


class DatasetPaths(BaseModel):
    dataset_id: str  # Add dataset_id field to the request body

def run_model_async(dataset_id, model_func):
    # Use dataset_id as needed in your model_func
    dataset_paths = f"dataprocess/datasets/{dataset_id}"
    model_func(dataset_paths)


@app.post("/resnet/{dataset_id}/{perturbation_func_name}/{severity}")
async def run_model1_background(dataset_id: str, perturbation_func_name: str, severity: int, background_tasks: BackgroundTasks):
    local_original_dataset_path = f"/home/z/Music/devnew_xaiservice/XAIport/datasets/{dataset_id}"
    local_perturbed_dataset_path = f"/home/z/Music/devnew_xaiservice/XAIport/datasets/{dataset_id}_{perturbation_func_name}_{severity}"

    # 构建 dataset_paths 列表
    dataset_paths = [local_original_dataset_path, local_perturbed_dataset_path]

    # 异步运行模型
    background_tasks.add_task(Model_ResNet.model_run, dataset_paths)

    return {
        "message": f"ResNet run for dataset {dataset_id} with perturbation {perturbation_func_name} and severity {severity} has started, results will be uploaded to Blob storage after computation."
    }



# Define directories
CLEAN_VIDEO_DIR = "dataprocess/videos"
OUTPUT_DIR = "output"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "clean"), exist_ok=True)


# Initialize extractors
timesformer_extractor = TimesformerAttentionExtractor(
    model_name="facebook/timesformer-base-finetuned-k400",
    device="cuda" if torch.cuda.is_available() else "cpu"
)

temporal_spatial_extractor = TemporalSpatialAttentionExtractor(
    model_name="facebook/timesformer-base-finetuned-k400",
    device="cuda" if torch.cuda.is_available() else "cpu"
)


@app.post("/facebook/timesformer-base-finetuned-k400/process-original-videos")
async def process_timesformer_original_videos():
    """
    This endpoint processes only original (clean) videos using Facebook Timesformer.
    """
    results = []

    if not os.path.exists(CLEAN_VIDEO_DIR):
        raise HTTPException(status_code=404, detail="Original video directory not found.")

    video_files = [f for f in os.listdir(CLEAN_VIDEO_DIR) if f.endswith(".mp4")]
    
    if not video_files:
        return {"message": "No original videos found in directory."}

    for video_file in video_files:
        video_path = os.path.join(CLEAN_VIDEO_DIR, video_file)

        try:
            # Extract attention & logits using Timesformer
            spatial_attention, temporal_attention, frames, logits = timesformer_extractor.extract_attention(video_path)
            prediction_idx = torch.argmax(logits, dim=1).item()
            prediction = timesformer_extractor.model.config.id2label[prediction_idx]

            # # Save results
            # video_result_dir = os.path.join(OUTPUT_DIR, "clean", os.path.splitext(video_file)[0])
            # os.makedirs(video_result_dir, exist_ok=True)
            # ✅ Create a dedicated `prediction` folder inside `clean`
            prediction_dir = os.path.join(OUTPUT_DIR, "clean", "prediction")
            os.makedirs(prediction_dir, exist_ok=True)

            # ✅ Create a subfolder for each video inside `clean/prediction/`
            video_result_dir = os.path.join(prediction_dir, os.path.splitext(video_file)[0])
            os.makedirs(video_result_dir, exist_ok=True)

            # Save visualization
            timesformer_extractor.visualize_attention(
                spatial_attention, temporal_attention, frames, video_result_dir, prediction, "clean"
            )

            results.append({
                "video_file": video_file,
                "video_type": "clean",
                "prediction": prediction,
                "results_dir": video_result_dir
            })

        except Exception as e:
            results.append({
                "video_file": video_file,
                "video_type": "clean",
                "error": str(e)
            })

    return {"results": results}


@app.post("/process-original-videos")
async def process_and_visualize_original_videos():
    """
    This endpoint processes only original (clean) videos using Temporal-Spatial XAI.
    """
    try:
        response = {"message": "Processing original videos", "clean_videos": []}

        # Process Clean Videos
        if os.path.exists(CLEAN_VIDEO_DIR):
            clean_videos = [f for f in os.listdir(CLEAN_VIDEO_DIR) if f.endswith(".mp4")]
            for video_name in clean_videos:
                clean_video_path = os.path.join(CLEAN_VIDEO_DIR, video_name)
                clean_output_dir = os.path.join(OUTPUT_DIR, "clean", video_name)
                os.makedirs(clean_output_dir, exist_ok=True)

                clean_predicted_label, clean_frames_dir, clean_json_path, clean_heatmap_video_path = process_video(
                    clean_video_path, clean_output_dir, extractor=temporal_spatial_extractor
                )

                clean_visualization_path = create_sample_frames_visualization(
                    video_name=os.path.splitext(video_name)[0],
                    results_dir=clean_output_dir
                )

                response["clean_videos"].append({
                    "video_name": video_name,
                    "predicted_label": clean_predicted_label,
                    "frames_directory": clean_frames_dir,
                    "attention_json_path": clean_json_path,
                    "heatmap_video_path": clean_heatmap_video_path,
                    "visualization_path": clean_visualization_path
                })

        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

@app.get("/health")
def health_check():
    return {"status": "OK"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)