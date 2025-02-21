
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


# Paths for adversarial videos
ADVERSARIAL_VIDEO_DIR = "dataprocess/FGSM"
OUTPUT_DIR = "output/adversarial"
os.makedirs(OUTPUT_DIR, exist_ok=True)

@app.post("/facebook/timesformer-base-finetuned-k400/process-adversarial-videos")
async def process_adversarial_timesformer():
    """Process adversarial videos using Facebook Timesformer"""
    if not os.path.exists(ADVERSARIAL_VIDEO_DIR):
        raise HTTPException(status_code=404, detail="Adversarial video directory not found")

    adversarial_videos = [f for f in os.listdir(ADVERSARIAL_VIDEO_DIR) if f.endswith(".mp4")]
    results = []

    for video_name in adversarial_videos:
        video_path = os.path.join(ADVERSARIAL_VIDEO_DIR, video_name)
        try:
            spatial_attention, temporal_attention, frames, logits = attention_extractor.extract_attention(video_path)
            prediction_idx = torch.argmax(logits, dim=1).item()
            prediction = attention_extractor.model.config.id2label[prediction_idx]

            # ✅ Create the `prediction` folder inside `adversarial`
            prediction_dir = os.path.join(OUTPUT_DIR, "adversarial", "prediction")
            os.makedirs(prediction_dir, exist_ok=True)

            # ✅ Create a subfolder for each video inside `adversarial/prediction/`
            video_result_dir = os.path.join(prediction_dir, os.path.splitext(video_name)[0])
            os.makedirs(video_result_dir, exist_ok=True)

            # Save visualization
            attention_extractor.visualize_attention(
                spatial_attention, temporal_attention, frames, video_result_dir, prediction, "adversarial"
            )
            results.append({
                "video_file": video_name,
                "video_type": "adversarial",
                "prediction": prediction,
                "results_dir": video_result_dir
            })
        except Exception as e:
            results.append({
                "video_file": video_name,
                "video_type": "adversarial",
                "error": str(e)
            })

    return {"results": results}


@app.post("/process-adversarial-videos")
async def process_adversarial_temporal_spatial():
    """Process adversarial videos using Temporal-Spatial XAI"""
    try:
        response = {"message": "Processing adversarial videos", "clean_videos": []}

        if not os.path.exists(ADVERSARIAL_VIDEO_DIR):
            raise HTTPException(status_code=404, detail="Adversarial video directory not found")

        adversarial_videos = [f for f in os.listdir(ADVERSARIAL_VIDEO_DIR) if f.endswith(".mp4")]
        response = {"message": "Processing adversarial videos", "adversarial_videos": []}

        for video_name in adversarial_videos:
            video_path = os.path.join(ADVERSARIAL_VIDEO_DIR, video_name)
            output_dir = os.path.join(OUTPUT_DIR, video_name)
            os.makedirs(output_dir, exist_ok=True)

            predicted_label, frames_dir, json_path, heatmap_video_path = process_video(
                video_path, output_dir, extractor=temporal_spatial_extractor
            )

            visualization_path = create_sample_frames_visualization(
                video_name=os.path.splitext(video_name)[0],
                results_dir=output_dir
            )

            response["adversarial_videos"].append({
                "video_name": video_name,
                "predicted_label": predicted_label,
                "frames_directory": frames_dir,
                "attention_json_path": json_path,
                "heatmap_video_path": heatmap_video_path,
                "visualization_path": visualization_path
            })

        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

@app.get("/health")
def health_check():
    return {"status": "OK"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8005)