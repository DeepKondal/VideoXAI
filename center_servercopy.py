
from fastapi import FastAPI, HTTPException, Request
import httpx
import json
import os
import asyncio
import logging
import datetime
import aiohttp
import time

app = FastAPI(title="Coordination Center")

# Track processing status
pipeline_status = {}
traffic_count = {"8002": 0, "8005": 0}

processed_videos = set()  
async def async_http_post(url, json_data=None):
    """Sends an async HTTP POST request and handles errors."""
    async with httpx.AsyncClient(timeout=600.0) as client:
        response = await client.post(url, json=json_data) if json_data else await client.post(url)

        if response.status_code == 307:  # Handle redirects
            redirect_url = response.headers.get('Location')
            if redirect_url:
                return await async_http_post(redirect_url, json_data)

        if response.status_code != 200:
            logging.error(f"Error in POST to {url}: {response.status_code} - {response.text}")
            raise HTTPException(status_code=response.status_code, detail=response.text)

        return response.json()


async def process_upload_config(upload_config):
    """Uploads videos before processing."""
    for dataset_id, dataset_info in upload_config['datasets'].items():
        local_video_dir = dataset_info.get('local_video_dir')

        if not local_video_dir or not os.path.exists(local_video_dir):
            raise HTTPException(status_code=400, detail=f"Video directory {local_video_dir} not found.")

        url = f"{upload_config['server_url']}/process-kinetics-dataset"
        json_data = {"video_dir": local_video_dir, "num_frames": 8}

        await async_http_post(url, json_data=json_data)
        print(f"✅ Uploaded dataset {dataset_id} successfully.")


async def process_perturbation_config(perturbation_config):
    """Applies perturbation before model processing."""
    if not perturbation_config['datasets']:
        print("No perturbation configured, skipping this step.")
        return

    url = perturbation_config['server_url']
    for dataset, settings in perturbation_config['datasets'].items():
        if settings['perturbation_type'] == "none":
            print(f"Skipping perturbation for dataset {dataset}.")
            continue

        full_url = f"{url}/apply-perturbation/{dataset}/{settings['perturbation_type']}/{settings['severity']}"
        await async_http_post(full_url)

    print("✅ Perturbation process started.")


async def wait_for_all_adversarial_videos(adversarial_video_dir, expected_video_count):
    """Waits until ALL adversarial videos are processed before proceeding."""
    print(f"⏳ Waiting for all {expected_video_count} adversarial videos in {adversarial_video_dir}...")

    while True:
        if not os.path.exists(adversarial_video_dir):
            print(f"⚠ Adversarial directory `{adversarial_video_dir}` not found. Retrying in 5 seconds...")
            await asyncio.sleep(5)
            continue

        adversarial_videos = [f for f in os.listdir(adversarial_video_dir) if f.endswith(".mp4")]

        if len(adversarial_videos) >= expected_video_count:
            break

        print(f"⚠ {len(adversarial_videos)}/{expected_video_count} adversarial videos found. Retrying in 5 seconds...")
        await asyncio.sleep(5)

    print("✅ All adversarial videos are ready. Proceeding with Model Processing...")



async def process_model_config(model_config, expected_video_count):
    """Processes videos for both original (8002) and adversarial (8005) videos asynchronously."""
    base_urls = model_config["base_urls"]
    original_video_dir = model_config["models"]["kinetics_video"]["original_video_dir"]
    adversarial_video_dir = model_config["models"]["kinetics_video"]["adversarial_video_dir"]
    num_frames = model_config["models"]["kinetics_video"]["num_frames"]

    tasks = []
    async with aiohttp.ClientSession() as session:
        for base_url in base_urls:
            if "8002" in base_url:
                # Process Original Videos on Server 8002
                full_url_1 = f"{base_url}/facebook/timesformer-base-finetuned-k400/process-original-videos"
                full_url_2 = f"{base_url}/process-original-videos"

                print(f"📡 Sending Original Videos to `{full_url_1}` and `{full_url_2}`")
                tasks.append(async_http_post(full_url_1, json_data={"video_directory": original_video_dir, "num_frames": num_frames}))
                tasks.append(async_http_post(full_url_2, json_data={"video_directory": original_video_dir, "num_frames": num_frames}))

            elif "8005" in base_url:
                # Process Adversarial Videos on Server 8005
                full_url_1 = f"{base_url}/facebook/timesformer-base-finetuned-k400/process-adversarial-videos"
                full_url_2 = f"{base_url}/process-adversarial-videos"

                print(f"📡 Sending Adversarial Videos to `{full_url_1}` and `{full_url_2}`")
                tasks.append(async_http_post(full_url_1, json_data={"video_directory": adversarial_video_dir, "num_frames": num_frames}))
                tasks.append(async_http_post(full_url_2, json_data={"video_directory": adversarial_video_dir, "num_frames": num_frames}))

        await asyncio.gather(*tasks)
        

        print("✅ Model processing started asynchronously. Proceeding to XAI Analysis")


async def process_xai_config(xai_config):
    """Processes XAI explanations for both clean and adversarial videos as defined in the config."""
    xai_server = xai_config['base_url']

    for dataset_name, settings in xai_config['datasets'].items():
        # Explicit paths from config
        original_video_dir = settings.get('video_path', 'dataprocess/videos')
        adversarial_video_dir = settings.get('adversarial_video_path', 'dataprocess/FGSM')
        num_frames = settings.get('num_frames', 8)

        video_types = {
            "clean": original_video_dir,
            "adversarial": adversarial_video_dir
        }

        for video_type, video_dir in video_types.items():
            if not os.path.isdir(video_dir):
                print(f"⚠ Directory not found for {video_type} videos: {video_dir}")
                continue

            video_files = [
                os.path.abspath(os.path.join(video_dir, f))
                for f in os.listdir(video_dir)
                if f.endswith(".mp4")
            ]

            if not video_files:
                print(f"⚠ No {video_type} videos found in: {video_dir}")
                continue

            for video_file in video_files:
                try:
                    data = {
                        "video_path": video_file,
                        "num_frames": num_frames
                    }

                    print(f"📡 Sending {video_type} video to XAI server: {video_file}")
                    xai_endpoint = f"{xai_server}/staa-video-explain/"
                    xai_response = await async_http_post(xai_endpoint, json_data=data)

                    # Save response to appropriate folder
                    result_dir = os.path.join("xai_results", video_type)
                    os.makedirs(result_dir, exist_ok=True)

                    json_file_name = os.path.basename(video_file).replace(".mp4", "_attention.json")
                    json_file_path = os.path.join(result_dir, json_file_name)

                    with open(json_file_path, "w") as f:
                        json.dump(xai_response, f, indent=4)

                    print(f"✅ XAI result saved for {video_type}: {json_file_path}")

                except Exception as e:
                    print(f"❌ Failed to process {video_type} video {video_file}: {str(e)}")

async def run_pipeline_from_config(config):
    """Runs the pipeline step-by-step while ensuring all videos are ready before model processing."""
    print("🔹 Uploading data...")
    await process_upload_config(config["upload_config"])
    print("✅ Upload complete.")

    print("🔹 Applying perturbations...")
    await process_perturbation_config(config["perturbation_config"])
    print("✅ Perturbation complete.")


    adversarial_video_dir = config["model_config"]["models"]["kinetics_video"]["adversarial_video_dir"]
    expected_video_count = len([
    f for f in os.listdir(config["upload_config"]["datasets"]["kinetics_400"]["local_video_dir"])
    if f.endswith(".mp4")
        ])
    await wait_for_all_adversarial_videos(adversarial_video_dir, expected_video_count)

    
    
    
    print("🔹 Processing videos using model server...")
    await process_model_config(config["model_config"], expected_video_count)
    print("✅ Model processing complete.")
    
    print("🔹 Running XAI analysis...")
    await process_xai_config(config["xai_config"])
    print("✅ XAI analysis complete.")

@app.get("/health")
def health_check():
    return {"status": "OK"}

@app.post("/run_pipeline/")
async def run_pipeline(request: Request):
    config = await request.json()
    await run_pipeline_from_config(config)
    return {"message": "Pipeline executed successfully"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8880)
