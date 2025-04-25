
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



# async def process_perturbation_config(perturbation_config):  # For FGSM attack perturbation , add below perturbation_config
#     """Applies perturbation before model processing."""
#     if not perturbation_config['datasets']:
#         print("No perturbation configured, skipping this step.")
#         return

#     url = perturbation_config['server_url']
#     for dataset, settings in perturbation_config['datasets'].items():
#         if settings['perturbation_type'] == "none":
#             print(f"Skipping perturbation for dataset {dataset}.")
#             continue

#         full_url = f"{url}/apply-perturbation/{dataset}/{settings['perturbation_type']}/{settings['severity']}"
#         await async_http_post(full_url)

#     print("✅ Perturbation process started.")



async def process_perturbation_config(perturbation_config):        # For VBAD attack perturbation , add below perturbation_config
    """Trigger both targeted and untargeted perturbations in parallel."""
    servers = perturbation_config.get("servers", {})
    tasks = []

    if "vbad_targeted" in servers:
        print("🚀 Launching targeted attack...")
        targeted_url = f"{servers['vbad_targeted']}/run-vbad-targeted"
        tasks.append(async_http_post(targeted_url))

    if "vbad_untargeted" in servers:
        print("🚀 Launching untargeted attack...")
        untargeted_url = f"{servers['vbad_untargeted']}/run-vbad-untargeted"
        tasks.append(async_http_post(untargeted_url))

    # Run both tasks in parallel
    await asyncio.gather(*tasks)
    print("✅ Both VBAD attacks triggered in parallel.")

async def wait_for_videos(video_dir: str, expected_video_count: int, poll_interval: float = 5.0, timeout: float = 120000.0):
    """
    Wait until the `video_dir` contains at least `expected_video_count` `.mp4` files.
    """
    start = time.time()
    print(f"⏳ Waiting for {expected_video_count} videos in {video_dir}...")

    while True:
        if not os.path.exists(video_dir):
            print(f"⚠ Directory `{video_dir}` not found. Retrying in {poll_interval} seconds...")
            await asyncio.sleep(poll_interval)
            continue

        video_files = [f for f in os.listdir(video_dir) if f.endswith(".mp4")]

        if len(video_files) >= expected_video_count:
            print(f"✅ Found {len(video_files)} videos in {video_dir}. Proceeding to model processing.")
            return

        # If the timeout has passed, raise an exception.
        if (time.time() - start) > timeout:
            raise HTTPException(status_code=408, detail=f"Timeout waiting for {expected_video_count} videos in {video_dir}")

        print(f"⚠ {len(video_files)}/{expected_video_count} videos found. Retrying in {poll_interval} seconds...")
        await asyncio.sleep(poll_interval)



# async def process_model_config(model_config, expected_video_count):
async def process_model_config(model_config):
    base_urls = model_config["base_urls"]
    targeted_dir = model_config["models"]["kinetics_video"]["targeted_dir"]
    untargeted_dir = model_config["models"]["kinetics_video"]["untargeted_dir"]
    num_frames = model_config["models"]["kinetics_video"]["num_frames"]

    tasks = []
    for base_url in base_urls:
        if "8010" in base_url:
            full_url_1 = f"{base_url}/facebook/timesformer-base-finetuned-k400/process-targeted-videos"
            full_url_2 = f"{base_url}/process-targeted-videos"

            print(f"📡 Sending targeted Videos to `{full_url_1}` and `{full_url_2}`")
            tasks.append(asyncio.create_task(async_http_post(full_url_1, json_data={"video_directory": targeted_dir, "num_frames": num_frames})))
            tasks.append(asyncio.create_task(async_http_post(full_url_2, json_data={"video_directory": targeted_dir, "num_frames": num_frames})))

        elif "8011" in base_url:
            full_url_1 = f"{base_url}/facebook/timesformer-base-finetuned-k400/process-untargeted-videos"
            full_url_2 = f"{base_url}/process-untargeted-videos"

            print(f"📡 Sending untargeted Videos to `{full_url_1}` and `{full_url_2}`")
            tasks.append(asyncio.create_task(async_http_post(full_url_1, json_data={"video_directory": untargeted_dir, "num_frames": num_frames})))
            tasks.append(asyncio.create_task(async_http_post(full_url_2, json_data={"video_directory": untargeted_dir, "num_frames": num_frames})))

    await asyncio.gather(*tasks)
    print("✅ Model processing started asynchronously. Proceeding to XAI Analysis")

async def process_xai_config(xai_config):
    """Processes XAI explanations for both clean and adversarial videos as defined in the config."""
    xai_server = xai_config['base_url']

    for dataset_name, settings in xai_config['datasets'].items():
        # Explicit paths from config
        targeted_dir = settings.get('video_path', 'untargeted/final_perturbed_videos/targeted')
        untargeted_dir = settings.get('adversarial_video_path', 'untargeted/final_perturbed_videos/untargeted')
        num_frames = settings.get('num_frames', 8)

        video_types = {
            "targeted": targeted_dir,
            "untargeted": untargeted_dir
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

     # Count how many raw `.mp4` files we have to determine how many we expect in the adversarial directories
    raw_video_dir = config["upload_config"]["datasets"]["kinetics_400"]["local_video_dir"]
    expected_video_count = len([f for f in os.listdir(raw_video_dir) if f.endswith(".mp4")])

    targeted_dir = config["model_config"]["models"]["kinetics_video"]["targeted_dir"]
    untargeted_dir = config["model_config"]["models"]["kinetics_video"]["untargeted_dir"]

    # Wait for the expected number of videos in both the targeted and untargeted directories.
    await wait_for_videos(targeted_dir, expected_video_count)
    await wait_for_videos(untargeted_dir, expected_video_count)
    
    
    
    print("🔹 Processing videos using model server...")
    # await process_model_config(config["model_config"], expected_video_count)
    await process_model_config(config["model_config"])
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
