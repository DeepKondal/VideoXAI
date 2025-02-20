'''
Version 0.0.1
This is the previous try to create a server that can handle the pipeline of the XAI service.
Without the status check system, we can't correctly run the whole pipeline.
This must be reconstructed to a new version.
'''

from fastapi import FastAPI, HTTPException, Request
import httpx
import json
import os
import asyncio
import logging
from adversarial.adversarial_model import evaluate_adversarial
import random
from neo4j_client import ProvenanceModel
import datetime
import aiohttp
import random


app = FastAPI(title="Coordination Center")
#Provenance_logic
provenance = ProvenanceModel()


# Traffic Counters
traffic_count = {"8002": 0, "8005": 0}

async def async_http_post(url, json_data=None, files=None):
    async with httpx.AsyncClient(timeout=120.0) as client:  # Timeout set to 60 seconds
        if json_data:
            response = await client.post(url, json=json_data)
        elif files:
            response = await client.post(url, files=files)
        else:
            response = await client.post(url)

        # 检查是否是307重定向响应
        if response.status_code == 307:
            redirect_url = response.headers.get('Location')
            if redirect_url:
                print(f"Redirecting to {redirect_url}")
                return await async_http_post(redirect_url, json_data, files)

        if response.status_code != 200:
            print(f"Error response: {response.text}")  # 打印出错误响应内容
            logging.error(f"Error in POST to {url}: {response.status_code} - {response.text}")
            raise HTTPException(status_code=response.status_code, detail=response.text)

        return response.json()


# 处理上传配置
async def process_upload_config(upload_config):
    # for dataset_id, dataset_info in upload_config['datasets'].items():
    #     url = upload_config['server_url'] + f"/upload-dataset/{dataset_id}"
    #     local_zip_file_path = dataset_info['local_zip_path']  # 每个数据集的本地 ZIP 文件路径

    #     async with httpx.AsyncClient() as client:
    #         with open(local_zip_file_path, 'rb') as f:
    #             files = {'zip_file': (os.path.basename(local_zip_file_path), f)}
    #             response = await client.post(url, files=files)
    #         response.raise_for_status()

    #     # 可以添加更多的逻辑处理上传后的结果
    #     print(f"Uploaded dataset {dataset_id} successfully.")
    for dataset_id, dataset_info in upload_config['datasets'].items():
            local_video_dir = dataset_info.get('local_video_dir')

            if not local_video_dir or not os.path.exists(local_video_dir):
                raise HTTPException(status_code=400, detail=f"Video directory {local_video_dir} not found.")

            # Use the endpoint /process-kinetics-dataset to process the video directory
            url = upload_config['server_url'] + "/process-kinetics-dataset"
            json_data = {"video_dir": local_video_dir, "num_frames": 8}

            # Trigger video processing
            await async_http_post(url, json_data=json_data)
            print(f"Processed video dataset {dataset_id} successfully.")



# 处理扰动配置
async def process_perturbation_config(perturbation_config):
    if not perturbation_config['datasets']:
        print("No perturbation configured, skipping this step.")
        return
    url = perturbation_config['server_url']
    for dataset, settings in perturbation_config['datasets'].items():

        if settings['perturbation_type'] == "none":        #skip perturbation if perturbation_type is none
            print(f"Skipping perturbation for dataset {dataset}.")
            continue
        full_url = f"{url}/apply-perturbation/{dataset}/{settings['perturbation_type']}/{settings['severity']}"
        await async_http_post(full_url)



async def wait_for_perturbation(adversarial_video_dir):
    print(f"⏳ Waiting for adversarial videos in {adversarial_video_dir}...")

    while not any(f.endswith(".mp4") for f in os.listdir(adversarial_video_dir) if os.path.exists(adversarial_video_dir)):
        print("⚠ No adversarial videos found. Retrying in 5 seconds...")
        await asyncio.sleep(5)

    print("✅ Adversarial videos detected. Proceeding with Model Processing...")
# async def process_model_config(model_config):
#     """
#     Process both model servers for original and adversarial videos.
#     """
#     base_urls = model_config["base_urls"]  # Fetch both base URLs
#     original_video_dir = model_config["models"]["kinetics_video"]["original_video_dir"]
#     adversarial_video_dir = model_config["models"]["kinetics_video"]["adversarial_video_dir"]
#     num_frames = model_config["models"]["kinetics_video"]["num_frames"]

#     tasks = []

#     async with aiohttp.ClientSession() as session:
#         for base_url in base_urls:
#             # Timesformer Attention Extraction
#             timesformer_url = f"{base_url}/facebook/timesformer-base-finetuned-k400/kinetics_video"
            
#             # Temporal-Spatial Processing
#             temporal_spatial_url = f"{base_url}/process-videos"

#             print(f"Processing Original Videos: {original_video_dir} and Adversarial Videos: {adversarial_video_dir}")

#             tasks.append(async_http_post(timesformer_url, json_data={"video_directory": original_video_dir, "num_frames": num_frames}))
#             tasks.append(async_http_post(temporal_spatial_url, json_data={"video_directory": adversarial_video_dir, "num_frames": num_frames}))

#         # Execute all API calls in parallel
#         responses = await asyncio.gather(*tasks)

#     return {"message": "Model processing tasks initiated.", "responses": responses}



async def process_model_config(model_config):
    """
    Distributes videos evenly between two model servers while ensuring each model type processes the videos correctly.
    """
    base_urls = model_config["base_urls"]  # Two model servers (8002 & 8005)
    original_video_dir = model_config["models"]["kinetics_video"]["original_video_dir"]
    adversarial_video_dir = model_config["models"]["kinetics_video"]["adversarial_video_dir"]
    num_frames = model_config["models"]["kinetics_video"]["num_frames"]
    load_split = model_config["load_split"]  # Example: 50 (50% of videos go to each server)

    # Collect videos
    original_videos = sorted([f for f in os.listdir(original_video_dir) if f.endswith(".mp4")])
    adversarial_videos = sorted([f for f in os.listdir(adversarial_video_dir) if f.endswith(".mp4")])

    all_videos = original_videos + adversarial_videos  # Combine both sets
    random.shuffle(all_videos)  # Shuffle videos for randomness

    # Split videos between two servers
    split_index = int((load_split / 100) * len(all_videos))
    server_1_videos = all_videos[:split_index]
    server_2_videos = all_videos[split_index:]

    tasks = []

    async with aiohttp.ClientSession() as session:
        for server_idx, (server_url, video_list) in enumerate(zip(base_urls, [server_1_videos, server_2_videos])):
            for video_name in video_list:
                is_adversarial = video_name in adversarial_videos
                video_dir = adversarial_video_dir if is_adversarial else original_video_dir

                # Assign endpoints properly:
                # - Server 1 gets `/process-videos` (Temporal-Spatial)
                # - Server 2 gets `/facebook/timesformer` (Timesformer Attention)
                assigned_endpoint = "/process-videos" if server_idx == 0 else "/facebook/timesformer-base-finetuned-k400/kinetics_video"

                full_url = f"{server_url}{assigned_endpoint}"
                print(f"📡 Sending `{video_name}` to `{server_url}` → {assigned_endpoint}")

                # Send request
                tasks.append(async_http_post(full_url, json_data={"video_directory": video_dir, "num_frames": num_frames}))

        # Run all async calls in parallel
        responses = await asyncio.gather(*tasks)

    return {"message": "Model processing tasks initiated.", "responses": responses}

# async def process_xai_config(xai_config):
#     xai_server = xai_config['base_url']  # XAI service at port 8003

#     for dataset, settings in xai_config['datasets'].items():
#         video_dir = settings.get('video_path', '')
#         num_frames = settings.get('num_frames', 8)

#         if not os.path.isdir(video_dir):
#             print(f"⚠ Video path {video_dir} is not a directory. Skipping XAI processing.")
#             continue

#         video_files = [os.path.abspath(os.path.join(video_dir, f)) for f in os.listdir(video_dir) if f.endswith(".mp4")]

#         if not video_files:
#             print(f"⚠ No video files found in {video_dir}. Skipping XAI processing.")
#             continue

#         for video_file in video_files:
#             try:
#                 data = {
#                     "video_path": video_file,
#                     "num_frames": num_frames
#                 }

#                 print(f"📡 Sending XAI request to {xai_server} for {video_file}")
#                 xai_full_url = f"{xai_server}/staa-video-explain/"
#                 xai_response = await async_http_post(xai_full_url, json_data=data)
#                 print(f"✅ XAI response for {video_file}: {xai_response}")

#             except Exception as e:
#                 print(f"❌ Error processing XAI from {xai_server} for video {video_file}: {e}")

async def process_xai_config(xai_config):
    """
    Process XAI explanation for both clean (original) and adversarial videos.
    """
    xai_server = xai_config['base_url']  # XAI service at port 8003

    for dataset, settings in xai_config['datasets'].items():
        original_video_dir = settings.get('video_path', 'dataprocess/videos/')
        adversarial_video_dir = original_video_dir.replace("videos", "FGSM")  # Path for adversarial videos
        num_frames = settings.get('num_frames', 8)

        # Check both directories
        for video_dir, video_type in [(original_video_dir, "clean"), (adversarial_video_dir, "adversarial")]:
            if not os.path.isdir(video_dir):
                print(f"⚠ Video path {video_dir} not found. Skipping {video_type} XAI processing.")
                continue

            video_files = [os.path.abspath(os.path.join(video_dir, f)) for f in os.listdir(video_dir) if f.endswith(".mp4")]

            if not video_files:
                print(f"⚠ No {video_type} video files found in {video_dir}. Skipping XAI processing.")
                continue

            for video_file in video_files:
                try:
                    data = {
                        "video_path": video_file,
                        "num_frames": num_frames
                    }

                    print(f"📡 Sending XAI request for {video_type} video to {xai_server}: {video_file}")
                    xai_full_url = f"{xai_server}/staa-video-explain/"
                    xai_response = await async_http_post(xai_full_url, json_data=data)

                    # Save the XAI response to a JSON file
                    xai_results_dir = os.path.join("xai_results", video_type)
                    os.makedirs(xai_results_dir, exist_ok=True)

                    json_file_path = os.path.join(xai_results_dir, os.path.basename(video_file).replace(".mp4", "_attention.json"))
                    with open(json_file_path, "w") as json_file:
                        json.dump(xai_response, json_file, indent=4)

                    print(f"✅ {video_type.capitalize()} XAI response saved: {json_file_path}")

                except Exception as e:
                    print(f"❌ Error processing XAI for {video_type} video {video_file}: {e}")




# 处理评估配置
async def process_evaluation_config(evaluation_config):
    base_url = evaluation_config['base_url']
    for dataset, settings in evaluation_config['datasets'].items():
        data = {
            "dataset_id": dataset,
            "model_name": settings['model_name'],
            "perturbation_func": settings['perturbation_func'],
            "severity": settings['severity'],
            "cam_algorithms": settings['algorithms']
        }
        full_url = f"{base_url}/evaluate_cam"
        await async_http_post(full_url, json_data=data)

# 按顺序处理每个配置步骤
async def process_pipeline_step(config, step_key, process_function):
    if step_key in config:
        await process_function(config[step_key])


# async def run_pipeline_from_config(config):
#     run_id = f"run_{datetime.datetime.now().isoformat()}"

#     try:
#         provenance.create_pipeline_run(run_id, datetime.datetime.now().isoformat(), "Running")
#         dataset_id = "kinetics_400"
#         provenance.create_dataset(dataset_id, "Kinetics-400", "dataprocess/videos/")
#     except Exception as e:
#         logging.error(f"❌ Failed to create provenance records: {e}")

#     # Step 1: Upload Data
#     try:
#         await process_pipeline_step(config, 'upload_config', process_upload_config)
#         provenance.create_processing_step("Upload Data", "upload", str(config["upload_config"]))
#         provenance.link_pipeline_step(run_id, "Upload Data")
#         provenance.link_dataset_to_processing(dataset_id, "Upload Data")
#     except Exception as e:
#         logging.error(f"❌ Upload processing failed: {e}")

#     # Step 2: Apply Perturbation
#     try:
#         await process_pipeline_step(config, 'perturbation_config', process_perturbation_config)
#         provenance.create_processing_step("Apply Perturbation", "perturbation", str(config["perturbation_config"]))
#         provenance.link_pipeline_step(run_id, "Apply Perturbation")
#     except Exception as e:
#         logging.error(f"❌ Perturbation processing failed: {e}")
    
#     # Step 3: Model Processing
#     try:
#         await process_pipeline_step(config, 'model_config', process_model_config)  # Fixed function call
#         provenance.create_processing_step("Model Processing", "model", str(config["model_config"]))
#         provenance.link_pipeline_step(run_id, "Model Processing")
#     except Exception as e:
#         logging.error(f"❌ Model processing failed: {e}")

#     # Step 4: XAI Analysis
#     try:
#         await process_pipeline_step(config, 'xai_config', process_xai_config)
#         provenance.create_processing_step("XAI Analysis", "XAI", str(config["xai_config"]))
#         provenance.link_pipeline_step(run_id, "XAI Analysis")
#     except Exception as e:
#         logging.error(f"❌ XAI processing failed: {e}")

#     try:
#         provenance.create_pipeline_run(run_id, datetime.datetime.now().isoformat(), "Completed")
#     except Exception as e:
#         logging.error(f"❌ Failed to mark pipeline as completed in provenance: {e}")

#     print(f"✅ Pipeline {run_id} execution completed.")

# async def run_pipeline_from_config(config):
#     await process_pipeline_step(config, 'upload_config', process_upload_config)
#     await process_pipeline_step(config, 'perturbation_config', process_perturbation_config)
#     await process_pipeline_step(config, 'model_config', process_model_config)
#     await process_pipeline_step(config, 'xai_config', process_xai_config)
#     await process_pipeline_step(config, 'evaluation_config', process_evaluation_config)

import time
import os
import asyncio

async def run_pipeline_from_config(config):
    """
    Executes pipeline sequentially, ensuring adversarial videos exist before proceeding to model processing.
    """
    run_id = f"run_{datetime.datetime.now().isoformat()}"

    try:
        print("🔹 Uploading data...")
        await process_pipeline_step(config, 'upload_config', process_upload_config)
        print("✅ Upload complete.")

        print("🔹 Applying perturbations (adversarial attack)...")
        await process_pipeline_step(config, 'perturbation_config', process_perturbation_config)
        print("✅ Perturbation complete.")

        # 🛑 **Wait until adversarial videos are present**
        adversarial_video_dir = config["model_config"]["models"]["kinetics_video"]["adversarial_video_dir"]
        print(f"⏳ Waiting for adversarial videos in {adversarial_video_dir}...")

        while not any(f.endswith(".mp4") for f in os.listdir(adversarial_video_dir) if os.path.exists(adversarial_video_dir)):
            print("⚠ No adversarial videos found. Retrying in 5 seconds...")
            await asyncio.sleep(5)  # Wait 5 seconds before checking again

        print("✅ Adversarial videos detected. Proceeding with Model Processing...")

        print("🔹 Processing videos using model server...")
        await process_pipeline_step(config, 'model_config', process_model_config)
        print("✅ Model processing complete.")

        print("🔹 Running XAI analysis...")
        await process_pipeline_step(config, 'xai_config', process_xai_config)
        print("✅ XAI analysis complete.")

        print("🔹 Running evaluation...")
        await process_pipeline_step(config, 'evaluation_config', process_evaluation_config)
        print("✅ Evaluation complete.")

        print(f"🎉 ✅ Pipeline {run_id} execution completed.")

    except Exception as e:
        print(f"❌ Pipeline failed: {e}")


  
pipeline_status = {}

@app.post("/run_pipeline/")
async def run_pipeline(request: Request):
    config = await request.json()
    run_id = f"run_{datetime.datetime.now().isoformat()}"
    pipeline_status[run_id] = "Running"

    try:
        await run_pipeline_from_config(config)
        pipeline_status[run_id] = "Completed"
        # Print final traffic distribution
        print(f"📊 Final Traffic Stats: Model Server (8002) → {traffic_count['8002']} requests, New Model Server (8005) → {traffic_count['8005']} requests")
        return {"message": f"Pipeline {run_id} executed successfully"}
    except Exception as e:
        pipeline_status[run_id] = f"Failed: {str(e)}"
        return HTTPException(status_code=500, detail=str(e))

@app.get("/pipeline_status/{run_id}")
async def get_pipeline_status(run_id: str):
    status = pipeline_status.get(run_id, "Not Found")
    return {"run_id": run_id, "status": status}

# 加载配置文件
def load_config():
    with open("config.json", "r") as file:
        return json.load(file)
    


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8880)

import asyncio
import json

# 假设您的处理函数和其他必要的导入已经完成

# # 加载配置文件
# def load_config():
#     with open("/home/z/Music/devnew_xaiservice/XAIport/task_sheets/task.json", "r") as file:
#         return json.load(file)



# 主函数
def main():
    config = load_config()
    asyncio.run(run_pipeline_from_config(config))

if __name__ == "__main__":
    main()


