import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException
import os
from typing import List
from fastapi import BackgroundTasks
import subprocess


# the below function is used to run the batch process
def run_vbad_untargeted_batch():
    script_path = "untargeted\\batch_process.py"  # Change this to actual full path
    cmd = [
        "python", script_path,
        "--video_list", "untargeted\kinetics400_val_list_videos.txt",
        "--sigma", "0.001",
        "--untargeted"
    ]
    subprocess.run(cmd)


app = FastAPI()




# To run the VBAD untargeted adversarial attack on a pre-defined video list
@app.post("/run-vbad-untargeted")
async def run_vbad_untargeted(background_tasks: BackgroundTasks):
    """
    Run untargeted VBAD adversarial attack on pre-defined video list.
    """
    background_tasks.add_task(run_vbad_untargeted_batch)
    return {"message": "Untargeted VBAD attack started in background."}


@app.get("/health")
def health_check():
    return {"status": "OK"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8006)
    