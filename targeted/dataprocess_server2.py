import uvicorn
from fastapi import FastAPI, BackgroundTasks
import subprocess

# This function runs the batch process for targeted VBAD attack
def run_vbad_targeted_batch():

    cmd = [
        "python", "untargeted/batch_process.py",
        "--video_list", "untargeted/kinetics400_val_list_videos.txt",
        "--sigma", "0.001",
    ]
    subprocess.run(cmd)

app = FastAPI()

# Endpoint for triggering the targeted VBAD attack
@app.post("/run-vbad-targeted")
async def run_vbad_targeted(background_tasks: BackgroundTasks):
    """
    Run targeted VBAD adversarial attack on pre-defined video list.
    """
    background_tasks.add_task(run_vbad_targeted_batch)
    return {"message": "Targeted VBAD attack started in background."}

@app.get("/health")
def health_check():
    return {"status": "OK"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8007)
