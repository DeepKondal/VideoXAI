from fastapi import FastAPI, HTTPException
import uvicorn
from adversarial_model import evaluate_adversarial

app = FastAPI()

@app.post("/adversarial-evaluation/")
async def adversarial_evaluation(config: dict):
    try:
        results = evaluate_adversarial(config)
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8004)
