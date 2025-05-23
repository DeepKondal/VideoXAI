
# VideoXAI

## Overview

**VideoXAI** is an end-to-end explainable AI pipeline for video classification, built as an extension to [XAIport](https://github.com/ZeruiW/XAIport) framework. While XAIport was designed for static data like images and tabular inputs, VideoXAI expands it to support dynamic video content through transformer-based modeling, adversarial robustness analysis, spatio-temporal attention attribution, and natural language querying.

This system combines clean architecture with interpretability and robustness evaluation. It processes video inputs, injects adversarial perturbations, performs classification with attention extraction, and enables semantic retrieval using Retrieval-Augmented Generation (RAG) based on stored attention data.

---

### Key Features

- **Full Video Explainability Pipeline**: From raw input to interpretable output.
- **Attack Injection**: Supports FGSM and VBAD (both targeted and untargeted) attacks to evaluate model vulnerabilities.
- **Transformer-Based Modeling**: Uses models like TimeSformer to extract spatial and temporal attention maps from videos.
- **Spatio-Temporal Attention Attribution (STAA)**: Computes fine-grained attention visualizations over space and time using internal transformer attention layers.
- **Provenance and Vector Indexing**: Stores frame-level attention embeddings in Pinecone, indexed by attack type and model version.
- **RAG-based Query System**: Enables natural language querying of model behavior using LLaMA 3.2 via Ollama.
- **Scalable & Reproducible**: The system is designed to operate flexibly in both environments—locally as a modular microservice-based architecture, and at scale via job submission on a HPC cluster, ensuring reproducibility and performance under varied deployment settings.

---

### System Architecture

The VideoXAI architecture is composed of the following modular microservices:

- `Data Processing`: Frame extraction and adversarial perturbation.
- `Model Inference`: Patch projection, classification, and attention extraction using vision transformers.
- `XAI Service (STAA)`: Attribution of spatial and temporal importance using multi-head attention layers.
- `Provenance & RAG`: Vector storage in Pinecone and semantic retrieval via LLaMA-based RAG.
- `Coordinator`: Manages data flow and orchestration between services.

---

### Architecture Diagram

![VideoXAI Pipeline](assets/videoxai_diagram.png)

---

### Technologies Used

- Python, PyTorch, Transformers (TimeSformer)
- Pinecone Vector Database
- Ollama + LLaMA 3.2
- FGSM, VBAD (I3D-based) Attacks
  
---

### Use Cases

- Analyze and visualize model decisions on video data.
- Evaluate model robustness under adversarial attacks.
- Retrieve and compare attention explanations via natural language queries.
- Extend to other video-based tasks requiring interpretability.

---

## Initial Setup

### Prerequisites

- Python 3.8 or later
- FastAPI
- httpx
- uvicorn
- Dependencies as listed in `requirements_fixed.txt`

### Installation Guide

1. **Environment Setup**:
   Ensure Python is installed on your system. It's recommended to use a virtual environment for Python projects:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

2. **Install Dependencies**:
   Install the necessary Python libraries with pip:

   ```bash
   pip install -r requirements_fixed.txt
   ```

3. **Clone the Repository**:
   Clone the repository to get the latest codebase:

   ```bash
   git clone https://github.com/DeepKondal/VideoXAI.git
   cd XAIport
   ```

### Configuration
Before running the system, configure all necessary details such as API endpoints, database connections, and service-related configurations using a JSON file. Adjust the config.json file based on the task—either for VBAD (Targeted/Untargeted) or Adversarial (FGSM) attacks.

Example `config.json` for VBAD attacks:

```json
{
  "upload_config": {
    "server_url": "http://127.0.0.1:8001",
    "datasets": {
      "kinetics_400": {
        "local_video_dir": "dataprocess/raw_videos"
      }
    }
  },
 "perturbation_config": {
  "servers": {
    "vbad_targeted": "http://127.0.0.1:8007",
    "vbad_untargeted": "http://127.0.0.1:8006"
  }
  },
  "model_config": {
    "base_urls": [
      "http://127.0.0.1:8010",
      "http://127.0.0.1:8011"
    ],
    "models": {
      "kinetics_video": {
        "model_name": "facebook/timesformer-base-finetuned-k400",
        "targeted_dir": "untargeted/final_perturbed_videos/targeted",
        "untargeted_dir": "untargeted/final_perturbed_videos/untargeted",
        "num_frames": 8
      }
    }
  },
  "xai_config": {
    "base_url": "http://127.0.0.1:8003",
    "datasets": {
      "kinetics_video": {
        "targeted_path": "untargeted/final_perturbed_videos/targeted",
        "untargeted_path": "untargeted/final_perturbed_videos/untargeted",
        "num_frames": 8
      }
    }
  }
}

```
config.json for adversarial file 
``` json
{
  "upload_config": {
    "server_url": "http://127.0.0.1:8001",
    "datasets": {
      "kinetics_400": {
        "local_video_dir": "dataprocess/videos"
      }
    }
  },
  "perturbation_config": {
    "server_url": "http://127.0.0.1:8001",
    "datasets": {
      "kinetics_400": {
        "perturbation_type": "adversarial_attack",
        "severity": 1,
        "video_directory": "dataprocess/videos"
      }
    }
  },
  "model_config": {
    "base_urls": [
      "http://127.0.0.1:8002",
      "http://127.0.0.1:8005"
    ],
    "models": {
      "kinetics_video": {
        "model_name": "facebook/timesformer-base-finetuned-k400",
        "original_video_dir": "dataprocess/videos",
        "adversarial_video_dir": "dataprocess/FGSM",
        "num_frames": 8
      }
    }
  },
  "xai_config": {
    "base_url": "http://127.0.0.1:8003",
    "datasets": {
      "kinetics_video": {
        "video_path": "dataprocess/videos",
        "adversarial_video_path": "dataprocess/FGSM",
        "num_frames": 8
      }
    }
  }
}

```
##  Required Servers to Run the VBAD Pipeline
Start the following services before executing the pipeline:
```
python dataprocess/dataprocess_server.py
python modelserver_untargeted/model_server.py
python modelserver_targeted/model_server.py
python untargeted/untargeted_server.py
python targeted/targeted_server.py
python xaiserver/xai_server.py
python center_server.py
```
## For FGSM Pipeline 
```
python dataprocess/dataprocess_server.py
python modelserver/model_server.py
python modelserver_adversarial/model_server.py
python xaiserver/xai_server.py
python center_server1.py
```
### Starting the Service

### Using the API

- **Execute XAI Task**:

  ```bash
  curl -X POST "http://127.0.0.1:8880/run_pipeline/" \
  -H "Content-Type: application/json" \
  --data-binary "@task_sheets/task.json"
  ```

All processed outputs will be stored in:

- `output/` directory → for **FGSM (Adversarial)** pipeline  
- `output1/` directory → for **VBAD (Targeted/Untargeted)** pipeline  

These include video frames, attention values, and model predictions.

Note: During the VBAD pipeline, ensure that the video you are processing has the exact corresponding label number listed in the kinetics400_val_list_videos.txt file located inside the untargeted/ directory.

### Running on HPC Cluster (SLURM-based)
VideoXAI is designed to be deployable not only as a local microservice pipeline but also as a batch job on HPC systems (e.g., Concordia’s Speed cluster). This enables efficient large-scale video explainability workflows in research environments.

The SLURM job script is also included in this repository as: `HPC_script_VideoXai.txt`
You can directly download and submit the job using: sbatch HPC_script_VideoXai.txt

### 📖 Citation

If you use this project in your research, please cite:

```bibtex
@misc{videoxai2025,
  author       = {Kondal, Abideep Singh}{Singh, Ravinder},
  title        = {VideoXAI: An Explainable AI Pipeline for Robust Video Classification},
  year         = {2025},
  note         = {Extension of XAIport Framework},
  url          = {https://github.com/DeepKondal/VideoXAI}
}
