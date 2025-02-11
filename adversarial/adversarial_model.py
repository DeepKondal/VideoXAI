import torch
import av
import numpy as np
import os
import torch.nn.functional as F
from transformers import TimesformerForVideoClassification, AutoImageProcessor
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from tqdm import tqdm

def load_video(video_path, num_frames=8):
    frames = []
    container = av.open(video_path)
    
    for frame in container.decode(video=0):
        frames.append(frame.to_rgb().to_ndarray())
        if len(frames) == num_frames:
            break
    
    while len(frames) < num_frames:
        frames.append(frames[-1])
    
    return np.stack(frames)

def fgsm_attack(model, data, epsilon, labels, device):
    data.pixel_values.requires_grad = True
    
    outputs = model(**data)
    loss = F.cross_entropy(outputs.logits, labels)
    model.zero_grad()
    loss.backward()
    
    data_grad = data.pixel_values.grad.data.sign()
    perturbed_data = data.copy()
    perturbed_data.pixel_values = data.pixel_values + epsilon * data_grad
    perturbed_data.pixel_values = torch.clamp(perturbed_data.pixel_values, 0, 1)
    
    return perturbed_data

def evaluate_adversarial(config):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = TimesformerForVideoClassification.from_pretrained(config['model_name']).to(device)
    processor = AutoImageProcessor.from_pretrained(config['model_name'])
    model.eval()

    video_files = [f for f in os.listdir(config['video_directory']) if f.endswith('.mp4')]
    results = []

    for video_file in tqdm(video_files, desc="Processing videos"):
        video_path = os.path.join(config['video_directory'], video_file)
        
        frames = load_video(video_path)
        inputs = processor(list(frames), return_tensors="pt").to(device)
        
        with torch.no_grad():
            clean_outputs = model(**inputs)
            clean_pred = clean_outputs.logits.argmax(-1).cpu().numpy()[0]

        labels = torch.tensor([clean_pred]).to(device)  
        perturbed_inputs = fgsm_attack(model, inputs, config['epsilon'], labels, device)

        with torch.no_grad():
            adv_outputs = model(**perturbed_inputs)
            adv_pred = adv_outputs.logits.argmax(-1).cpu().numpy()[0]

        results.append({
            "video_file": video_file,
            "clean_prediction": clean_pred,
            "adversarial_prediction": adv_pred
        })

    return results
