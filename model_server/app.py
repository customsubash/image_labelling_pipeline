import io
import os
import glob
import json
import uuid
import time
import threading
from fastapi import FastAPI, File, UploadFile, BackgroundTasks, HTTPException, Path
from fastapi.responses import JSONResponse
from PIL import Image, ImageDraw
import torch
import torchvision.ops as ops
import numpy as np

# Paths and config
MODEL_PATH = "./model/model.pt"
CLASS_MAPPING_PATH = "./model/class_mapping.json"

# Load class mapping
with open(CLASS_MAPPING_PATH) as f:
    mappings = json.load(f)
class_mapping = {item['model_idx']: item['class_name'] for item in mappings}

# Device and model loading
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torch.jit.load(MODEL_PATH).to(device)

app = FastAPI(title="Fish Detector Model Server with Batch Job Tracking")

# In-memory job store: job_id -> {status, submitted_at, completed_at, input_location, output_location, error, usage}
job_store = {}

def predict_image(image: Image.Image):
    np_image = np.array(image)
    x = torch.from_numpy(np_image).to(device).permute(2,0,1).float()
    with torch.no_grad():
        y = model(x)
        to_keep = ops.nms(y['pred_boxes'], y['scores'], 0.3)
        pred_boxes = y['pred_boxes'][to_keep].cpu().int().tolist()
        pred_scores = y['scores'][to_keep].cpu().tolist()
        pred_classes = y['pred_classes'][to_keep].cpu().tolist()

    bboxes = [
        {
            "category_id": c,
            "class_name": class_mapping.get(c, "Unknown"),
            "bbox": box,
            "score": score
        }
        for box, score, c in zip(pred_boxes, pred_scores, pred_classes)
    ]
    return {
        "width": image.width,
        "height": image.height,
        "bboxes": bboxes
    }

def draw_predictions_on_image(image: Image.Image, prediction: dict) -> Image.Image:
    draw = ImageDraw.Draw(image)
    for bbox_info in prediction.get("bboxes", []):
        bbox = bbox_info["bbox"]
        score = bbox_info.get("score", 1.0)
        class_name = bbox_info.get("class_name", "Unknown")
        draw.rectangle(bbox, outline="red", width=2)
        draw.text((bbox[0], max(0, bbox[1]-15)), f"{class_name} {score:.2f}", fill="red")
    return image

def run_batch_prediction(job_id: str, input_location: str, output_location: str):
    """Process batch prediction and update job status accordingly."""
    try:
        job_store[job_id]['status'] = 'running'
        job_store[job_id]['started_at'] = time.time()

        os.makedirs(output_location, exist_ok=True)
        image_paths = []
        exts = ('*.jpg', '*.jpeg', '*.png', '*.bmp')
        for ext in exts:
            image_paths.extend(glob.glob(os.path.join(input_location, ext)))

        if not image_paths:
            job_store[job_id]['status'] = 'failed'
            job_store[job_id]['error'] = "No images found in input directory"
            return

        image_count = 0
        for img_path in image_paths:
            image_count += 1
            try:
                image = Image.open(img_path).convert("RGB")
                pred = predict_image(image)

                base_name = os.path.basename(img_path)
                json_path = os.path.join(output_location, base_name + ".json")
                with open(json_path, "w") as f_json:
                    json.dump(pred, f_json, indent=2)

                image_with_boxes = draw_predictions_on_image(image.copy(), pred)
                image_save_path = os.path.join(output_location, base_name)
                image_with_boxes.save(image_save_path)
            except Exception as ex_img:
                print(f"Error processing image {img_path}: {ex_img}")

        job_store[job_id]['status'] = 'completed'
        job_store[job_id]['completed_at'] = time.time()
        job_store[job_id]['usage'] = {
            "images_processed": image_count,
            "output_location": output_location,
            "input_location": input_location,
        }
    except Exception as e:
        job_store[job_id]['status'] = 'failed'
        job_store[job_id]['error'] = str(e)

from pydantic import BaseModel

class BatchPredictRequest(BaseModel):
    input_location: str
    output_location: str

@app.post("/batch_predict")
async def batch_predict(request: BatchPredictRequest, background_tasks: BackgroundTasks):
    input_location = request.input_location
    output_location = request.output_location

    if not os.path.isdir(input_location):
        raise HTTPException(status_code=400, detail=f"Input folder '{input_location}' does not exist")

    job_id = str(uuid.uuid4())
    job_store[job_id] = {
        "status": "pending",
        "submitted_at": time.time(),
        "input_location": input_location,
        "output_location": output_location,
        "error": None,
        "usage": None,
    }

    background_tasks.add_task(run_batch_prediction, job_id, input_location, output_location)
    return {"message": "Batch prediction started", "job_id": job_id}

@app.get("/batch_status/{job_id}")
async def batch_status(job_id: str = Path(..., description="The job ID to query status for")):
    job = job_store.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job ID not found")

    return {
        "job_id": job_id,
        "status": job["status"],
        "submitted_at": job["submitted_at"],
        "started_at": job.get("started_at"),
        "completed_at": job.get("completed_at"),
        "error": job.get("error"),
        "usage": job.get("usage"),
    }

# Single image prediction endpoint unchanged

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    prediction = predict_image(image)
    return prediction


@app.get("/healthcheck")
async def predict():
    return "API is up and running."
