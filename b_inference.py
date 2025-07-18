import os
import requests
import json
import time

MODEL_SERVER_URL = "http://localhost:8000/predict"
INFER_DIR = "results/inference_results"


def run_inference(input_dir):
    os.makedirs(INFER_DIR, exist_ok=True)
    predictions = []
    for fname in os.listdir(input_dir):
        img_path = os.path.join(input_dir, fname)
        print(f"Sending image {fname} for inference.")
        with open(img_path, 'rb') as img_file:
            response = requests.post(
                MODEL_SERVER_URL, files={'file': img_file})
        if response.status_code != 200:
            print(
                f"Warning: inference failed for {fname} with status {response.status_code}")
            continue
        pred = response.json()
        pred["file_name"] = fname
        predictions.append(pred)

        # Save the raw prediction per image for reference/debugging
        pred_path = os.path.join(INFER_DIR, fname + ".json")
        with open(pred_path, "w") as f:
            json.dump(pred, f, indent=2)
        print(f"Saved prediction output to {pred_path}")

    return predictions


def run_batch_inference(input_location: str, output_location: str, poll_interval: float = 5.0, timeout: float = 600.0):
    url = "http://localhost:8000/batch_predict"
    data = {"input_location": input_location,
            "output_location": output_location}

    try:
        # Submit job
        response = requests.post(url, json=data)
        if response.status_code != 200:
            raise Exception(
                f"Failed to start batch prediction: {response.status_code} {response.text}")

        job_info = response.json()
        job_id = job_info.get("job_id")
        if not job_id:
            raise Exception("No job_id returned from batch_predict")

        print(f"Batch prediction job started with job_id: {job_id}")

        # Poll the job status
        status_url = f"http://localhost:8000/batch_status/{job_id}"
        start_time = time.time()

        while True:
            status_resp = requests.get(status_url)
            if status_resp.status_code != 200:
                raise Exception(
                    f"Failed to get job status: {status_resp.status_code} {status_resp.text}")

            status_data = status_resp.json()
            job_status = status_data.get("status")
            print(f"Job status: {job_status}")

            if job_status in ("completed", "failed"):
                print(f"Job {job_status}!")
                return status_data

            elapsed = time.time() - start_time
            if elapsed > timeout:
                raise Exception(
                    f"Timeout exceeded ({timeout} sec) waiting for job completion")

            time.sleep(poll_interval)

    except Exception as e:
        print(f"Error during batch inference or polling: {e}")
        raise
