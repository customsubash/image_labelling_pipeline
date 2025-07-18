# Batch Labeling Pipeline

## Table of Contents
- [Overview](#overview)
- [Directory Structure](#directory-structure)
- [Prerequisites](#prerequisites)
- [Setup and Running the Pipeline](#setup-and-running-the-pipeline)
- [How It Works](#how-it-works)
- [Details of Components](#details-of-components)
- [Testing](#testing)
- [Notes](#notes)
- [Contact](#contact)

## Overview


```
+------------------+
|  Raw Input Images |
|  (input_images/)  |
+---------+--------+
          |
          |  (1) Load raw images
          v
+---------------------------+
+---------------------------+
|  Augmentation Component    |
|  (a_augment.py)            |
| - Loads transforms.json    |
| - Applies Albumentations   |
+---------------------------+
+---------------------------+
          |
          | (2) Save augmented images
          v
+-------------------+
| Augmented Images   |
| (results/aug_images/)      |
+---------+---------+
          |
          | (3) Send augmented images for inference
          v
+---------------------------+
+---------------------------+       +------------------------------+
|  Model Server Container    |<----->|  fish_detector model (model.pt)|
|  (b_inference.py - FastAPI call to port 8000)    |       +------------------------------+
|  - /predict endpoint       |
|  - /batch_predict endpoint |
+---------------------------+
+---------------------------+
          |
          |(4) Inference results saved per image JSON and Anotated images
          v
+----------------------+
| Inference Results     |
| (results/inference_results/)  |
+----------+-----------+
           |
           | (5) Aggregate all JSON predictions
           v
+----------------------+
+----------------------+
| COCO Formatter        |
| (c_coco_formatter.py)   |
| - Combines JSON files |
| - Outputs results.json|
+----------+-----------+
+----------------------+
           |
           | (6) Final COCO annotations
           v
+----------------------+
| Final Output File     |
| (results/coco_results/results.json)        |
+----------------------+
```

This repository contains a complete batch image labeling pipeline designed to:

- Host a PyTorch model (`fish_detector`) as a REST API using FastAPI inside a Docker container.
- Perform image augmentation using Albumentations with augmentations defined in a serialized JSON format.
- Run inference on a batch of augmented images by querying the hosted model server.
- Collect individual predictions and aggregate them into a single COCO-format JSON output.
- Support batch inference through a dedicated model server endpoint.
- Use Docker Compose for managing multi-container infrastructure, including model hosting and batch processing.
- Provide unit test examples to facilitate reliable development.

Each task is broken into independant subtask to make them independant to each other. It will also ensure modularity and configurability.

## Directory Structure

```
batch_labeling_pipeline/
├── input_images/                    # Put your raw input images here
├── results/(auto-created)
├── inference_results/               # Individual JSON predictions output directory (auto-created)
├── model_server/                   # Model server FastAPI app and Dockerfile
├──  ├── model/                          # Contains your fish_detector model files (e.g. model.pt)
│         └── model.pt
│    ├── app.py
│    └── Dockerfile
├── configs/
│    └── transforms.json            # Serialized Albumentations augmentation JSON config
├── a_augment.py                   # Augmentation script applying JSON-defined pipeline
├── b_inference.py                  # Sends augmented images to model server, saves per-image results
├── c_coco_formatter.py             # Converts all per-image JSON results to COCO format
├── infra_control.py              # Infrastructure setup and teardown commands using Docker Compose
├── main.py                      # Orchestrates the entire pipeline
├── requirements.txt             # Python dependencies for pipeline container
├── Dockerfile                   # Dockerfile for pipeline container (running main.py)
├── docker-compose.yml           # Compose file
```


## Prerequisites

- Docker and Docker Compose installed and running on your machine.
- Python 3.10+ if you want to run unit tests or parts of the pipeline outside Docker.
- Model file `model.pt` placed in the `model/` directory.
- Input images placed inside `input_images/` folder.

## Setup and Running the Pipeline

### 1. Build and start everything with Docker Compose:

```bash
docker compose up --build
````

It will take time to setup and install dependancies, could be optimized by creating and host the image.
or,
```bash
python main.py # you might have to setup virtual env before running and install requirements manually
````

This will:

* Build and start the model server exposing the PyTorch model on port 8000.
* Wait for the model server to be ready (using healthchecks).
* Run the batch labeling pipeline container which:

  * Augments images using `configs/transforms.json`.
  * Sends augmented images for inference to the model server.
  * Saves per-image predictions in `results/inference_results/`.
  * Aggregates predictions into COCO JSON format saved as `results/coco_results/results.json`.
* `aug_images/`, `inference_results/` and `coco_results/` directories are created inside `results/` automatically on each fresh on each run.

### 2. When finished, shut down services and clean up:

```bash
docker compose down
```

## How It Works

**Model Server**

A FastAPI app (`model_server/app.py`) loads the `fish_detector` model and exposes two endpoints:

   * `/predict` for single image inference (used by the pipeline).
   * `/batch_predict` to start batch inference on a server folder asynchronously.

**Augmentation**

   `a_augment.py` loads the Albumentations augmentation pipeline from `configs/transforms.json` and applies it to all images in `input_images/`, saving augmented images to `aug_images/`.

**Inference**

   `inference.py` sends each augmented image to the model server’s `/batch_predict` endpoint, saves each JSON response in `inference_results/`.

**COCO Formatting**

   `coco_formatter.py` reads all JSON prediction files from `inference_results/`, merges them into a single COCO-format JSON file `results.json`.

**Infrastructure Control**

   `infra_control.py` handles starting and stopping Docker Compose services, and manages intermediate folders.

**Orchestration**

   `main.py` executes the full pipeline steps in order.

## Details of Components

* **Model Server (`model_server/app.py`)**
  FastAPI app serving the model, supports:

  * Single image prediction via `/predict`.
  * Batch prediction of all images in a given folder via `/batch_predict` (runs asynchronously).

* **Augmentation Pipeline (`a_augment.py`)**
  Loads augmentation pipeline from `configs/transforms.json` using Albumentations serialization API and applies to images.

* **Inference Client (`b_inference.py`)**
  Reads augmented images, sends them to model server, and saves individual prediction JSONs.

* **COCO Formatter (`c_coco_formatter.py`)**
  Aggregates all per-image JSONs into COCO format, managing unique IDs and metadata.

* **Infrastructure (`infra_control.py`)**
  Uses Docker Compose to orchestrate model server and batch labeling client.

* **Docker Compose (`docker-compose.yml`)**
  Defines two services: `model-server` and `batch-labeling` with healthchecks ensuring correct startup order.

* **Unit Tests (`tests/`)**
  Examples given for augmentation, inference mocking, and COCO formatting.

## Testing

Run unit tests locally (inside your Python environment or Docker container with `pytest` installed):

```bash
pytest tests/
```

Tests include:

* Checking augmentation outputs sizes remain consistent.
* Mocking inference server responses.
* Verifying COCO formatter output structure.

## Notes

* Ensure `model.pt` corresponds exactly to your `fish_detector` model and its expected preprocessing.
* Edit category labels inside `coco_formatter.py` as per your use case.
* You can customize augmentations by updating `configs/transforms.json`.
* Intermediate results are saved in dedicated folders (`results/aug_images/`, `results/inference_results/`) and cleaned at pipeline start.
* The batch inference endpoint expects folder paths  `/results` accessible within the model server container.
* For large datasets consider enhancing pipeline with job queues and monitoring.

## Contact

For further questions or support, contact the author or open an issue.

---

Hope it will run smoothly on your machine!