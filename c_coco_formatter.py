import json
import os


def save_as_coco(preds, output_file):
    images, annotations = [], []
    for image_id, pred in enumerate(preds):
        images.append({
            "id": image_id,
            "file_name": pred["file_name"],
            "width": pred.get("width", 0),
            "height": pred.get("height", 0)
        })
        for ann_id, bbox in enumerate(pred.get("bboxes", [])):
            annotations.append({
                "id": ann_id,
                "image_id": image_id,
                "category_id": bbox["category_id"],
                "bbox": bbox["bbox"],
                "score": bbox.get("score", 1.0),
                "area": bbox["bbox"][2] * bbox["bbox"][3],
                "iscrowd": 0
            })
    coco_dict = {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 0, "name": "object"}]  # Update categories as required
    }
    with open(output_file, 'w') as f:
        json.dump(coco_dict, f)


def save_as_coco_from_files(predictions_folder, output_file):
    images = []
    annotations = []
    ann_id = 0  # Unique annotation ID across all images

    json_files = [f for f in os.listdir(predictions_folder) if f.endswith('.json')]

    print(json_files)

    for image_id, json_fname in enumerate(sorted(json_files)):
        json_path = os.path.join(predictions_folder, json_fname)
        with open(json_path, 'r') as f:
            pred = json.load(f)

        images.append({
            "id": image_id,
            "file_name": pred.get("file_name", json_fname.replace('.json','')),
            "width": pred.get("width", 0),
            "height": pred.get("height", 0)
        })

        for bbox in pred.get("bboxes", []):
            bbox_area = bbox["bbox"][2] * bbox["bbox"][3]
            annotations.append({
                "id": ann_id,
                "image_id": image_id,
                "category_id": bbox.get("category_id", 0),
                "bbox": bbox.get("bbox", []),
                "score": bbox.get("score", 1.0),
                "area": bbox_area,
                "iscrowd": 0
            })
            ann_id += 1

    coco_dict = {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 0, "name": "object"}]  # Update categories if needed
    }

    with open(output_file, 'w') as f:
        json.dump(coco_dict, f, indent=2)

    print(f"Saved combined COCO annotations to {output_file}")


if __name__ == "__main__":
    save_as_coco_from_files(predictions_folder="results/inference_results", output_file='results/coco_results/results.json')