import datetime
import os

from a_augment import augment_images
from b_inference import run_inference, run_batch_inference
from c_coco_formatter import save_as_coco, save_as_coco_from_files
from infra_control import setup_infrastructure, teardown_infrastructure

def main():
    current_time = datetime.datetime.now().strftime("%A, %B %d, %Y, %I:%M %p %z")
    print(f"Pipeline started on: {current_time}\n")

    try:
        print("#" * 80)
        print("STEP 1: Setting up infrastructure...")
        setup_infrastructure()
        print("Infrastructure setup completed.\n")

        print("#" * 80)
        print("STEP 2: Running image augmentation...")
        augment_images(
            input_folder='input_images',
            output_folder='results/aug_images',
            transform_json_path='config/transforms.json'
        )
        print("Image augmentation completed.\n")

        print("#" * 80)
        print("STEP 3: Running batch inference on augmented images...")
        result = run_batch_inference(
            input_location='results/aug_images',
            output_location='results/inference_results'
        )
        print("Batch inference started.\n")

        print("#" * 80)
        print("STEP 4: Converting predictions to COCO format...")
        os.makedirs('results/coco_results', exist_ok=True)
        save_as_coco_from_files(
            predictions_folder="results/inference_results",
            output_file='results/coco_results/results.json'
        )
        print("COCO format saved successfully.\n")

    except Exception as e:
        print(f"ERROR encountered: {e}")
    finally:
        print("#" * 80)
        print("STEP 5: Tearing down infrastructure...")
        teardown_infrastructure()
        print("Infrastructure teardown completed.\n")
        print("Pipeline completed successfully.")

if __name__ == '__main__':
    main()
