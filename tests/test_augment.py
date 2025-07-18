
from a_augment import augment_images
from PIL import Image
import os

def test_augment_output_shape(tmp_path):
    img = Image.new('RGB', (100, 100))
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    output_dir.mkdir()
    img_path = input_dir / "test.jpg"
    img.save(img_path)
    augment_images(str(input_dir), str(output_dir), config_path="configs/augment.yaml")
    out_img = Image.open(output_dir / "test.jpg")
    assert out_img.size == (100, 100)
