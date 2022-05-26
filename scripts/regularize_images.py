# Takes images from test-images-raw and makes them 28x28 greyscale images.
# Image are saved to test-images.
# Usage: python scripts/regularize_images.py

FINAL_HEIGHT = 28
FINAL_WIDTH = 28

import os
import shutil
import PIL
from PIL import Image

def main():
    if os.path.exists("test-images"):
        shutil.rmtree("test-images")

    for img_file in os.scandir("test-images-raw"):
        img_name = img_file.name
        img_path = os.path.join("test-images-raw", img_name)
        img = PIL.Image.open(img_path)

        shape = img_name.split("-")[0]
        type = img_name.split("-")[1]
        final_file_name = img_name.split("-")[2]

        new_img = PIL.ImageOps.grayscale(img)

        in_size = min(img.size[0], img.size[1])
        in_box = (
            (img.size[0] - in_size) / 2,
            (img.size[1] - in_size) / 2,
            in_size,
            in_size)

        out_size = (FINAL_WIDTH, FINAL_HEIGHT)

        new_img = new_img.resize(out_size, box = in_box)

        make_dirs(shape, type)
        new_img.save(os.path.join("test-images", shape, type, final_file_name))

def make_dirs(shape, type):
    if not os.path.exists("test-images"):
        os.mkdir("test-images")
    if not os.path.exists(os.path.join("test-images", shape)):
        os.mkdir(os.path.join("test-images", shape))
    if not os.path.exists(os.path.join("test-images", shape, type)):
        os.mkdir(os.path.join("test-images", shape, type))

if __name__ == "__main__":
    main()
