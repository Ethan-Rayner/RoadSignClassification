# Takes images from SOURCE_DIRECTORY and makes them 28x28 greyscale images.
# Image are saved to DESTINATION_DIRECTORY.
# Usage: python regularize_images.py

FINAL_HEIGHT = 28
FINAL_WIDTH = 28
SOURCE_DIRECTORY = "test-images-raw"
DESTINATION_DIRECTORY = "test-images"

import os
import shutil
import PIL

def main():
    if os.path.exists(DESTINATION_DIRECTORY):
        shutil.rmtree(DESTINATION_DIRECTORY)

    for img_file in os.scandir(SOURCE_DIRECTORY):
        img_name = img_file.name
        img_path = os.path.join(SOURCE_DIRECTORY, img_name)
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
        new_img.save(os.path.join(DESTINATION_DIRECTORY, shape, type, final_file_name))

def make_dirs(shape, type):
    if not os.path.exists(DESTINATION_DIRECTORY):
        os.mkdir(DESTINATION_DIRECTORY)
    if not os.path.exists(os.path.join(DESTINATION_DIRECTORY, shape)):
        os.mkdir(os.path.join(DESTINATION_DIRECTORY, shape))
    if not os.path.exists(os.path.join(DESTINATION_DIRECTORY, shape, type)):
        os.mkdir(os.path.join(DESTINATION_DIRECTORY, shape, type))

if __name__ == "__main__":
    main()
