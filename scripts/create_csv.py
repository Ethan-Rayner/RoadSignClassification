# Creates images.csv file. Must be run from root folder.
# Usage: python scripts/create_csv.py [images/test-images]

import sys
import os
from csv import writer

def main():
    if len(sys.argv) != 2:
        print("Usage: python scripts/create_csv.py [images/test-images]")
        return
    
    images_dir = sys.argv[1]

    csv_file = open(images_dir + ".csv", "w", newline="")
    csv = writer(csv_file)
    csv.writerow(["path", "shape", "type"])

    for shape_dir in os.scandir(images_dir):
        if not shape_dir.is_dir():
            continue
        shape = shape_dir.name
        for type_dir in os.scandir(os.path.join(images_dir, shape)):
            if not type_dir.is_dir():
                continue
            type = type_dir.name
            for img_file in os.scandir(os.path.join(images_dir, shape, type)):
                img_path = os.path.join(images_dir, shape, type, img_file.name)
                csv.writerow([img_path, shape, type])
    
    csv_file.close()

if __name__ == "__main__":
    main()
