# Creates images.csv file. Must be run from root folder.
# Usage: python scripts/create_csv.py

import os
from csv import writer

def main():
    csv_file = open("images.csv", "w", newline="")
    csv = writer(csv_file)
    csv.writerow(["path", "shape", "type"])

    for shape_dir in os.scandir("images"):
        if not shape_dir.is_dir():
            continue
        shape = shape_dir.name
        for type_dir in os.scandir(os.path.join("images", shape)):
            if not type_dir.is_dir():
                continue
            type = type_dir.name
            for img_file in os.scandir(os.path.join("images", shape, type)):
                img_path = os.path.join("images", shape, type, img_file.name)
                csv.writerow([img_path, shape, type])
    
    csv_file.close()

if __name__ == "__main__":
    main()
