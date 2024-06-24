import os
import shutil
import sys
from sklearn.model_selection import train_test_split


def split_data(data_dir, train_ratio=0.8, seed=42):
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")

    # Ensure the train and val directories exist
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    # Iterate over each class folder
    for class_name in os.listdir(data_dir):
        class_path = os.path.join(data_dir, class_name)
        if os.path.isdir(class_path) and class_name not in ["train", "val"]:
            # Get a list of all files in the class directory
            files = os.listdir(class_path)
            files = [f for f in files if os.path.isfile(os.path.join(class_path, f))]

            # Split the files into training and validation sets
            train_files, val_files = train_test_split(
                files, train_size=train_ratio, random_state=seed
            )

            # Create class directories in train and val directories
            train_class_dir = os.path.join(train_dir, class_name)
            val_class_dir = os.path.join(val_dir, class_name)
            os.makedirs(train_class_dir, exist_ok=True)
            os.makedirs(val_class_dir, exist_ok=True)

            # Move training files
            for file in train_files:
                shutil.move(
                    os.path.join(class_path, file), os.path.join(train_class_dir, file)
                )

            # Move validation files
            for file in val_files:
                shutil.move(
                    os.path.join(class_path, file), os.path.join(val_class_dir, file)
                )

    print("Data split complete.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python split_data.py <data_dir>")
        sys.exit(1)

    data_dir = sys.argv[1]
    split_data(data_dir)
