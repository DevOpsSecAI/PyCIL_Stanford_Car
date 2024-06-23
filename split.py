import os
import random
import shutil
import argparse


def split_files(source_folder, train_ratio=0.8):
    train_folder = os.path.join(source_folder, "train")
    test_folder = os.path.join(source_folder, "test")

    # Ensure the train and test folders exist
    try:
        os.makedirs(train_folder, exist_ok=True)
        os.makedirs(test_folder, exist_ok=True)
    except FileExistsError as e:
        print(f"Directory {test_folder} already exists. {e}")

    # Get all files in the source folder, excluding the train and test folders
    all_files = [
        f
        for f in os.listdir(source_folder)
        if os.path.isfile(os.path.join(source_folder, f))
    ]

    if len(all_files) < 2:
        raise ValueError(
            "Not enough files to split. The source folder must contain at least 2 files."
        )

    # Shuffle the list of files
    random.shuffle(all_files)

    # Calculate the number of training files
    train_count = max(1, int(len(all_files) * train_ratio))
    test_count = max(1, len(all_files) - train_count)

    # Adjust if we have too many files in one category due to rounding
    if train_count + test_count > len(all_files):
        train_count -= 1

    # Split the files
    train_files = all_files[:train_count]
    test_files = all_files[train_count : train_count + test_count]

    # Move the files to their respective folders
    for file in train_files:
        shutil.move(os.path.join(source_folder, file), os.path.join(train_folder, file))

    for file in test_files:
        shutil.move(os.path.join(source_folder, file), os.path.join(test_folder, file))

    print(f"Moved {len(train_files)} files to {train_folder}")
    print(f"Moved {len(test_files)} files to {test_folder}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Split files in a folder into train and test sets."
    )
    parser.add_argument(
        "-source",
        type=str,
        required=True,
        help="The path to the source folder containing the files to split.",
    )
    args = parser.parse_args()

    split_files(args.source)
