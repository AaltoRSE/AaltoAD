import os
import numpy as np

from AaltoAD.constants import DEFAULT_DATA_FOLDER, DEFAULT_OUTPUT_FOLDER


def process_csv_file(
    category, filename, dataset, dataset_folder, output_folder=DEFAULT_OUTPUT_FOLDER
):
    """Load a CSV file and save as normalized numpy array.

    Args:
            category (str): Subdirectory name ('train', 'test', etc.).
            filename (str): Name of the file to load.
            dataset (str): Dataset identifier for output filename.
            dataset_folder (str): Root path to dataset folder.

    Returns:
            tuple: Shape of loaded array as (num_samples, num_features).
    """
    temp = np.genfromtxt(
        os.path.join(dataset_folder, category, filename),
        dtype=np.float64,
        delimiter=",",
    )
    print(dataset, category, filename, temp.shape)
    np.save(os.path.join(output_folder, f"SMD/{dataset}_{category}.npy"), temp)
    return temp.shape


def process_label_file(
    category,
    filename,
    dataset,
    dataset_folder,
    shape,
    output_folder=DEFAULT_OUTPUT_FOLDER,
):
    """Load anomaly labels from interpretation label file and save as binary numpy array.

    Args:
            category (str): Label category name.
            filename (str): Name of the label file to parse.
            dataset (str): Dataset identifier for output filename.
            dataset_folder (str): Root path to dataset folder.
            shape (tuple): Target shape (num_samples, num_features) for labels array.

    Returns:
            None. Saves binary label array as '{dataset}_{category}.npy'.

    File Format Expected:
            Lines formatted as 'start-end:feature1,feature2,...' where start and end are indices
            and features are 1-indexed column numbers to mark as anomalous.
    """
    temp = np.zeros(shape)
    with open(os.path.join(dataset_folder, "interpretation_label", filename), "r") as f:
        ls = f.readlines()
    for line in ls:
        pos, values = line.split(":")[0], line.split(":")[1].split(",")
        start, end, indx = (
            int(pos.split("-")[0]),
            int(pos.split("-")[1]),
            [int(i) - 1 for i in values],
        )
        temp[start - 1 : end - 1, indx] = 1
    print(dataset, category, filename, temp.shape)
    np.save(os.path.join(output_folder, f"SMD/{dataset}_{category}.npy"), temp)



def load_SMD(folder):
    dataset_folder = os.path.join(DEFAULT_DATA_FOLDER, "SMD")
    train_dir = os.path.join(dataset_folder, "train")
    test_dir = os.path.join(dataset_folder, "test")

    if not os.path.isdir(train_dir):
        raise FileNotFoundError(f"Expected SMD train directory not found: {train_dir}")

    file_list = os.listdir(train_dir)
    for filename in file_list:
        if filename.endswith(".txt"):
            name = filename[:-4]
            process_csv_file("train", filename, name, dataset_folder)
            s = process_csv_file("test", filename, name, dataset_folder)
            process_label_file("labels", filename, name, dataset_folder, s)
