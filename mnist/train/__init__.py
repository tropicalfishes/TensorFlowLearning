import os
from ..parsedata import extract_labels
from ..parsedata import extract_images

images_file_path = os.path.join(os.getcwd(), "mnist", "train", "train-images-idx3-ubyte.gz")
labels_file_path = os.path.join(os.getcwd(), "mnist", "train", "train-labels-idx1-ubyte.gz")
images = extract_images(images_file_path)
labels = extract_labels(labels_file_path)


