import os
from ..parsedata import extract_labels
from ..parsedata import extract_images

images_file_path = os.path.join(os.getcwd(), "mnist", "test", "t10k-images-idx3-ubyte.gz")
labels_file_path = os.path.join(os.getcwd(), "mnist", "test", "t10k-labels-idx1-ubyte.gz")

images = extract_images(images_file_path)
labels = extract_labels(labels_file_path)