import os
from ..parsedata import extract_labels
from ..parsedata import extract_images

images_file_path = os.path.join(os.getcwd(), "mnist", "train", "train-images-idx3-ubyte.gz")
labels_file_path = os.path.join(os.getcwd(), "mnist", "train", "train-labels-idx1-ubyte.gz")
images = extract_images(images_file_path)
labels = extract_labels(labels_file_path)

g_LastIndex = 0

def next_batch(batch_size, fake_data=False):
    if fake_data:
        fake_image = [1] * 784
        fake_label = 0
        return [fake_image for _ in range(batch_size)], [
            fake_label for _ in range(batch_size)
        ]



