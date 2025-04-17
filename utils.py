import os
import zipfile
import urllib.request

def download_dataset():
    url = "https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip"
    path = "data/cats_and_dogs_filtered.zip"
    if not os.path.exists("data"):
        os.mkdir("data")
    if not os.path.exists("data/cats_and_dogs_filtered"):
        print("Downloading dataset...")
        urllib.request.urlretrieve(url, path)
        with zipfile.ZipFile(path, 'r') as zip_ref:
            zip_ref.extractall("data")
        print("Dataset ready!")
