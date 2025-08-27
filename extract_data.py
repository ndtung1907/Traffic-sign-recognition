import os
import requests
import zipfile

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

dataset_url = "https://d17h27t6h515a5.cloudfront.net/topher/2017/February/5898cd6f_traffic-signs-data/traffic-signs-data.zip"
dataset_zip = os.path.join(DATA_DIR, "traffic-signs-data.zip")
train_p_path = os.path.join(DATA_DIR, "train.p")

if not os.path.exists(train_p_path):
    print("Downloading and extracting dataset...")

    if not os.path.exists(dataset_zip):
        print("Downloading dataset...")
        response = requests.get(dataset_url, stream=True)
        with open(dataset_zip, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
    
    with zipfile.ZipFile(dataset_zip, "r") as zip_ref:
        zip_ref.extractall(DATA_DIR)
    print("Done.")