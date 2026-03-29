import os
import requests
import zipfile
from io import BytesIO

def download_tiny_imagenet(dest_dir='./data'):
    """
    Downloads and extracts Tiny ImageNet into the recommended data directory.
    """
    url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
    
    # Ensure the 'data/' directory exists 
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
        print(f"Created directory: {dest_dir}")

    print("Downloading Tiny ImageNet (this may take a minute)...")
    response = requests.get(url)
    
    if response.status_code == 200:
        print("Extracting files...")
        with zipfile.ZipFile(BytesIO(response.content)) as zip_ref:
            # Extracting into the data folder as per the project skeleton 
            zip_ref.extractall(dest_dir)
        print(f"Dataset ready in {dest_dir}/tiny-imagenet-200")
    else:
        print(f"Download failed. Status code: {response.status_code}")

if __name__ == "__main__":
    download_tiny_imagenet()