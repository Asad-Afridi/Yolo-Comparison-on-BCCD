import os
import zipfile
import yaml
import subprocess
# Load configuration from config.yaml
def load_config():
    with open("config/config.yaml", 'r') as stream:
        return yaml.safe_load(stream)

def download_dataset(dataset_url, download_path):
    print(f"Downloading dataset from {dataset_url}...")
    try:
        # Create the destination directory if it doesn't exist
        os.makedirs(download_path, exist_ok=True)
        
        # Construct the git clone command
        command = ["git", "clone", dataset_url, download_path]
        
        # Execute the command
        subprocess.run(command, check=True)
        print(f"Repository cloned successfully to: {os.path.abspath(download_path)}")
    except subprocess.CalledProcessError as e:
        print(f"Error cloning repository: {e}")
    except FileNotFoundError:
        print("Git command not found. Please ensure Git is installed and in your PATH.")

if __name__ == "__main__":
    config = load_config()
    dataset_url = config['dataset_url']
    download_path = os.path.join(config['raw_data_path'], 'BCCD_Dataset.zip')
    download_dataset(dataset_url, download_path)
