import argparse
import yaml
from download_dataset import download_dataset
from preprocess import preprocess_data

def main():
    # Load the config
    with open('config/config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    # Download the dataset
    download_dataset(config['dataset_url'], config['raw_data_path'] + '/BCCD_Dataset.zip')

    # Preprocess the data
    preprocess_data(config['raw_images_path'], config['raw_annotations_path'], config['processed_images_path'], config['processed_annotations_path'])

if __name__ == "__main__":
    main()
