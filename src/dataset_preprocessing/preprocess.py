import os
import shutil
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import yaml

# Load configuration from config.yaml
def load_config():
    with open("config/config.yaml", 'r') as stream:
        return yaml.safe_load(stream)

# Preprocess data into YOLO format
def preprocess_data(raw_images_path, raw_annotations_path, processed_images_path, processed_annotations_path):
    # Create necessary directories if they don't exist
    os.makedirs(processed_images_path, exist_ok=True)
    os.makedirs(processed_annotations_path, exist_ok=True)

    annotations = sorted(os.listdir(raw_annotations_path))
    df = []
    cnt = 0
    for file in annotations:
        prev_filename = file.split('.')[0] + '.jpg'
        filename = str(cnt) + '.jpg'
        row = []
        parsedXML = ET.parse(os.path.join(raw_annotations_path, file))
        for node in parsedXML.getroot().iter('object'):
            blood_cells = node.find('name').text
            xmin = int(node.find('bndbox/xmin').text)
            xmax = int(node.find('bndbox/xmax').text)
            ymin = int(node.find('bndbox/ymin').text)
            ymax = int(node.find('bndbox/ymax').text)

            row = [prev_filename, filename, blood_cells, xmin, xmax, ymin, ymax]
            df.append(row)
        cnt += 1

    data = pd.DataFrame(df, columns=['prev_filename', 'filename', 'cell_type', 'xmin', 'xmax', 'ymin', 'ymax'])
    data[['prev_filename','filename', 'cell_type', 'xmin', 'xmax', 'ymin', 'ymax']].to_csv('dataset/blood_cell_detection.csv', index=False)

    # Normalize coordinates
    def width(df):
        return int(df.xmax - df.xmin)

    def height(df):
        return int(df.ymax - df.ymin)

    def x_center(df):
        return int(df.xmin + (df.width / 2))

    def y_center(df):
        return int(df.ymin + (df.height / 2))
    
    df  = pd.read_csv('dataset/blood_cell_detection.csv')

    df['width'] = df.apply(width, axis=1)
    df['height'] = df.apply(height, axis=1)
    df['x_center'] = df.apply(x_center, axis=1)
    df['y_center'] = df.apply(y_center, axis=1)

    # Normalize coordinates to YOLO format
    df['x_center_norm'] = df['x_center'] / 640
    df['width_norm'] = df['width'] / 640
    df['y_center_norm'] = df['y_center'] / 480
    df['height_norm'] = df['height'] / 480

    # Label encoding for cell types
    le = preprocessing.LabelEncoder()
    le.fit(df['cell_type'])
    df['labels'] = le.transform(df['cell_type'])

    # Split data into training and validation sets
    df_train, df_valid = train_test_split(df, test_size=0.1, random_state=13, shuffle=True)

    # Create directories for processed data
    os.makedirs(processed_images_path + '/train/', exist_ok=True)
    os.makedirs(processed_images_path + '/valid/', exist_ok=True)
    os.makedirs(processed_annotations_path + '/train/', exist_ok=True)
    os.makedirs(processed_annotations_path + '/valid/', exist_ok=True)

    # Function to save images and labels in YOLO format
    def segregate_data(df, img_path, label_path, train_img_path, train_label_path):
        filenames = set(df.filename)
        for filename in filenames:
            yolo_list = []
            for _, row in df[df.filename == filename].iterrows():
                yolo_list.append([row.labels, row.x_center_norm, row.y_center_norm, row.width_norm, row.height_norm])

            yolo_list = np.array(yolo_list)
            txt_filename = os.path.join(train_label_path, f"{row.prev_filename.split('.')[0]}.txt")
            np.savetxt(txt_filename, yolo_list, fmt=["%d", "%f", "%f", "%f", "%f"])
            shutil.copyfile(os.path.join(img_path, row.prev_filename), os.path.join(train_img_path, row.prev_filename))

    # Segregate data into training and validation sets
    segregate_data(df_train, raw_images_path, raw_annotations_path, processed_images_path + '/train/', processed_annotations_path + '/train/')
    segregate_data(df_valid, raw_images_path, raw_annotations_path, processed_images_path + '/valid/', processed_annotations_path + '/valid/')

if __name__ == "__main__":
    config = load_config()
    preprocess_data(config['raw_images_path'], config['raw_annotations_path'], config['processed_images_path'], config['processed_annotations_path'])
