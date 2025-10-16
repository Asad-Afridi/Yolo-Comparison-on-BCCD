from ultralytics import YOLO
import cv2
import os

def make_predictions(model_name, img_path, results_folder):
    # Load the trained model
    model = YOLO(f'{results_folder}/weights/best.pt')

    # Read the input image
    img = cv2.imread(img_path)

    # Make prediction
    results = model(img)

    # Save predictions to the results folder
    results.save(os.path.join(results_folder, 'predictions/'))
    print(f"Predictions saved to {results_folder}/predictions/")

if __name__ == "__main__":
    model_name = "yolov5"  # Use your trained model here
    img_path = "./test_image.jpg"  # Path to your test image
    results_folder = "./results/yolov5/batch_16_epochs_50_lr_0.001"  # Specify your result folder path
    make_predictions(model_name, img_path, results_folder)
