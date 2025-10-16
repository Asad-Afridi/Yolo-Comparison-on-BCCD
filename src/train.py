import argparse
import os
from ultralytics import YOLO

# Helper function to handle results folder creation
def create_results_folder(model_name, batch_size, epochs, learning_rate):
    results_dir = f'./results/{model_name}'
    os.makedirs(results_dir, exist_ok=True)
    return results_dir

# Main training function
def train_model(model_name, batch_size, epochs, learning_rate, img_size=640, pretrained=True):
    # Check if the model is valid
    
        
    try:
        # Initialize the YOLO model from ultralytics
        model = YOLO(model_name)  # Automatically loads the model based on the name
    except:
        raise ValueError(f"Model '{model_name}' not available.")

    # Create results folder
    results_folder = create_results_folder(model_name, batch_size, epochs, learning_rate)

    # Start training
    model.train(
        data='dataset\data.yaml',  # Ensure to have the proper dataset config file
        epochs=epochs,
        pretrained=pretrained,
        project=results_folder,
        name=model_name,
        exist_ok=True  # Overwrite results folder if it exists
    )

    print(f"Training complete! Results saved to {results_folder}")

# Parse command line arguments
def main():
    parser = argparse.ArgumentParser(description="Train a YOLO model (v3, v4, or v5) with custom hyperparameters.")
    parser.add_argument('--model', type=str, required=True, help="YOLO model to train: yolov3, yolov4, yolov5")
    parser.add_argument('--batch_size', type=int, default=16, help="Batch size for training")
    parser.add_argument('--epochs', type=int, default=50, help="Number of epochs for training")
    parser.add_argument('--learning_rate', type=float, default=0.001, help="Learning rate for training")
    parser.add_argument('--img_size', type=int, default=640, help="Image size for training")
    parser.add_argument('--pretrained', type=bool, default=True, help="Use pretrained weights (True/False)")

    args = parser.parse_args()

    # Train the model
    try:
        train_model(args.model, args.batch_size, args.epochs, args.learning_rate, args.img_size, args.pretrained)
    except ValueError as e:
        print(e)

if __name__ == "__main__":
    main()
