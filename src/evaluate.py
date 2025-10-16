from ultralytics import YOLO

def evaluate_model(model_name, results_folder):
    # Load the trained model
    model = YOLO(f'{results_folder}/weights/best.pt')

    # Evaluate on validation data
    results = model.val()

    # Save the metrics to the results folder
    with open(f'{results_folder}/metrics/evaluation_metrics.txt', 'w') as f:
        f.write(str(results))
    print(f"Evaluation complete! Metrics saved to {results_folder}/metrics/evaluation_metrics.txt")

if __name__ == "__main__":
    model_name = "yolov5"  # Use your trained model here
    results_folder = "./results/yolov5/batch_16_epochs_50_lr_0.001"  # Specify your result folder path
    evaluate_model(model_name, results_folder)
