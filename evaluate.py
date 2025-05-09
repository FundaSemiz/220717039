import torch
from sklearn.metrics import (classification_report, accuracy_score)
import config
import time #to calculate the amount of time spent

#takes the model type,criterion(to calculate average loss) etc.
def evaluate_model(model, val_loader, criterion, class_names):

    device = config.DEVICE
    #to set the model to evaluation
    model.eval()
    all_labels = []
    all_predictions = []
    running_loss = 0.0
    total_samples = 0
    #to keep track of the time
    start_time = time.time()

    # print(f"Evaluating on {len(val_loader.dataset)} samples using device: {device}") # Length might be inaccurate

    #we don't need gradients so I will disable them cuz we are evaluating not training
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            #gets the raw scores of the batch
            outputs = model(inputs)
            #loss calculation
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)

            #for predicted class for the image
            _, predicted = torch.max(outputs.data, 1)

            #move results to cpu ,turns them to numpy arrays
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
            total_samples += labels.size(0)
    # calculting the amount of time passed
    eval_duration = time.time() - start_time
    print(f"Evaluation completed in {eval_duration:.2f}s")

    # in case val_loader is empty
    if total_samples == 0:
        print("Warning: No samples found in validation loader. Cannot calculate metrics.")
        return 0.0, {}
    #the average validationloss and accuracy
    val_loss = running_loss / total_samples
    accuracy = accuracy_score(all_labels, all_predictions)


    report_text = classification_report(all_labels, all_predictions,
                                        target_names=class_names, zero_division=0)

    # creating a dictionary to keep track of impportant data
    report_dict = classification_report(all_labels, all_predictions,
                                        target_names=class_names, zero_division=0, output_dict=True)

    print(f"\nValidation Loss: {val_loss:.4f}")
    print(f"Validation Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(report_text) # Print the text report

    return accuracy, report_dict