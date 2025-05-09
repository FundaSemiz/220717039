import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.optim as optim
import os
import time # for timing


import config
import data_loader
import model as model_def
import train
import evaluate
import visualize
from data_loader import get_sample_for_viz

def main_custom_cnn_experiment():

    print(f"--- Starting Custom CNN Experiment ---")
    print(f"Using Device: {config.DEVICE}")
    print(f"Using Seed: {config.RANDOM_SEED}")

    #I am going to be using a different image size for cnn(not 224)
    #spesific config
    current_img_size = 128
    current_epochs = 15
    current_lr = 0.001
    current_optimizer_type = 'adam'#decided to use adam instead of SGD
    weight_decay_value = 1e-4 # to apply L2 regularization
    model_save_path = config.SAVE_MODEL_PATH_CUSTOM
    model_load_path = config.LOAD_MODEL_PATH_CUSTOM


    print(f"Image Size: {current_img_size}x{current_img_size}")

    #setting a random seed
    torch.manual_seed(config.RANDOM_SEED)
    #Checks if theres a gpu
    if config.DEVICE == 'cuda':
        torch.cuda.manual_seed_all(config.RANDOM_SEED)

    #Loading the data
    train_loader, val_loader, class_names, full_dataset = data_loader.load_data(img_size=current_img_size)
    if train_loader is None:
        print("Failed to load data. Exiting.")
        return

    # initializeing the model
    model = model_def.SimpleCNN(num_classes=config.NUM_CLASSES, img_size=current_img_size).to(config.DEVICE)
    print("\nModel Architecture:")
    # print(model) # Can be very long

    #Loading the weights if the path exists
    if model_load_path and os.path.exists(model_load_path):
        try:
            print(f"Loading model weights from: {model_load_path}")
            model.load_state_dict(torch.load(model_load_path, map_location=config.DEVICE))
            print("Model weights loaded successfully.")
        except Exception as e:
            print(f"Error loading model weights: {e}. Training from scratch.")
    # -----------------------------------------

    #defining loss and optimizer
    criterion = nn.CrossEntropyLoss()
    params_to_train = model.parameters()
    #checks the optimizer type
    if current_optimizer_type.lower() == 'adam':
        optimizer = optim.Adam(params_to_train, lr=current_lr, weight_decay=weight_decay_value)
    elif current_optimizer_type.lower() == 'sgd':
        optimizer = optim.SGD(params_to_train, lr=current_lr, momentum=0.9, weight_decay=weight_decay_value)
    else: # Default is adam
        optimizer = optim.Adam(params_to_train, lr=current_lr, weight_decay=weight_decay_value)

    print(f"Using Optimizer: {current_optimizer_type}, LR: {current_lr}, Weight Decay: {weight_decay_value}")



    print("\n--- Training ---")
    train_losses = []
    train_accuracies = []
    start_time = time.time()
    for epoch in range(current_epochs):
        print(f"Starting Epoch {epoch+1}/{current_epochs}")
        epoch_loss, epoch_acc = train.train_model(model, train_loader, criterion, optimizer)
        print(f"Epoch [{epoch+1}/{current_epochs}] Completed - Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}")
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc)
    end_time = time.time()
    print(f"--- Finished Training (Total Time: {end_time - start_time:.2f}s) ---")

    #evaluation of the cnn model
    evaluate.evaluate_model(model, val_loader, criterion, class_names)

    #visualizing features for Custom CNN
    if hasattr(model, 'conv_layers_for_viz') and model.conv_layers_for_viz:
        raw_sample_image, sample_label_idx = get_sample_for_viz(full_dataset)
        if raw_sample_image is not None:
            _, current_val_transform = data_loader.get_transforms(current_img_size)
            sample_tensor = current_val_transform(raw_sample_image)
            sample_label_name = class_names[sample_label_idx]
            print(f"\nVisualizing features for a sample image (Label: {sample_label_name}) for Custom CNN")

            #wanted layers for visualization
            custom_cnn_viz_layers = ['conv1', 'conv3', 'conv5']
            visualize.visualize_features(model, sample_tensor, layers_to_visualize_names=custom_cnn_viz_layers)
        else:
            print("Could not get sample image for visualization.")
    else:
         print("Skipping visualization: Custom CNN model does not have 'conv_layers_for_viz' attribute.")


    #Saving the model if wanted
    if model_save_path:
       try:
           print(f"\nSaving final model to {model_save_path}")
           torch.save(model.state_dict(), model_save_path)
           print("Model saved successfully.")
       except Exception as e:
           print(f"Error saving model: {e}")
    # ------------------------------------

    #calling necessay functions for the visualization
    if train_losses and train_accuracies:
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(range(1, len(train_losses) + 1), train_losses, marker='o', label='Training Loss')
        plt.title('Custom CNN Training Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.plot(range(1, len(train_accuracies) + 1), train_accuracies, marker='o', label='Training Accuracy')
        plt.title('Custom CNN Training Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show()

    print("\n--- Custom CNN Experiment Complete ---")

if __name__ == "__main__":
    main_custom_cnn_experiment()