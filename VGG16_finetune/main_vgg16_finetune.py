import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os
import time #for timing

#importing the modules we decleared
import config
import data_loader
#fine tuning model
import model_vgg16_ft
import train
import evaluate
#for visualizing the layers wanted
import visualize
from data_loader import get_sample_for_viz

def main_vgg16_ft_experiment():

    print(f"--- Starting VGG16 Fine-Tuning Experiment ---")
    print(f"Using Device: {config.DEVICE}")
    print(f"Using Seed: {config.RANDOM_SEED}")

    #we now set the image size 224(unlike custtom cnnn 128)
    current_img_size = 224
    current_epochs = 20
    current_lr = 0.0001
    current_optimizer_type = 'adam'#I will use adam instead of SGD because of the epoch amount
    weight_decay_value = 1e-4 #for L2 regularization
    freeze_index = config.FREEZE_UPTO_LAYER_INDEX
    model_save_path = config.SAVE_MODEL_PATH_VGG16_FT
    model_load_path = config.LOAD_MODEL_PATH_VGG16_FT


    print(f"Image Size: {current_img_size}x{current_img_size}")
    print(f"Freezing layers up to index: {freeze_index}")

    # setting a random seed
    torch.manual_seed(config.RANDOM_SEED)
    # Checks if there is a gpu
    if config.DEVICE == 'cuda':
        torch.cuda.manual_seed_all(config.RANDOM_SEED)

    # loading the data
    train_loader, val_loader, class_names, full_dataset = data_loader.load_data(img_size=current_img_size)
    if train_loader is None:
        print("Failed to load data. Exiting.")
        return

    # initializing the model
    model = model_vgg16_ft.build_vgg16_finetune(
        num_classes=config.NUM_CLASSES,
        pretrained=True,
        freeze_upto_layer=freeze_index
    )
    if model is None:
        print("Failed to build model. Exiting.")
        return
    model = model.to(config.DEVICE)

    # Loading the weights if the path exists
    if model_load_path and os.path.exists(model_load_path):
        try:
            print(f"Loading model state dict from: {model_load_path}")
            model.load_state_dict(torch.load(model_load_path, map_location=config.DEVICE))
            print("Model state dict loaded successfully.")
        except Exception as e:
            print(f"Error loading model state dict: {e}. Training model.")
    # -----------------------------------------

    # defining loss and optimizer
    params_to_train = filter(lambda p: p.requires_grad, model.parameters())
    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)


    criterion = nn.CrossEntropyLoss()
    # checks the optimizer type
    if current_optimizer_type.lower() == 'adam':
        optimizer = optim.Adam(params_to_train, lr=current_lr, weight_decay=weight_decay_value)
    elif current_optimizer_type.lower() == 'sgd':
        optimizer = optim.SGD(params_to_train, lr=current_lr, momentum=0.9, weight_decay=weight_decay_value)
    else: # Default is adam
        optimizer = optim.Adam(params_to_train, lr=current_lr, weight_decay=weight_decay_value)

    print(f"Optimizer will train {num_trainable_params} parameters.")
    print(f"Using Optimizer: {current_optimizer_type}, LR: {current_lr}, Weight Decay: {weight_decay_value}")


    print("\n--- Fine-Tuning Model ---")
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
    print(f"--- Finished Fine-Tuning (Total Time: {end_time - start_time:.2f}s) ---")

    #evaluation of the VGG16 fine tune model
    evaluate.evaluate_model(model, val_loader, criterion, class_names)

    # visualizing features for VGG16 finetune
    if hasattr(model, 'conv_layers_for_viz') and model.conv_layers_for_viz:
        raw_sample_image, sample_label_idx = get_sample_for_viz(full_dataset)
        if raw_sample_image is not None:
            # Need the correct validation transform for the image size used
            _, current_val_transform = data_loader.get_transforms(current_img_size)
            sample_tensor = current_val_transform(raw_sample_image)
            sample_label_name = class_names[sample_label_idx]
            print(f"\nVisualizing features for a sample image (Label: {sample_label_name}) for Fine-Tuned VGG16")

            # <<< Explicitly pass the layer names for VGG16 (or use config) >>>
            vgg16_viz_layers = config.LAYERS_TO_VISUALIZE # Assumes config has VGG names
            # vgg16_viz_layers = ['conv1_1', 'conv3_1', 'conv5_1'] # Or be explicit
            visualize.visualize_features(model, sample_tensor, layers_to_visualize_names=vgg16_viz_layers)
        else:
            print("Could not get sample image for visualization.")
    else:
        print("Skipping visualization: Fine-Tuned VGG16 model does not have 'conv_layers_for_viz' attribute properly set up.")


    #Saving the model if wanted
    if model_save_path:
       try:
           print(f"\nSaving final model to {model_save_path}")
           torch.save(model.state_dict(), model_save_path)
           print("Model saved successfully.")
       except Exception as e:
           print(f"Error saving model: {e}")
    # ---------------------------------------------

    # calling necessay functions for the visualization
    if train_losses and train_accuracies:
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(range(1, len(train_losses) + 1), train_losses, marker='o', label='Training Loss')
        plt.title('VGG16 Fine-Tune Training Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.plot(range(1, len(train_accuracies) + 1), train_accuracies, marker='o', label='Training Accuracy')
        plt.title('VGG16 Fine-Tune Training Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show()

    print("\n--- VGG16 Fine-Tuning Experiment Complete ---")

if __name__ == "__main__":
    main_vgg16_ft_experiment()