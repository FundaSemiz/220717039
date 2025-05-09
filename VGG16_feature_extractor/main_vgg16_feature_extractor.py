import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os

#importing the modules we need
import config
import data_loader
import model_vgg16_fe # Import the feature extractor model builder
import train
import evaluate
#I will not be importing visualize since it wasn't required

def main_vgg16_fe_experiment():

    print(f"--- Starting VGG16 Feature Extractor Experiment ---")
    print(f"Using Device: {config.DEVICE}")
    print(f"Using Seed: {config.RANDOM_SEED}")

    #using 224 pixel as image size unlike custom cnn
    current_img_size = 224
    current_epochs = 10
    current_lr = 0.001
    current_optimizer = 'adam'#I will use adam because of the epoch amount
    model_save_path = config.SAVE_MODEL_PATH_VGG16_FE
    model_load_path = config.LOAD_MODEL_PATH_VGG16_FE

    print(f"Image Size: {current_img_size}x{current_img_size}")

    #setting a random seed
    torch.manual_seed(config.RANDOM_SEED)
    #Checks if theres a gpu
    if config.DEVICE == 'cuda':
        torch.cuda.manual_seed_all(config.RANDOM_SEED)



    #Loading the data
    train_loader, val_loader, class_names, _ = data_loader.load_data(img_size=current_img_size)
    if train_loader is None:
        print("Failed to load data. Exiting.")
        return

    # initializing the model
    model = model_vgg16_fe.build_vgg16_feature_extractor(
        num_classes=config.NUM_CLASSES,
        pretrained=True
    )
    if model is None:
        print("Failed to build model. Exiting.")
        return
    model = model.to(config.DEVICE)


    #Loading the weights if the path exists
    if model_load_path and os.path.exists(model_load_path):
        try:
            print(f"Loading model state dict from: {model_load_path}")
            model.load_state_dict(torch.load(model_load_path, map_location=config.DEVICE))
            print("Model state dict loaded successfully.")
        except Exception as e:
            print(f"Error loading model state dict: {e}. Training classifier from scratch.")
    # --------------------------------------------------------------------


    #defining loss and optimizer
    params_to_train = filter(lambda p: p.requires_grad, model.parameters())
    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    #checks the optimizer type
    criterion = nn.CrossEntropyLoss()
    if current_optimizer.lower() == 'adam':
        optimizer = optim.Adam(params_to_train, lr=current_lr)
    elif current_optimizer.lower() == 'sgd':
        optimizer = optim.SGD(params_to_train, lr=current_lr, momentum=0.9)
    else: # Default is adam
        optimizer = optim.Adam(params_to_train, lr=current_lr)

    print(f"Optimizer will train {num_trainable_params} parameters.")
    print(f"Using Optimizer: {current_optimizer}, LR: {current_lr}")


    print("\n--- Training Only the Classifier ---")
    train_losses = []
    train_accuracies = []
    for epoch in range(current_epochs):
        epoch_loss, epoch_acc = train.train_model(model, train_loader, criterion, optimizer)
        print(f"Epoch [{epoch+1}/{current_epochs}] - Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}")
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc)

    print("--- Finished Training Classifier ---")

    #evaluation of the VGG16 feature extractor
    evaluate.evaluate_model(model, val_loader, criterion, class_names)

    #Saving the model if wanted
    if model_save_path:
       try:
           print(f"\nSaving final model to {model_save_path}")
           torch.save(model.state_dict(), model_save_path)
           print("Model saved successfully.")
       except Exception as e:
           print(f"Error saving model: {e}")
    # -------------------------------------------------------------------------------

    ##calling necessay functions for the visualization
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, current_epochs + 1), train_losses, marker='o', label='Training Loss')
    plt.title('VGG16 FE Classifier Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(range(1, current_epochs + 1), train_accuracies, marker='o', label='Training Accuracy')
    plt.title('VGG16 FE Classifier Training Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    print("\n--- VGG16 Feature Extractor Experiment Complete ---")

if __name__ == "__main__":
    main_vgg16_fe_experiment()