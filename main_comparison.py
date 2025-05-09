import torch
import torch.nn as nn
import torch.optim as optim
import time
import pandas as pd

import config
import data_loader
from Custom_CNN import model as model_custom
from VGG16_feature_extractor import model_vgg16_fe
from VGG16_finetune import model_vgg16_ft
import train
import evaluate


def run_experiment(model_name, build_model_func, img_size, epochs, learning_rate,
                   optimizer_type, weight_decay_val=0.0, # New parameter for L2
                   freeze_index=None):


    print(f"\n{'='*20} Running Experiment: {model_name} {'='*20}")
    results = None


    original_img_size = config.IMG_SIZE
    original_lr = config.LEARNING_RATE
    original_epochs = config.EPOCHS
    original_optimizer = config.OPTIMIZER
    original_freeze_index = config.FREEZE_UPTO_LAYER_INDEX

    try:

        current_img_size = img_size
        current_learning_rate = learning_rate
        current_epochs = epochs
        current_optimizer_type = optimizer_type
        current_freeze_index = freeze_index if freeze_index is not None else config.FREEZE_UPTO_LAYER_INDEX # Use passed or config

        print(f"Settings - Img Size: {current_img_size}, LR: {current_learning_rate}, Epochs: {current_epochs}, Optimizer: {current_optimizer_type}, Weight Decay: {weight_decay_val}")
        if current_freeze_index is not None and model_name == "VGG16 Fine-Tune":
             print(f"Settings - Freeze Index: {current_freeze_index}")


        torch.manual_seed(config.RANDOM_SEED)
        if config.DEVICE == 'cuda':
            torch.cuda.manual_seed_all(config.RANDOM_SEED)

        # Load Data using the specific image size for this run
        train_loader, val_loader, class_names, _ = data_loader.load_data(img_size=current_img_size)
        if train_loader is None:
            print(f"Data loading failed for {model_name}. Skipping.")
            return None

        #  Build Model
        print("Building model...")
        if model_name == "Custom CNN":
            model = build_model_func(num_classes=config.NUM_CLASSES, img_size=current_img_size)
        elif model_name == "VGG16 Fine-Tune":
             model = build_model_func(num_classes=config.NUM_CLASSES, pretrained=True, freeze_upto_layer=current_freeze_index)
        elif model_name == "VGG16 Feature Extractor":
             model = build_model_func(num_classes=config.NUM_CLASSES, pretrained=True)
        else:
            print(f"Unknown model_name: {model_name}. Skipping.")
            return None

        if model is None:
            print(f"Model building failed for {model_name}. Skipping.")
            return None
        model = model.to(config.DEVICE)

        # define Loss and Optimizer
        criterion = nn.CrossEntropyLoss()
        params_to_train = filter(lambda p: p.requires_grad, model.parameters())
        num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Optimizer will train {num_trainable_params} parameters.")

        if current_optimizer_type.lower() == 'adam':
            optimizer = optim.Adam(params_to_train,
                                   lr=current_learning_rate,
                                   weight_decay=weight_decay_val) # L2 Regularization
        elif current_optimizer_type.lower() == 'sgd':
            optimizer = optim.SGD(params_to_train,
                                  lr=current_learning_rate,
                                  momentum=0.9,
                                  weight_decay=weight_decay_val) # L2 Regularization
        else:
            print(f"Warning: Unknown optimizer '{current_optimizer_type}'. Defaulting to Adam with weight decay.")
            optimizer = optim.Adam(params_to_train,
                                   lr=current_learning_rate,
                                   weight_decay=weight_decay_val)

        #  Train and Measure
        print(f"\n--- Training {model_name} for {current_epochs} epochs ---")
        start_time = time.time()
        for epoch in range(current_epochs):
            epoch_loss, epoch_acc = train.train_model(model, train_loader, criterion, optimizer)
            print(f"Epoch [{epoch+1}/{current_epochs}] - Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}")
        end_time = time.time()
        training_duration_seconds = end_time - start_time
        print(f"--- Finished Training {model_name} ---")
        print(f"Training Time: {training_duration_seconds:.2f} seconds")

        #  Evaluate
        accuracy, report_dict = evaluate.evaluate_model(model, val_loader, criterion, class_names)

        precision = report_dict.get('weighted avg', {}).get('precision', 0.0)
        recall = report_dict.get('weighted avg', {}).get('recall', 0.0)
        f1_score = report_dict.get('weighted avg', {}).get('f1-score', 0.0)

        results = {
            "Model": model_name,
            "Accuracy": accuracy,
            "Precision (Weighted)": precision,
            "Recall (Weighted)": recall,
            "F1-score (Weighted)": f1_score,
            "Training Time (s)": training_duration_seconds
        }

    except Exception as e:
        print(f"!!!!!!!!!! Error during experiment {model_name}: {e} !!!!!!!!!!")
        import traceback
        traceback.print_exc() # Print full traceback for debugging
        results = None



    return results



if __name__ == "__main__":
    all_results_list = []
    default_weight_decay = 1e-4

    print(f"\nStarting Performance Comparison on device: {config.DEVICE}")

    # --- Run Experiment 1: Custom CNN ---
    custom_cnn_results = run_experiment(
        model_name="Custom CNN",
        build_model_func=model_custom.SimpleCNN,
        img_size=128,
        epochs=15,
        learning_rate=0.001,
        optimizer_type='Adam',
        weight_decay_val=default_weight_decay
    )
    if custom_cnn_results:
        all_results_list.append(custom_cnn_results)


    vgg16_fe_results = run_experiment(
        model_name="VGG16 Feature Extractor",
        build_model_func=model_vgg16_fe.build_vgg16_feature_extractor,
        img_size=224,
        epochs=10,
        learning_rate=0.001,
        optimizer_type='Adam',
        weight_decay_val=default_weight_decay
    )
    if vgg16_fe_results:
        all_results_list.append(vgg16_fe_results)


    vgg16_ft_results = run_experiment(
        model_name="VGG16 Fine-Tune",
        build_model_func=model_vgg16_ft.build_vgg16_finetune,
        img_size=224,
        epochs=20,
        learning_rate=0.0001,
        optimizer_type='Adam',
        weight_decay_val=default_weight_decay, # Apply L2 here as well
        freeze_index=config.FREEZE_UPTO_LAYER_INDEX # Get from config or specify
    )
    if vgg16_ft_results:
        all_results_list.append(vgg16_ft_results)



    if all_results_list:
        print("\n\n--- Performance Comparison Summary ---")
        df = pd.DataFrame(all_results_list)

        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 1000)
        pd.set_option('display.colheader_justify', 'center')
        pd.set_option('display.precision', 4)

        columns_ordered = [
            "Model",
            "Accuracy",
            "Precision (Weighted)",
            "Recall (Weighted)",
            "F1-score (Weighted)",
            "Training Time (s)"
        ]

        df_display_cols = [col for col in columns_ordered if col in df.columns]
        df_display = df[df_display_cols]


        print(df_display.to_string(index=False))
    else:
        print("\nNo experiment results collected.")

    print("\n--- Comparison Complete ---")