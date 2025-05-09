import torch
import matplotlib.pyplot as plt
import config

# dictionary to store activations
#decleare it here to access it globally
feature_maps = {}

def get_activation(name):

    def hook(model, input, output):
        #used to minimize the GPU memory usage
        feature_maps[name] = output.detach().cpu()
    return hook
#takes the names of the layers to visualize if theree i no need for it (feature extractor) it is none
def visualize_features(model, sample_tensor, layers_to_visualize_names=None):

    try:
        #getting device from model
         device = next(model.parameters()).device
    except StopIteration:
         print("Warning: Model has no parameters, using device from config.")
         device = config.DEVICE

    if layers_to_visualize_names is None:
        layers_to_visualize_names = config.LAYERS_TO_VISUALIZE #falls back to config
        print(f"Using default layers from config for visualization: {layers_to_visualize_names}")

    num_filters_to_show = config.NUM_FILTERS_TO_SHOW_VIZ

    #checking if the model has the necessary attribute
    if not hasattr(model, 'conv_layers_for_viz') or not isinstance(model.conv_layers_for_viz, dict):
         print("Error: Model object does not have the required 'conv_layers_for_viz' dictionary attribute.")
         return

    print("\n--- Visualizing Features ---")
    #To ensure that the model is actually in evaluation mode
    model.eval()

    hooks = []
    #clearing the previious actions
    feature_maps.clear()
    layers_actually_hooked = []

    #registering hooks
    for layer_name in layers_to_visualize_names:
        # checks if the given dictionary contains layer_name
        if layer_name in model.conv_layers_for_viz:
            try:
                layer_obj = model.conv_layers_for_viz[layer_name]
                # checking if it's a module
                if layer_obj is not None and isinstance(layer_obj, torch.nn.Module):
                     hooks.append(layer_obj.register_forward_hook(get_activation(layer_name)))
                     layers_actually_hooked.append(layer_name)
                else:
                    print(f"Warning: Layer object for '{layer_name}' is None or not a Module. Skipping hook.")
            except Exception as e:
                 print(f"Warning: Could not register hook for layer '{layer_name}'. Error: {e}")
        else:
            print(f"Warning: Layer name '{layer_name}' not found in model.conv_layers_for_viz. Skipping.")


    if not hooks:
        print("No valid layers were hooked for visualization.")
        return

    # Run forward pass
    print(f"Running forward pass for visualization on device: {device}...")
    #diabling gradient calculation(temporarily)
    with torch.no_grad():
        # To ensure sample_tensor has batch dimension and is actuallly on the correct device
        if sample_tensor.ndim == 3: # add batch dimension if missing (C, H, W) -> (1, C, H, W)
            sample_tensor_batch = sample_tensor.unsqueeze(0)
        elif sample_tensor.ndim == 4: # Already has batch dimension
             sample_tensor_batch = sample_tensor
        else:
             print(f"Error: sample_tensor has unexpected dimensions: {sample_tensor.ndim}. Expected 3 or 4.")
             #removing hooks before returning
             for hook in hooks: hook.remove()
             return

        try:
             _ = model(sample_tensor_batch.to(device))
        except Exception as e:
            print(f"Error during model forward pass for visualization: {e}")
             #removing hooks before returning
            for hook in hooks: hook.remove()
            return


    #Here we remove hooks immediately after use
    for hook in hooks:
        hook.remove()
    print("Forward pass complete, hooks removed.")

    #Ploting
    # iterating over layers we hooked
    for layer_name in layers_actually_hooked:
        if layer_name in feature_maps:
            # removing batch dim (B, C, H, W) -> (C, H, W)
            maps = feature_maps[layer_name].squeeze(0)
            num_filters = maps.shape[0]
            map_h, map_w = maps.shape[1], maps.shape[2]
            print(f"Plotting {layer_name} (Shape: {num_filters}x{map_h}x{map_w}) - Showing {min(num_filters_to_show, num_filters)} filters")

            num_plot = min(num_filters_to_show, num_filters)
            if num_plot == 0: continue

            fig, axes = plt.subplots(1, num_plot, figsize=(num_plot * 2 + 1, 3))
            #adjusting title position
            fig.suptitle(f"Feature Maps from {layer_name}", fontsize=14, y=0.98)
            # to handle case where theres only one subplot
            if num_plot == 1:
                 axes = [axes]

            for i in range(num_plot):
                ax = axes[i]
                try:
                    #ensuring map is 2D for imshow
                    feature_map_to_plot = maps[i].numpy()
                    if feature_map_to_plot.ndim != 2:
                         print(f"  Warning: Feature map {i} for {layer_name} is not 2D (shape: {feature_map_to_plot.shape}). Skipping plot.")
                         ax.set_title(f'Filter {i+1}\n(Invalid Dim)')
                         ax.axis('off')
                         continue

                    im = ax.imshow(feature_map_to_plot, cmap='viridis')
                    ax.set_title(f'Filter {i+1}')
                    ax.axis('off')

                except Exception as plot_err:
                    print(f"  Error plotting filter {i+1} for {layer_name}: {plot_err}")
                    ax.set_title(f'Filter {i+1}\n(Plot Error)')
                    ax.axis('off')

            #using constrained_layout=True for potentially better spacing
            plt.subplots_adjust(top=0.85) # Adjust top spacing if suptitle overlaps
            # plt.tight_layout(rect=[0, 0.03, 1, 0.90]) # Alternative spacing adjustment
            plt.show(block=False) # Use block=False if running in interactive env and want script to continue
        else:
            print(f"Feature maps for layer '{layer_name}' not found in collected results.")