import torch
import torch.nn as nn
import torchvision.models as models
import config

#FREEZE_UPTO_LAYER_INDEX is 5 from config
def build_vgg16_finetune(num_classes=config.NUM_CLASSES, pretrained=True, freeze_upto_layer=config.FREEZE_UPTO_LAYER_INDEX):

    print(f"Building VGG16 model for Fine-Tuning (Pretrained: {pretrained})...")

    weights = models.VGG16_Weights.IMAGENET1K_V1 if pretrained else None

    #loading and initializing the VGG16 structure
    model = models.vgg16(weights=None)
    if pretrained and weights:
        print("Loading pretrained ImageNet weights...")

        model.load_state_dict(weights.get_state_dict(progress=True))

    #freezing the layers
    if hasattr(model, 'features'):
        print(f"Freezing feature layer parameters up to index {freeze_upto_layer}...")
        params_frozen_count = 0
        total_params_checked = 0

        # iterating through layers
        for i, layer in enumerate(model.features.children()):
             is_frozen = i < freeze_upto_layer
             #iterating through parameters
             for param in layer.parameters():
                 #setting this way in case to prevent unwanted parameters from updating
                 param.requires_grad = not is_frozen
                 total_params_checked += param.numel() # Counts parameters
                 if is_frozen:
                      params_frozen_count += param.numel()

        print(f"Checked {total_params_checked} parameters in 'features'. Froze {params_frozen_count} parameters (layers < index {freeze_upto_layer}).")

    else:
        print("Warning: model.features not found. Cannot freeze layers.")

    # changing the classifier
    try:
        if hasattr(model, 'classifier') and isinstance(model.classifier, nn.Sequential) and len(model.classifier) > 0:
             last_linear_layer_index = -1
             #iterating to find the last linear layer in classifier
             for i in range(len(model.classifier) - 1, -1, -1):
                 if isinstance(model.classifier[i], nn.Linear):
                     last_linear_layer_index = i
                     break
            #if the last linear layer is found
             if last_linear_layer_index != -1:
                num_ftrs = model.classifier[last_linear_layer_index].in_features
                print(f"Original classifier's last Linear layer input features: {num_ftrs}")
                #replacing the last layer
                model.classifier[last_linear_layer_index] = nn.Linear(num_ftrs, num_classes)
                print(f"Replaced final classifier layer for {num_classes} classes.")
                #new classifier layer parameters automatically has requires_grad=True
             else:
                 print("Error: Could not find a Linear layer in VGG16 classifier.")
                 return None
        else:
            print("Error: VGG16 model structure not as expected (missing or invalid classifier).")
            return None
    except Exception as e:
        print(f"Error modifying VGG16 classifier: {e}")
        return None

    #Visualization
    #mapping the layer names to feature layers
    model.conv_layers_for_viz = {}
    if hasattr(model, 'features'):
        try:
            # 1st Conv Layer (conv1_1) at index 0
            if len(model.features) > 0 and isinstance(model.features[0], nn.Conv2d):
                 model.conv_layers_for_viz['conv1_1'] = model.features[0]
            # Middle Conv Layer (conv3_1, start of 3rd block) at index 10
            if len(model.features) > 10 and isinstance(model.features[10], nn.Conv2d):
                 model.conv_layers_for_viz['conv3_1'] = model.features[10]
            # Deepest Conv Layer (conv5_1, start of 5th block) at index 24
            if len(model.features) > 24 and isinstance(model.features[24], nn.Conv2d):
                 model.conv_layers_for_viz['conv5_1'] = model.features[24]
            print(f"Set up layers for visualization: {list(model.conv_layers_for_viz.keys())}")
        except IndexError:
            print("Warning: Could not access all expected VGG16 layers for visualization hook setup.")
        except Exception as e:
            print(f"Warning: Error setting up visualization layers: {e}")
    else:
         print("Warning: model.features not found, cannot set up visualization layers.")


    return model