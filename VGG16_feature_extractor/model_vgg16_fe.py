import torch
import torch.nn as nn
import torchvision.models as models
import config

def build_vgg16_feature_extractor(num_classes=config.NUM_CLASSES, pretrained=True):

    print(f"Building VGG16 model (Feature Extractor - Pretrained: {pretrained})...")

    #loading the VGG16 with the weights then creating an instance of the VGG16 model architecture
    weights = models.VGG16_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.vgg16(weights=weights)

    #freezing the  layers in the features part
    print("Freezing convolutional feature layers...")
    #if the model has attribute features
    if hasattr(model, 'features'):
        #iterating thorough weights and biases
        for param in model.features.parameters():
            #setting them False to freeze them
            param.requires_grad = False
        print("Feature layers frozen.")
    else:
        print("Warning: model.features not found. Cannot freeze layers.")


    #getting the number of input features for the classifiers last layer
    try:
        #checking the classifiers existance and if its sequential also if it has layers
        if hasattr(model, 'classifier') and isinstance(model.classifier, nn.Sequential) and len(model.classifier) > 0:
             #initializing a variable to  store the index.
             last_linear_layer_index = -1
             for i in range(len(model.classifier) - 1, -1, -1):
                 #checks if the current layer is a Linear in case it is loop will not continiue
                 if isinstance(model.classifier[i], nn.Linear):
                     last_linear_layer_index = i
                     break
            #if a linear layer exists
             if last_linear_layer_index != -1:
                 #getting the number of features of original last layer(generally is 4096)
                num_ftrs = model.classifier[last_linear_layer_index].in_features
                print(f"Original classifier's last Linear layer input features: {num_ftrs}")

                # replacing the last linear layer with a new one for our number of classes
                model.classifier[last_linear_layer_index] = nn.Linear(num_ftrs, num_classes)
                print(f"Replaced final classifier layer for {num_classes} classes.")
             else:
                 print("Error: Could not find a Linear layer in VGG16 classifier.")
                 return None
        else:
            print("Error: VGG16 model structure not as expected (missing or invalid classifier).")
            return None

    except Exception as e:
        print(f"Error modifying VGG16 classifier: {e}")
        return None

    return model