import torch
import os

#directory setting
#setting  relative filepath(according to config module)
#can't set absolute path in case of a mac usage
_this_dir = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(_this_dir, 'data', 'flowers')
NUM_CLASSES = 5 # Daisy, Dandelion, Rose, Sunflower, Tulip
RANDOM_SEED = 42 #a fixed value  for initializing model weights

# --- Device Configuration ---
#checks if the  gpu is avaliable
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")



IMG_SIZE = 224
#the number of items proessed in one iteration
BATCH_SIZE = 32
LEARNING_RATE = 0.001 # Base LR, can be overridden
#number of training epochs
EPOCHS = 15
#percantage of training (we use this to calculate validation rate)
TRAIN_SPLIT = 0.8
OPTIMIZER = 'SGD'


#to use VGG16 to fine tune I am going to freeze upto this layer
FREEZE_UPTO_LAYER_INDEX = 5


# list of layers to visualize
LAYERS_TO_VISUALIZE = ['conv1_1', 'conv3_1', 'conv5_1']
#amount of feature maps to how per model layer
NUM_FILTERS_TO_SHOW_VIZ = 8

# defining the paths for  saving
SAVE_MODEL_PATH_CUSTOM = os.path.join(_this_dir, 'flower_cnn_model.pth')
LOAD_MODEL_PATH_CUSTOM = None
SAVE_MODEL_PATH_VGG16_FE = os.path.join(_this_dir, 'vgg16_feature_extractor.pth')
LOAD_MODEL_PATH_VGG16_FE = None
SAVE_MODEL_PATH_VGG16_FT = os.path.join(_this_dir, 'vgg16_finetune.pth')
LOAD_MODEL_PATH_VGG16_FT = None