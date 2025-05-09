# data_loader.py
import torch
import torchvision.transforms as transforms #to transform the images
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
import os
import copy
import config
#Functions to transform and change images
def get_transforms(img_size):
    """Returns train and validation transforms for a given image size."""
    # Define transformations based on config
    #for train config use this
    train_transform = transforms.Compose([
        #resizes the images according to given size
        transforms.Resize((img_size, img_size)),
        #these methods changes the image (i only did it for training)
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        #transforms the images to PyTorch tensors
        transforms.ToTensor(),
        #normalizes tensor images
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    #for validation config use this
    val_transform = transforms.Compose([
        #resizes the image to a square using the given size
        transforms.Resize((img_size, img_size)),
        # transforms the images to PyTorch tensors
        transforms.ToTensor(),
        # normalizes tensor images
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return train_transform, val_transform

#function for loading the data
def load_data(img_size=None, batch_size=None, train_split=None, data_path=None):
    """Loads the dataset, applies transforms, splits, and creates DataLoaders."""
    # if the arguments are ot passed they will be taken as None
    #learn more about these ig is not else find other simple ways if possible
    current_img_size = img_size if img_size is not None else config.IMG_SIZE
    current_batch_size = batch_size if batch_size is not None else config.BATCH_SIZE
    current_train_split = train_split if train_split is not None else config.TRAIN_SPLIT
    current_data_path = data_path if data_path is not None else config.DATASET_PATH

    train_transform, val_transform = get_transforms(current_img_size)
     #check if the path exists
    if not os.path.exists(current_data_path):
        print(f"Error: Dataset path '{current_data_path}' not found.")
        print("Please download the flower dataset (5 classes) and organize it.")
        return None, None, None, None

    #we know that data pah exist so
    # Loading dataset while passing the data path
    full_dataset = ImageFolder(root=current_data_path)
    print(f"Found dataset with {len(full_dataset)} images in {len(full_dataset.classes)} classes.")
    print("Classes:", full_dataset.classes)

    #if the number of classes and class names lenght doen't match
    if len(full_dataset.classes) != config.NUM_CLASSES:
         print(f"Warning: Expected {config.NUM_CLASSES} classes (from config.py), "
               f"but found {len(full_dataset.classes)} in dataset folder.")
    #if it matches
    class_names = full_dataset.classes

    #length of the dataset
    total_len = len(full_dataset)
    #turns it into an int from float
    train_len = int(current_train_split * total_len)
    #lftovers are validation
    val_len = total_len - train_len

    # Splitting dataset accorrding to the lengths
    train_dataset_split, val_dataset_split = random_split(full_dataset, [train_len, val_len],
                                                          generator=torch.Generator().manual_seed(config.RANDOM_SEED))

    # Apply transforms correctly after splitting
    #applying transforms to the training dataset
    train_dataset = copy.deepcopy(train_dataset_split)
    train_dataset.dataset.transform = train_transform # Apply train transforms to the underlying dataset
    train_dataset.indices = train_dataset_split.indices # Keep original split indices
    # applying transforms to the validation dataset
    val_dataset = copy.deepcopy(val_dataset_split)
    val_dataset.dataset.transform = val_transform # Apply val transforms
    val_dataset.indices = val_dataset_split.indices

    # creating dataloaders(using cuda for gpu throughout the code )
    train_loader = DataLoader(train_dataset, batch_size=current_batch_size, shuffle=True, num_workers=2, pin_memory=True if config.DEVICE=='cuda' else False)
    val_loader = DataLoader(val_dataset, batch_size=current_batch_size, shuffle=False, num_workers=2, pin_memory=True if config.DEVICE=='cuda' else False)

    print(f"Data loaded: {train_len} training samples, {val_len} validation samples (Img size: {current_img_size}x{current_img_size}).")
    return train_loader, val_loader, class_names, full_dataset # Return original dataset for sampling
#to retrive a raw image
def get_sample_for_viz(full_dataset):
    """Gets a single raw image and its label index from the original dataset."""
    if full_dataset and len(full_dataset) > 0:
        # gets the first image/label tuple
        return full_dataset[0]
    return None, None