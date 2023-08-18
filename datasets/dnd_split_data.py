import random
import yaml
import pickle as pkl
from torch.utils.data import Dataset

class DND(Dataset):
    def __init__(self, split_ratio=0.8):
        self.split_data(split_ratio)

    def split_data(self, split_ratio):
        # Load the YAML file and assign Python variables to the image paths and label names
        with open('../../DND-Diko-WWWR/WR2021/labels_trainval.yml', 'r') as file:
            labels = yaml.load(file, Loader=yaml.FullLoader)
        
        # Create empty lists for the training and validation data
        train_images = []
        train_labels = []
        val_images = []
        val_labels = []
        
        # Randomly divide each (image_path, label_name) pair to one of two lists (“TRAIN” list and “VAL” list)
        for img_path, label_name in labels.items():
            if random.random() < split_ratio: 
                train_images.append("C:\\Users\\alant\\Documents\\POLYGENCE_PROJECT\\DND-Diko-WWWR\\WR2021\\images\\"+img_path)
                train_labels.append(label_name)
            else:
                val_images.append("C:\\Users\\alant\\Documents\\POLYGENCE_PROJECT\\DND-Diko-WWWR\\WR2021\\images\\"+img_path)
                val_labels.append(label_name)
        
        # Save each of the lists to a pickle file
        with open('train_images.pkl', 'wb') as file:
            pkl.dump(train_images, file)
        
        with open('train_labels.pkl', 'wb') as file:
            pkl.dump(train_labels, file)
        
        with open('val_images.pkl', 'wb') as file:
            pkl.dump(val_images, file)
        
        with open('val_labels.pkl', 'wb') as file:
            pkl.dump(val_labels, file)

        self.train_images = train_images
        self.train_labels = train_labels
        self.val_images = val_images
        self.val_labels = val_labels

# Create an instance of the DND class
dnd = DND()
