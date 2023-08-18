# from sklearn.metrics import confusion_matrix
# import numpy as np
# import os

# # Paths as mentioned in evaluate.py script
# input_dir = 'codalab'
# split = 'test'
# submit_dir = os.path.join(input_dir, 'res'+'_'+split)
# truth_dir = os.path.join(input_dir, 'ref'+'_'+split)

# labels_ww2020_path = os.path.join(truth_dir, "labels_WW2020.txt")
# labels_wr2021_path = os.path.join(truth_dir, "labels_WR2021.txt")
# pred_ww2020_path = os.path.join(submit_dir, "predictions_WW2020.txt")
# pred_wr2021_path = os.path.join(submit_dir, "predictions_WR2021.txt")

# # Extracting true labels and predictions
# sorted_truth0 = [item.split(' ')[1].strip() for item in sorted(open(labels_ww2020_path).readlines())]
# sorted_pred0 = [item.split(' ')[1].strip() for item in sorted(open(pred_ww2020_path).readlines())]
# sorted_truth1 = [item.split(' ')[1].strip() for item in sorted(open(labels_wr2021_path).readlines())]
# sorted_pred1 = [item.split(' ')[1].strip() for item in sorted(open(pred_wr2021_path).readlines())]

# # Computing the confusion matrices
# conf_matrix_ww2020 = confusion_matrix(sorted_truth0, sorted_pred0, labels=np.unique(sorted_truth0))
# conf_matrix_wr2021 = confusion_matrix(sorted_truth1, sorted_pred1, labels=np.unique(sorted_truth1))

# print("Confusion Matrix for WW2020:")
# print(conf_matrix_ww2020)
# print("\nConfusion Matrix for WR2021:")
# print(conf_matrix_wr2021)


"""
Script to generate predictions for the validation set.
Based on main_dnd.py and dnd_dataset.py.
"""

from pprint import pprint
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

from configs.config import cfg_from_file, project_root, get_arguments
from datasets.dnd_dataset import DND
from models.my_model import MyModel
from utils.pytorch_misc import * 
import yaml

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def predict_val():
    # Load configuration
    args = get_arguments()
    cfg_from_file(args.cfg)
    cfg.codalab_pred = 'val'  # Ensure we're targeting the validation set

    # Create validation dataset and loader
    val_dataset = DND(cfg, split='val')
    val_loader = DataLoader(val_dataset, batch_size=1, num_workers=cfg.NWORK, drop_last=False)

    # Load the model
    model = MyModel(cfg).to(device)
    model = torch.nn.DataParallel(model)

    # Restore model weights from the checkpoint
    checkpoint = torch.load(args.restore_from)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Evaluate on the validation set
    model.eval()
    with torch.no_grad():
        for batch in val_loader:
            img = batch['img'].to(device)
            labels = batch['label_idxs'].to(device)
            scores = model(img)
            # TODO: Store the predictions as needed

    print("Validation predictions generated!")

if __name__ == '__main__':
    predict_val()
