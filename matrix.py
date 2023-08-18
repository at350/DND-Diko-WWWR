from sklearn.metrics import confusion_matrix
import numpy as np
import os

# Paths as mentioned in evaluate.py script
input_dir = 'codalab'
split = 'test'
submit_dir = os.path.join(input_dir, 'res'+'_'+split)
truth_dir = os.path.join(input_dir, 'ref'+'_'+split)

labels_ww2020_path = os.path.join(truth_dir, "labels_WW2020.txt")
labels_wr2021_path = os.path.join(truth_dir, "labels_WR2021.txt")
pred_ww2020_path = os.path.join(submit_dir, "predictions_WW2020.txt")
pred_wr2021_path = os.path.join(submit_dir, "predictions_WR2021.txt")

# Extracting true labels and predictions
sorted_truth0 = [item.split(' ')[1].strip() for item in sorted(open(labels_ww2020_path).readlines())]
sorted_pred0 = [item.split(' ')[1].strip() for item in sorted(open(pred_ww2020_path).readlines())]
sorted_truth1 = [item.split(' ')[1].strip() for item in sorted(open(labels_wr2021_path).readlines())]
sorted_pred1 = [item.split(' ')[1].strip() for item in sorted(open(pred_wr2021_path).readlines())]

# Computing the confusion matrices
conf_matrix_ww2020 = confusion_matrix(sorted_truth0, sorted_pred0, labels=np.unique(sorted_truth0))
conf_matrix_wr2021 = confusion_matrix(sorted_truth1, sorted_pred1, labels=np.unique(sorted_truth1))

print("Confusion Matrix for WW2020:")
print(conf_matrix_ww2020)
print("\nConfusion Matrix for WR2021:")
print(conf_matrix_wr2021)
