from pprint import pprint
import torch
from torch.utils.data import DataLoader
from datasets.dnd_dataset import DND
from models.my_model import MyModel
from utils.pytorch_misc import load_model
from configs.config import cfg_from_file, project_root, get_arguments

def predict_val(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize validation dataset and loader
    val_dataset = DND(cfg, split='val')
    val_loader = DataLoader(val_dataset, batch_size=1, num_workers=cfg.NWORK, drop_last=False)

    cfg.num_classes = len(val_dataset.class_to_ind)

    # Load model
    # model = MyModel(cfg)
    # model, _ = load_model(cfg, model)
    # model.to(device)
    # model.eval()

    model = MyModel(cfg)
    model_path = cfg.restore_from
    print(f"Loading model from {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    model.eval()

    model_weights = checkpoint
    model.load_state_dict(model_weights, strict=False)

    model.to(device)

    val_predictions = []

    # Loop through validation data and predict
    with torch.no_grad():
        for batch in val_loader:
            images = batch['img'].to(device)
            scores = model(images)
            _, predicted = scores.max(1)
            val_predictions.extend(predicted.tolist())
    
    # Save predictions to a file
    with open('val_predictions.txt', 'w') as f:
        for i, pred in enumerate(val_predictions):
            f.write(f"{val_dataset.img_paths[i]} {pred}\n")

    print(f"Predictions for validation data saved to val_predictions.txt.")

if __name__ == '__main__':
    args = get_arguments()  # Get command line arguments
    cfg = cfg_from_file(args.cfg)  # Load the configuration from the provided file
    cfg.update(vars(args))  # Update the configuration with command line arguments
    predict_val(cfg)  # Pass the configuration to the predict_val function
