import time

import numpy as np
import pandas as pd
import torch

from torchvision import transforms, models
from dataset import ImageDataset
from model import TransferNet
from torch.utils.data import DataLoader

trained_models_path = "trained_models/"
sub_df_path = "track1_predictions_example.csv"


def predict(config):
    since = time.time()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    clf_model_path = trained_models_path + "clf.pth"
    reg_model_path = trained_models_path + "reg.pth"

    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.CenterCrop(config['input_size'])])

    sub_df = pd.read_csv(sub_df_path)

    test_dataset = ImageDataset(
        df=sub_df,
        transforms=test_transforms,
        is_train=False
    )

    test_loader = DataLoader(
        test_dataset, batch_size=config['batch_size'] * 2, shuffle=False, num_workers=config['workers'])
    reg_model = TransferNet(models.resnet18(pretrained=False))
    clf_model = TransferNet(models.resnet18(pretrained=False))

    reg_model.load_state_dict(torch.load(reg_model_path))
    clf_model.load_state_dict(torch.load(clf_model_path))
    reg_model.to(device)
    clf_model.to(device)

    reg_preds_accum = []
    clf_preds_accum = []

    for batch in test_loader:
        image = batch['image'].to(device)
        with torch.no_grad():
            reg_preds = reg_model(image)
            clf_preds = clf_model(image)

            reg_preds = reg_preds.cpu().detach().numpy()
            clf_preds = clf_preds.cpu().detach().numpy()

            reg_preds_accum.extend(reg_preds)
            clf_preds_accum.extend(clf_preds)

    time_elapsed = time.time() - since
    print('Inference complete {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    reg_preds_accum = np.array(reg_preds_accum)
    clf_preds_accum = torch.tensor(clf_preds_accum)
    clf_preds_accum = torch.round(torch.sigmoid(clf_preds_accum)).numpy()

    return clf_preds_accum, reg_preds_accum
