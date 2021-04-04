import time
import torch

import numpy as np

from model import TransferNet
from torchvision import models
from utils import get_data_loaders
from torch import nn
from sklearn.metrics import accuracy_score


trained_models_path = "trained_models/"


def run_training(fold, config, mode='clf'):
    since = time.time()

    dataloaders, dataset_sizes = get_data_loaders(fold, config)

    model = TransferNet(models.resnet18(pretrained=True), use_dropout=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    if mode == 'clf':
        epochs = config['clf_epochs']
    else:
        epochs = config['reg_epochs']

    for epoch in range(epochs):
        print('Epoch {}/{}'.format(epoch, epochs - 1))
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            preds_accum = []
            targets_accum = []

            for batch in dataloaders[phase]:
                image = batch['image'].to(device)
                if mode == 'clf':
                    target = batch['clf_target']
                else:
                    target = batch['reg_target']

                target = target.to(device, torch.float)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    pred = model(image)

                    target = target.unsqueeze(1)

                    if mode == 'clf':
                        loss = nn.BCEWithLogitsLoss()(pred, target)
                    else:
                        loss = nn.L1Loss()(pred, target)

                    if mode == 'clf':
                        pred = torch.round(torch.sigmoid(pred))

                    pred = pred.cpu().detach().numpy().tolist()
                    target = target.cpu().detach().numpy().tolist()

                    preds_accum.extend(pred)
                    targets_accum.extend(target)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * image.size(0)

            epoch_loss = running_loss / dataset_sizes[phase]

            if mode == 'clf':
                epoch_acc = accuracy_score(pred, target)
                print('ACC: {}'.format(np.round(epoch_acc, 3)))

            print('{} Loss: {}'.format(
                phase, np.round(epoch_loss, 3)))

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    torch.save(model.state_dict(), trained_models_path + "{}.pth".format(mode))
