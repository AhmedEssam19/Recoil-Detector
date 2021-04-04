import pathlib

import pandas as pd
from sklearn.model_selection import StratifiedKFold

from dataset import ImageDataset
from torch.utils.data import DataLoader
from torchvision import transforms


def get_data_set(training_set_path: str, save_to_csv: bool = True):
    def get_energy_from_image_name(name: str):
        tmp = name.split('_')
        if tmp[6] == 'NR':
            return int(tmp[7])
        else:
            return int(tmp[6])

    train_set = []

    for category in pathlib.Path(training_set_path).iterdir():
        for image in category.iterdir():
            train_set.append([str(image), category.name, get_energy_from_image_name(image.name)])

    df = pd.DataFrame(train_set, columns=['image_path', 'classification_label', 'regression_label'])

    if save_to_csv:
        df.to_csv('train.csv')
    
    else:
        return df


def get_data(train_df, val_df, config):
    data_transforms = {
        'train':
            transforms.Compose([
                transforms.ToTensor(),
                transforms.CenterCrop(config['input_size'])]),

        'val':
            transforms.Compose([
                transforms.ToTensor(),
                transforms.CenterCrop(config['input_size'])])
    }

    image_datasets = {'train': ImageDataset(train_df, transforms=data_transforms['train']),
                      'val': ImageDataset(val_df, transforms=data_transforms['val'])}

    dataloaders = {
        'train': DataLoader(image_datasets['train'], batch_size=config['batch_size'], shuffle=True,
                            num_workers=config['workers']),
        'val': DataLoader(image_datasets['val'], batch_size=2 * config['batch_size'], shuffle=False,
                          num_workers=config['workers'])
    }

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

    return dataloaders, dataset_sizes


def create_folds(df, n_folds, config):
    df["kfold"] = -1
    df = df.sample(frac=1).reset_index(drop=True)
    y = df.regression_label.values
    kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=config['seed'])
    for f, (t_, v_) in enumerate(kf.split(X=df, y=y)):
        df.loc[v_, 'kfold'] = f

    df.to_csv("train_folds.csv", index=False)


def get_data_loaders(fold, config):
    df = pd.read_csv("train_folds.csv")
    df_train = df[df.kfold != fold].reset_index(drop=True)
    df_val = df[df.kfold == fold].reset_index(drop=True)

    return get_data(df_train, df_val, config)


def transform(value):
    energies = [1, 3, 6, 10, 20, 30]
    right_energy = None
    min_diff = -1

    for energy in energies:
        diff = abs(energy - value)
        if right_energy is None or diff < min_diff:
            min_diff = diff
            right_energy = energy

    return right_energy
