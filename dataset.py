import os
import torch

from torch.utils.data import Dataset
from PIL import Image


public_test_dir = "idao_dataset/public_test/"
private_test_dir = "idao_dataset/private_test/"


class ImageDataset(Dataset):
    def __init__(self, df, is_train=True, transforms=None):

        self.df = df
        self.is_train = is_train
        self.transforms = transforms
        if is_train:
            self.df['classification_label'] = self.df['classification_label'] == 'ER'
            self.df['regression_label'] = df['regression_label']

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        clf_target = None
        reg_target = None

        if self.is_train:
            image_path = self.df.iloc[idx]['image_path']
            clf_target = torch.tensor(self.df.iloc[idx]['classification_label'], dtype=torch.float)
            reg_target = torch.tensor(self.df.iloc[idx]['regression_label'], dtype=torch.float)
        else:
            image_name = self.df.iloc[idx]['id']
            if os.path.exists(public_test_dir + image_name + '.png'):
                image_path = public_test_dir + image_name + '.png'
            else:
                image_path = private_test_dir + image_name + '.png'
        image = Image.open(image_path)
        image = image.convert('RGB')

        if self.transforms:
            image = self.transforms(image)

        if self.is_train:
            return {
                'image': image,
                'clf_target': clf_target,
                'reg_target': reg_target,
            }

        return {'image': image}

    def __len__(self):
        return len(self.df)
