import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.v2 as T

from tqdm import tqdm

from datasets import load_dataset


MEAN = [0.49139968, 0.48215827, 0.44653124]
STD = [0.24703233, 0.24348505, 0.26158768]


class CIFAR10DataModule(Dataset):
    def __init__(self, batch_size: int = 8, cut: float = 1.0, cut_valid: bool = False):
        dataset = load_dataset('cifar10')

        self.train_transform = T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomRotation(10),
            T.RandomAffine(0, shear=10, scale=(0.8,1.2)),
            T.PILToTensor(),
            T.ConvertImageDtype(dtype=torch.float),
            T.Normalize(mean=MEAN, std=STD),
        ])

        self.valid_transform = T.Compose([
            T.PILToTensor(),
            T.ConvertImageDtype(dtype=torch.float),
            T.Normalize(mean=MEAN, std=STD),
        ])

        self.label_encoding = torch.eye(10)

        self.train_dataset = dataset['train']
        self.valid_dataset = dataset['test']

        # if cut != 1.0:
        #     self.sample_count = int(5000 * cut)
        #     self._cut_dataset()
        #     if cut_valid:
        #         self._cut_valid_dataset()
        
        # self._prepare_data()
    
    # def _prepare_data(self):
    #     for idx, data in tqdm(enumerate(self.train_dataset), desc='preparing train dataset\'s labels'):
    #         current_label = data['label']
    #         self.train_dataset[idx]['label'] = self.label_encoding[current_label]
        
    #     for idx, data in tqdm(enumerate(self.valid_dataset), desc='preparing valid dataset\'s labels'):
    #         current_label = data['label']
    #         self.valid_dataset[idx]['label'] = self.label_encoding[current_label]

    # def _cut_dataset(self):
    #     label_count = {k: 0 for k in range(10)}
    #     train_dataset = []
    #     for data in tqdm(self.train_dataset, desc='cutting the dataset'):
    #         if label_count[data['label']] >= self.sample_count:
    #             continue
    #         train_dataset.append(data)
    #         label_count[data['label']] += 1

    #     self.train_dataset = train_dataset

    # def _cut_valid_dataset(self):
    #     label_count = {k: 0 for k in range(10)}
    #     valid_dataset = []
    #     for data in tqdm(self.valid_dataset, desc='cutting the dataset'):
    #         if label_count[data['label']] >= self.sample_count // 5:
    #             continue
    #         valid_dataset.append(data)
    #         label_count[data['label']] += 1

    #     self.valid_dataset = valid_dataset

    # def _transform(self, data):
    #     data['img'] = [self.train_transform]

    def train_collate(self, batch):
        images = torch.stack([self.train_transform(d['img']) for d in batch])
        labels = torch.stack([self.label_encoding[d['label']] for d in batch])
        return {
            'images': images,
            'labels': labels,
        }

    def valid_collate(self, batch):
        images = torch.stack([self.valid_transform(d['img']) for d in batch])
        labels = torch.stack([self.label_encoding[d['label']] for d in batch])
        return {
            'images': images,
            'labels': labels,
        }
