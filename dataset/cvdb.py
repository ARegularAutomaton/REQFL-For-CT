import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import Dataset

import os

def CVDB_CVPR(dataset_name='CT10', mode='train', batch_size=1, shuffle=True, crop_size=(256, 256), resize=False):
    assert os.path.exists(f'dataset/{dataset_name}/')
    imgs_path = f'dataset/{dataset_name}/'

    if resize:
        transform_data = transforms.Compose([transforms.CenterCrop(crop_size),
                                             transforms.Resize(int(crop_size[0]/2)),
                                             transforms.ToTensor()])
    else:
        transform_data = transforms.Compose([transforms.Resize(crop_size),
                                            transforms.CenterCrop(crop_size),
                                            transforms.ToTensor(),
                                            transforms.Grayscale()])
    if mode == 'train':
        imgs_path = imgs_path + 'train/'
    if mode == 'test':
        imgs_path = imgs_path + 'test/'

    dataset = datasets.ImageFolder(imgs_path, transform=transform_data, target_transform=None)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    return dataloader