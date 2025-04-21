import glob
import os
import random
import cv2
import numpy as np
from PIL import Image, ImageOps, ImageEnhance
import torch
from torch.utils import data

def get_data_paths():
    path_train_nir = './FANVID_dataset/NIR'
    path_train_rgb = './FANVID_dataset/RGB'
    path_val = './FANVID_dataset/Val'

    nir_files_png1 = glob.glob(path_train_nir + '/*.jpg')
    # nir_files_png2 = glob.glob(path_train_nir + '/NIR*.png')
    nir_files = sorted(nir_files_png1 )

    nir_files_png1_val = glob.glob(os.path.join(path_val, 'nir_val/*.jpg'))
    # nir_files_png2_val = glob.glob(os.path.join(path_val, 'NIR_*.png'))
    nir_files_val = sorted(nir_files_png1_val )

    rgb_files_png = glob.glob(path_train_rgb + '/*.jpg')
    # rgb_files_jpg = glob.glob(path_train_rgb + '/RGBR*.jpg')
    rgb_files = sorted( rgb_files_png)

    rgb_files_png_val = sorted(glob.glob(os.path.join(path_val, 'rgb_val/*.jpg')))
    # rgb_files_jpg_val = sorted(glob.glob(os.path.join(path_val, 'RGBR*.jpg')))
    rgb_files_val = sorted( rgb_files_png_val)

    print("Number of NIR files:", len(nir_files))
    print("Number of RGB files:", len(rgb_files))
    print("Number of val NIR files:", len(nir_files_val))
    print("Number of val RGB files:", len(rgb_files_val))

    train_files = np.stack([nir_files, rgb_files], axis=1)
    val_files = np.stack([nir_files_val, rgb_files_val], axis=1)
    return train_files, val_files

def get_test_paths():

    path_val = './RGB2NIR_datasets/Testing'
    nir_files_png2 = sorted(glob.glob(os.path.join(path_val, 'nir/*.tiff')))
    # nir_files_png2 = sorted(glob.glob(os.path.join(path_val, 'NIR*.png')))
    nir_files_val = sorted(nir_files_png2)

    rgb_files_png = sorted(glob.glob(os.path.join(path_val, 'rgb/*.tiff')))
    # rgb_files_png = sorted(glob.glob(os.path.join(path_val, 'RGBR*.jpg')))
    rgb_files_val = sorted(rgb_files_png)

    print("Number of NIR files:", len(nir_files_val))
    print("Number of RGB files:", len(rgb_files_val))
    val_files = np.stack([nir_files_val, rgb_files_val], axis=1)
    return val_files


def nor(img):
    img -= np.min(img)
    img = img / (np.max(img) + 1e-3)
    return img

class Dataset(data.Dataset):
    def __init__(self, files, target_shape=None, return_name=False):
        self.files = files
        self.return_name = return_name
        self.target_shape = target_shape

    def __len__(self):
        return len(self.files)

    def resize_if_needed(self, img):
        width, height = img.size
        if width > 512 or height > 512:
            if width >= height:
                new_width = 512
                new_height = int((512 / width) * height)
            else:
                new_height = 512
                new_width = int((512 / height) * width)
            img = img.resize((new_width, new_height), Image.BICUBIC)
        return img

    def read_data(self, img_path):
        img = Image.open(img_path).convert('RGB')
        img_gray = Image.open(img_path).convert('L')

        img = self.resize_if_needed(img)
        img_gray = self.resize_if_needed(img_gray)

        img_rgb = np.array(img)
        img_rgb = nor(img_rgb)
        img_gray = np.array(img_gray)
        img_gray = nor(img_gray)
        img_hsv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2HSV)
        img_hsv = nor(img_hsv)

        img_rgb = torch.from_numpy(img_rgb.transpose(2, 0, 1)).float()
        img_hsv = torch.from_numpy(img_hsv.transpose(2, 0, 1)).float()
        img_gray = torch.from_numpy(img_gray).unsqueeze(0).float()

        return img_gray, img_rgb, img_hsv

    def __getitem__(self, index):
        nir_path = self.files[index][0]
        rgb_path = self.files[index][1]
        nir_gray, nir_rgb, nir_hsv = self.read_data(nir_path)
        rgb_gray, rgb_rgb, rgb_hsv = self.read_data(rgb_path)

        sample = {
            'nir_gray': nir_gray,
            'nir_rgb': nir_rgb,
            'nir_hsv': nir_hsv,
            'rgb_gray': rgb_gray,
            'rgb_rgb': rgb_rgb,
            'rgb_hsv': rgb_hsv
        }

        if self.return_name:
            sample['nir_path'] = nir_path
            sample['rgb_path'] = rgb_path

        return sample

class Dataset_test(data.Dataset):
    def __init__(self, files, target_shape=None, return_name=True):
        self.files = files
        self.return_name = return_name
        self.target_shape = target_shape

    def __len__(self):
        return len(self.files)

    def read_data(self, img_path):
        img = Image.open(img_path).convert('RGB')
        img_gray = Image.open(img_path).convert('L')

        # Store original size
        original_size = img.size  # (width, height)

        # No resizing unless explicitly needed later; keep original size
        img_rgb = np.array(img)
        img_rgb = nor(img_rgb)
        img_gray = np.array(img_gray)
        img_gray = nor(img_gray)
        img_hsv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2HSV)
        img_hsv = nor(img_hsv)

        img_rgb = torch.from_numpy(img_rgb.transpose(2, 0, 1)).float()
        img_hsv = torch.from_numpy(img_hsv.transpose(2, 0, 1)).float()
        img_gray = torch.from_numpy(img_gray).unsqueeze(0).float()

        return img_gray, img_rgb, img_hsv, original_size

    def __getitem__(self, index):
        nir_path = self.files[index][0]
        rgb_path = self.files[index][1]
        nir_gray, nir_rgb, nir_hsv, nir_size = self.read_data(nir_path)
        rgb_gray, rgb_rgb, rgb_hsv, rgb_size = self.read_data(rgb_path)

        sample = {
            'nir_gray': nir_gray,
            'nir_rgb': nir_rgb,
            'nir_hsv': nir_hsv,
            'rgb_gray': rgb_gray,
            'rgb_rgb': rgb_rgb,
            'rgb_hsv': rgb_hsv,
            'rgb_path': rgb_path,
            'original_size': rgb_size  # Store original size of RGB image
        }
        return sample

if __name__ == '__main__':
    train_files, _ = get_data_paths()
    dataset = Dataset(train_files, target_shape=None)
    sample = dataset[0]
    print("NIR image shape:", sample['nir_rgb'].shape)
    print("RGB image shape:", sample['rgb_rgb'].shape)