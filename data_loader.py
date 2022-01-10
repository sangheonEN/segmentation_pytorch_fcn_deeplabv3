import torch
import numpy as np
from torch.utils.data import Dataset
import cv2
import os
import albumentations as A
import matplotlib.pyplot as plt

VOC_COLORMAP = [
    [0, 0, 0],
    [128, 0, 0],
    [0, 128, 0],
    [128, 128, 0],
    [0, 0, 128],
    [128, 0, 128],
    [0, 128, 128],
    [128, 128, 128],
    [64, 0, 0],
    [192, 0, 0],
    [64, 128, 0],
    [192, 128, 0],
    [64, 0, 128],
    [192, 0, 128],
    [64, 128, 128],
    [192, 128, 128],
    [0, 64, 0],
    [128, 64, 0],
    [0, 192, 0],
    [128, 192, 0],
    [0, 64, 128],
]

class CustomImageDataset(Dataset):

    class_names = np.array([
        'background',
        'aeroplane',
        'bicycle',
        'bird',
        'boat',
        'bottle',
        'bus',
        'car',
        'cat',
        'chair',
        'cow',
        'diningtable',
        'dog',
        'horse',
        'motorbike',
        'person',
        'potted plant',
        'sheep',
        'sofa',
        'train',
        'tv/monitor',
    ])

    def decode_segmap(self, label_mask, plot=False):
        """Decode segmentation class labels into a color image

        Args:
            label_mask (np.ndarray): an (M,N) array of integer values denoting
              the class label at each spatial location.
            plot (bool, optional): whether to show the resulting color image
              in a figure.

        Returns:
            (np.ndarray, optional): the resulting decoded color image.
        """
        label_colours = self.get_pascal_labels()
        r = label_mask.copy()
        g = label_mask.copy()
        b = label_mask.copy()
        for ll in range(0, self.n_classes):
            r[label_mask == ll] = label_colours[ll, 0]
            g[label_mask == ll] = label_colours[ll, 1]
            b[label_mask == ll] = label_colours[ll, 2]
        rgb = np.zeros((label_mask.shape[0], label_mask.shape[1], 3))
        rgb[:, :, 0] = r / 255.0
        rgb[:, :, 1] = g / 255.0
        rgb[:, :, 2] = b / 255.0
        if plot:
            plt.imshow(rgb)
            plt.show()
        else:
            return rgb


    def _convert_to_segmentation_mask(self, mask):
        """Encode segmentation label images as pascal classes

        Args:
            mask (np.ndarray): raw segmentation label image of dimension
              (M, N, 3), in which the Pascal classes are encoded as colours.

        Returns:
            (np.ndarray): class map with dimensions (M,N), where the value at
            a given location is the integer denoting the class index.
        """
        mask = mask.astype(int)
        label_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int16)
        for ii, label in enumerate(VOC_COLORMAP):
            label_mask[np.where(np.all(mask == label, axis=-1))[:2]] = ii
        label_mask = label_mask.astype(int)
        return label_mask

    def untransform(self, img, lbl):
        mean_resnet = np.array([0.485, 0.456, 0.406])
        std_resnet = np.array([0.229, 0.224, 0.225])

        # 입력은 channel, height, width형식으로 들어오니
        # height, width, channel shape으로 transpose 해야함.
        img = img.numpy()
        img = img.transpose(1, 2, 0)
        img *= std_resnet
        img += mean_resnet
        img *= 255
        img = img.astype(np.uint8)

        lbl = lbl.numpy()
        lbl = lbl.transpose(1, 2, 0)
        lbl * 255

        return img, lbl

    def __init__(self, img_list, mask_list, img_path, mask_path, is_train_data = False, resize=None, transform=None):
        self.img_list = img_list
        self.mask_list = mask_list
        self.img_path = img_path
        self.mask_path = mask_path
        self.transform = transform
        self.resize = resize
        self.is_train_data = is_train_data

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        image = cv2.imread(os.path.join(self.img_path, self.img_list[idx]))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(os.path.join(self.mask_path, self.mask_list[idx]))
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        mask = self._convert_to_segmentation_mask(mask)
        mask = np.expand_dims(mask, -1)

        if self.transform is not None:
            image, mask = self.transform(image, mask, self.is_train_data)

        image, mask = self.resize(image, mask)

        return image, mask

def transform(img, lbl, _is_train):
    # train is done separately in transform. see utils.py
    if _is_train == False:
        mean_resnet = np.array([0.485, 0.456, 0.406])
        std_resnet = np.array([0.229, 0.224, 0.225])

        img = np.array(img, dtype=np.float64)
        img /= 255.
        lbl = np.array(lbl, dtype=np.float64)
        # lbl /= 255
        img -= mean_resnet
        img /= std_resnet
        return img, lbl

    else:
        return img, lbl

def resize_data(inputs_train_feed, masks_train_feed):
    # augmentation
    transform = A.Compose([
        A.Resize(1024, 1024)
    ])

    transformed = transform(image=inputs_train_feed, mask=masks_train_feed)

    img = transformed["image"]
    lbl = transformed["mask"]

    img = img.transpose(2, 0, 1) # mask 는 h, w shape이니까 필요 없음??
    lbl = lbl.transpose(2, 0, 1) # mask 는 h, w shape이니까 필요 없음??
    img = torch.from_numpy(img.copy()).float()
    lbl = torch.from_numpy(lbl.copy()).long()

    return img, lbl





































