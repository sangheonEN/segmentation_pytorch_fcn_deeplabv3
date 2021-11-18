import os
import torch
from torch.utils.data import DataLoader
from utils import get_config, get_log_dir, get_cuda
from data_loader import *
from trainer import Trainer
import argparse
import warnings
warnings.filterwarnings('ignore')

train_input_path = './datasets/VOCdata/train/input_data'
train_mask_path = './datasets/VOCdata/train/mask_data'
validation_input_path = './datasets/VOCdata/val/input_data'
validation_mask_path = './datasets/VOCdata/val/mask_data'

def data_sort_list(input_path, mask_path):

    sort_function = lambda f: int(''.join(filter(str.isdigit, f)))

    input_list = os.listdir(input_path)
    input_list = [file for file in input_list if file.endswith("png")]
    input_list.sort(key=sort_function)
    mask_list = os.listdir(mask_path)
    mask_list = [file for file in mask_list if file.endswith('png')]
    mask_list.sort(key=sort_function)

    return input_list, mask_list


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # Model hyper-parameters
    parser.add_argument('--mode', type=str, default='trainval', choices=['trainval', 'inference'])
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--backbone", type=str, default='resnet')
    parser.add_argument("--model", type=str, default='fcn', choices=['fcn', 'deeplabv3'])
    parser.add_argument("--resume", type=str, default='', help='model saver path')
    parser.add_argument("--backbone_layer", type=str, default='101', choices=['50', '101'])
    opts = parser.parse_args()

    # os.environ['CUDA_VISIBLE_DEVICES'] = str(opts.gpu_id)
    opts.cuda = get_cuda(torch.cuda.is_available() and opts.gpu_id != -1,
                         opts.gpu_id)
    print('Cuda', opts.cuda)
    cfg = get_config()[1]
    opts.cfg = cfg

    if opts.model == 'deeplabv3':
        if opts.mode in ['train', 'trainval']:
            opts.out = get_log_dir('deeplabv3_' + opts.backbone_layer, cfg)
            print('Output logs: ', opts.out)
    else:
        if opts.mode in ['train', 'trainval']:
            opts.out = get_log_dir('fcn_' + opts.backbone_layer, cfg)
            print('Output logs: ', opts.out)

    train_input_list, train_mask_list = data_sort_list(train_input_path, train_mask_path)
    val_input_list, val_mask_list = data_sort_list(validation_input_path, validation_mask_path)

    training_data = CustomImageDataset(train_input_list, train_mask_list, train_input_path, train_mask_path,
                                       is_train_data=True, resize=resize_data, transform=transform)
    validation_data = CustomImageDataset(val_input_list, val_mask_list, validation_input_path, validation_mask_path,
                                         is_train_data=False, resize=resize_data, transform=transform)

    train_dataloader = DataLoader(training_data, batch_size=3, shuffle=True)
    valid_dataloader = DataLoader(validation_data, batch_size=3, shuffle=False)

    trainer = Trainer(train_dataloader, valid_dataloader, opts)

    trainer.Train()









