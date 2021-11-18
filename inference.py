import os
import torch
from torch.utils.data import DataLoader
from utils import get_config, get_log_dir, get_cuda
from data_loader import *
from inference_utils import Inference
import argparse
import warnings
warnings.filterwarnings('ignore')

test_input_path = './datasets/VOCdata/test/input_data'
test_mask_path = './datasets/VOCdata/test/mask_data'


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

    opts.cuda = get_cuda(torch.cuda.is_available() and opts.gpu_id != -1,
                         opts.gpu_id)
    print('Cuda', opts.cuda)
    cfg = get_config()[1]
    opts.cfg = cfg

    if opts.model == 'deeplabv3':
        if opts.mode in ['trainval', 'inference']:
            opts.out = get_log_dir('deeplabv3_' + opts.backbone_layer, cfg)
            print('Output logs: ', opts.out)
    else:
        if opts.mode in ['trainval', 'inference']:
            opts.out = get_log_dir('fcn_' + opts.backbone_layer, cfg)
            print('Output logs: ', opts.out)

    test_input_list, test_mask_list = data_sort_list(test_input_path, test_mask_path)

    test_data = CustomImageDataset(test_input_list, test_mask_list, test_input_path, test_mask_path,
                                   is_train_data=False, resize=resize_data, transform=transform)

    test_dataloader = DataLoader(test_data, batch_size=3, shuffle=False)

    inference = Inference(test_dataloader, opts)

    inference.Test()

