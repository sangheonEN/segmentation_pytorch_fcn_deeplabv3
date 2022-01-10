import torch
import numpy as np
import os
import os.path as osp
import cv2
import yaml
import albumentations as A

def get_log_dir(model_name, cfg):

    name = 'MODEL-%s' % (model_name)

    log_dir = osp.join('logs', name)
    if not osp.exists(log_dir):
        os.makedirs(log_dir)
    with open(osp.join(log_dir, 'config.yaml'), 'w') as f:
        yaml.safe_dump(cfg, f, default_flow_style=False)
    return log_dir

def get_config():
    return {
        # same configuration as original work
        # https://github.com/shelhamer/fcn.berkeleyvision.org
        1:
        dict(
            max_iteration=26620,
            lr=1.0e-4,
            momentum=0.99,
            weight_decay=0.0005
            # interval_validate=4000
        )
    }


def get_cuda(cuda, _id):
    if not cuda:
        return torch.device('cpu')
    else:
        return torch.device('cuda:{}'.format(_id))


def run_fromfile(model, img_file, cuda):
    img_torch = img_file
    img_torch = img_torch.to(cuda)
    model.eval()
    with torch.no_grad():
        score = model(img_torch)
        return score


def _fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(n_class * label_true[mask].astype(int) +
                       label_pred[mask],
                       minlength=n_class**2).reshape(n_class, n_class)
    return hist


def label_accuracy_score(label_trues, label_preds, n_class):
    """Returns accuracy score evaluation result.

      - overall accuracy
      - mean accuracy
      - mean IU
      - fwavacc

    """
    hist = np.zeros((n_class, n_class))
    for lt, lp in zip(label_trues, label_preds):
        hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)
    acc = np.diag(hist).sum() / hist.sum()
    acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    mean_iu = np.nanmean(iu)
    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    return acc, acc_cls, mean_iu, fwavacc


def augmentation_train(inputs_train_feed, masks_train_feed):
    # torch tensor operation기준인 input shape이 (batch, channel, height, width)로 들어오니
    # 이걸 augmentation 하기위해서는 (batch, height, width, channel)로 변환 필요.
    inputs_train_feed = np.array(inputs_train_feed).transpose(0, 2, 3, 1)
    masks_train_feed = np.array(masks_train_feed).transpose(0, 2, 3, 1)

    mean_resnet = np.array([0.485, 0.456, 0.406])
    std_resnet = np.array([0.229, 0.224, 0.225])

    transform = A.Compose([
        A.RandomRotate90(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5)
    ])

    iter_num = inputs_train_feed.shape[0]

    inputs_list = list()
    masks_list = list()

    for i in range(iter_num):
        transformed = transform(image=inputs_train_feed[i], mask=masks_train_feed[i])
        input = transformed["image"]
        mask = transformed["mask"]
        input /= 255.
        # mask_ = mask / 255
        input -= mean_resnet
        input /= std_resnet

        inputs_list.append(input)
        masks_list.append(mask)

    # torch tensor operation기준인 input shape(batch, channel, height, width)으로 다시 변환.
    img = np.array(inputs_list, dtype=np.float32).transpose(0, 3, 1, 2)
    lbl = np.array(masks_list, dtype=np.int32).transpose(0, 3, 1, 2)

    img = torch.from_numpy(img.copy()).float()
    lbl = torch.from_numpy(lbl.copy()).long()

    return img, lbl
