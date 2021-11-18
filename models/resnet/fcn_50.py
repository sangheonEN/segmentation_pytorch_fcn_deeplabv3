from torchvision.models.segmentation import fcn_resnet50
import torch.nn as nn


class FCN(nn.Module):
    def __init__(self, n_class=21):

        self.n_class = n_class
        super(FCN, self).__init__()
        self.fcn = fcn_resnet50(pretrained=False, num_classes=self.n_class)

    def forward(self, x, debug=False):
        return self.fcn(x)['out']

    def resume(self, file, test=False):
        import torch
        if test and not file:
            self.fcn = fcn_resnet50(pretrained=True, num_classes=self.n_class)
            return
        if file:
            print('Loading checkpoint from: ' + file)
            checkpoint = torch.load(file)
            print(f"best epoch : {checkpoint['epoch']}")
            checkpoint = checkpoint['model_state_dict']
            self.load_state_dict(checkpoint)
