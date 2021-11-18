from torchvision.models.segmentation import deeplabv3_resnet101
import torch.nn as nn


class Deeplabv3(nn.Module):
    def __init__(self, n_class=21):

        self.n_class = n_class
        super(Deeplabv3, self).__init__()
        self.deeplabv3 = deeplabv3_resnet101(pretrained=False, num_classes=self.n_class)

    def forward(self, x, debug=False):
        return self.deeplabv3(x)['out']

    def resume(self, file, test=False):
        import torch
        if test and not file:
            self.deeplabv3 = deeplabv3_resnet101(pretrained=True, num_classes=self.n_class)
            return
        if file:
            print('Loading checkpoint from: ' + file)
            checkpoint = torch.load(file)
            print(f"best epoch : {checkpoint['epoch']}")
            checkpoint = checkpoint['model_state_dict']
            self.load_state_dict(checkpoint)