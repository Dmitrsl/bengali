import torch
import torch.nn as nn
import torch.nn.functional as F
import pretrainedmodels


class ResModel(nn.Module):

    def __init__(self, arch, pretrained="imagenet"):
        super(ResModel, self).__init__()
        self.arch = arch
        self.pretrained = pretrained
        self.model = pretrainedmodels.__dict__[self.arch](pretrained=self.pretrained)
        #self.n_feats = model.last_linear.in_features
        self.l0 = nn.Linear(2048, 168)
        self.l1 = nn.Linear(2048, 11)
        self.l2 = nn.Linear(2048, 7)

    def forward(self, x):
        bs, _, _, _= x.shape
        x = self.model.features(x)
        x = F.adaptive_avg_pool2d(x, 1).reshape(bs, -1)
        l0 = self.l0(x)
        l1 = self.l1(x)
        l2 = self.l2(x)
        return l0, l1, l2

def get_model(arch: str, pretrained: str = "imagenet"):
    
    model = ResModel(arch, pretrained=pretrained)
    model_grey = model

    # Sum over the weights to convert the kernel
    weight_grey = model.model.layer0.conv1.weight.sum(dim=1, keepdim=True)
    bias = model_grey.model.layer0.conv1.bias

    # Instantiate a new convolution module and set weights
    model_grey.model.layer0.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    model_grey.model.layer0.conv1.weight = torch.nn.Parameter(weight_grey)
    model_grey.model.layer0.conv1.bias = bias

    #model_grey.layer0.relu1 = Mish()
    model_grey

    return model_grey

model_name = "se_resnext50_32x4d"   
model = get_model(arch=model_name, pretrained=None) #.eval()