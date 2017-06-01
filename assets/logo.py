import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models
import torchvision.transforms as transforms

from matplotlib import pyplot as plt
from skimage.transform import resize
from skimage.filters import gaussian

import numpy as np

from PIL import Image

# source of image: https://www.flickr.com/photos/rod_waddington/18095645666/in/photolist-Sejx2b-bskntp-bz17wQ-jg8His-bMFqEa-byLQE3-bxERy5-2oYMf5-bMUJXK-bsjNVP-brXp27-2oUshi-2oYGBN-9mumG9-nTvDZP-nRDYEw-gRxLFx-zjxdBn-9mvHab-9msNmn-2oYFHb-c8cYyY-gRxH6L-c8cZWo-a7pmSu-tz3V1N-9hMiM2-pUe2Pk-c8E4ZS-dtyeSR-c8cXYJ-MPFsG9-9mrZMF-gRxNjH-gRxHLd-r7aJ9U-a7pmww-gRyD9p-tMexGR-tLXpGG-2oUxwX-tM8fsH-sQdSDj-zFAurz-GoxHjf-egh4iB-eggUTt-bGPqRX-bGPoVz
inp = 'kob.jpg'
outp = 'logo.jpg'
resnet = 34

cmaps = [('Perceptually Uniform Sequential', [
            'viridis', 'plasma', 'inferno', 'magma']),
         ('Sequential', [
            'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
            'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
            'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']),
         ('Sequential2', [
            'binary', 'gist_yarg', 'gist_gray', 'gray', 'bone', 'pink',
            'spring', 'summer', 'autumn', 'winter', 'cool', 'Wistia',
            'hot', 'afmhot', 'gist_heat', 'copper']),
         ('Diverging', [
            'PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu',
            'RdYlBu', 'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic']),
         ('Qualitative', [
            'Pastel1', 'Pastel2', 'Paired', 'Accent',
            'Dark2', 'Set1', 'Set2', 'Set3']),
         ('Miscellaneous', [
            'flag', 'prism', 'ocean', 'gist_earth', 'terrain', 'gist_stern',
            'gnuplot', 'gnuplot2', 'CMRmap', 'cubehelix', 'brg', 'hsv',
            'gist_rainbow', 'rainbow', 'jet', 'nipy_spectral', 'gist_ncar'])]


def get_model(resnet):
    return eval("models.resnet{}".format(resnet))


def basic_block(layer, relu=False):
    def forward(x):
        residual = x

        out = layer.conv1(x)
        out = layer.bn1(out)
        out = layer.relu(out)

        out = layer.conv2(out)
        out = layer.bn2(out)

        if layer.downsample is not None:
            residual = layer.downsample(x)

        out += residual
        if relu:
            out = layer.relu(out)

        return out
    return forward


class FeatureModel(nn.Module):
    def __init__(self, resnet):
        super(FeatureModel, self).__init__()
        self.resnet = resnet
        self.fn = get_model(self.resnet)(pretrained=True)

        # Turn off inplace
        for p in self.fn.modules():
            if "ReLU" in p.__repr__():
                p.inplace = False

    def forward(self, x, request=["layer4_2", "fc"]):
        model = self.fn

        ret = []

        layers = [
            (model.conv1, 'conv1'),
            (model.bn1, 'bn1'),
            (model.relu, 'relu'),
            (model.maxpool, 'maxpool'),
            (model.layer1, 'layer1'),
            (model.layer2, 'layer2'),
            (model.layer3, 'layer3'),
        ]

        if self.resnet == 34:
            layers += [
                (model.layer4[0], 'layer4_0_relu'),
                (model.layer4[1], 'layer4_1_relu'),
                (basic_block(model.layer4[2]), 'layer4_2'),
                (lambda x: model.layer4[2].relu(x), 'layer4_2_relu'),
            ]
        else:
            raise NotImplementedError()

        layers += [
            (model.avgpool, 'avgpool'),
            (lambda x: x.view(x.size(0), -1), 'avgpool_512'),
            (model.fc, 'fc'),
        ]

        for module, name in layers:
            x = module(x)
            # print(" N", x.data.numel())
            # print("<0", (x.data < 0.).sum())
            # print("=0", (x.data == 0.).sum())
            # print("{} - {}".format(name, tuple(x.size())))
            if name in request:
                ret.append(x)

        return ret


def run():
    image = Image.open(inp)
    image_numpy = np.asarray(Image.open(inp))

    _transforms = transforms.Compose([
                       transforms.Scale(227),
                       transforms.CenterCrop(227),
                       transforms.ToTensor(),
                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                       ])

    t_inp = _transforms(image).unsqueeze(0)
    assert t_inp.size() == (1, 3, 227, 227)

    model = FeatureModel(resnet)

    layer4_2 = model(Variable(t_inp))[0]

    att = F.sigmoid(layer4_2.mean(1)).squeeze()

    to_plot = gaussian(resize(att.data.numpy(), image_numpy.shape[:2]), sigma=5)

    plt.imshow(image_numpy)
    plt.imshow(to_plot, alpha=0.3)
    plt.savefig(outp, dpi=80)

    for cmap_category, cmap_list in cmaps:
        for cm in cmap_list:
            fig = plt.figure()

            fig.set_size_inches(1, 1)
            ax = plt.Axes(fig, [0., 0., 1., 1.])
            ax.set_axis_off()
            fig.add_axes(ax)

            plt.set_cmap(cm)
            plt.imshow(to_plot)
            plt.savefig(outp + '.' + cmap_category + '.' + cm + '.jpg', dpi=80)
            plt.close(fig)

if __name__ == '__main__':
    run()
