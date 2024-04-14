import torch.nn as nn

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def make_layers(cfg_dic):
    layers = []
    in_channels = 3
    for x in cfg_dic:
        if x == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                       nn.BatchNorm2d(x),
                       nn.ReLU(inplace=True)]
            in_channels = x
    layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
    return nn.Sequential(*layers)


class VGG(nn.Module):
    def __init__(self, vgg_name):
        super(VGG, self).__init__()
        self.features = make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, 10)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out


class VGG11(VGG):
    def __init__(self):
        super(VGG11, self).__init__('VGG11')


class VGG13(VGG):
    def __init__(self):
        super(VGG13, self).__init__('VGG13')


class VGG16(VGG):
    def __init__(self):
        super(VGG16, self).__init__('VGG16')


class VGG19(VGG):
    def __init__(self):
        super(VGG19, self).__init__('VGG19')


class TinyImageNetVGG11(VGG):
    def __init__(self):
        super(TinyImageNetVGG11, self).__init__('VGG11')
        self.classifier = nn.Linear(512, 200)


class TinyImageNetVGG13(VGG):
    def __init__(self):
        super(TinyImageNetVGG13, self).__init__('VGG13')
        self.classifier = nn.Linear(512, 200)


class TinyImageNetVGG16(VGG):
    def __init__(self):
        super(TinyImageNetVGG16, self).__init__('VGG16')
        self.classifier = nn.Linear(512, 200)


class TinyImageNetVGG19(VGG):
    def __init__(self):
        super(TinyImageNetVGG19, self).__init__('VGG19')
        self.classifier = nn.Linear(512, 200)


class Cifar100NetVGG11(VGG):
    def __init__(self):
        super(Cifar100NetVGG11, self).__init__('VGG11')
        self.classifier = nn.Linear(512, 100)


class Cifar100NetVGG13(VGG):
    def __init__(self):
        super(Cifar100NetVGG13, self).__init__('VGG13')
        self.classifier = nn.Linear(512, 100)


class Cifar100NetVGG16(VGG):
    def __init__(self):
        super(Cifar100NetVGG16, self).__init__('VGG16')
        self.classifier = nn.Linear(512, 100)


class Cifar100NetVGG19(VGG):
    def __init__(self):
        super(Cifar100NetVGG19, self).__init__('VGG19')
        self.classifier = nn.Linear(512, 100)
