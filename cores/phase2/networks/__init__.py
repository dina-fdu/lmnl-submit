import torch
from pretrainedmodels import models

from torch import nn
from torch.nn import DataParallel
import torch.backends.cudnn as cudnn

from networks.wideresnet import WideResNet
from .resnet import ResNet34
from .preact_resnet import PreActResNet18

import os

def get_model(conf, num_class=10, data_parallel=True, devices=None): #None means using all devices
    name = conf['type']

    if name == 'wresnet40_2':
        model = WideResNet(40, 2, dropout_rate=0.0, num_classes=num_class)
    elif name == 'wresnet28_2':
        model = WideResNet(28, 2, dropout_rate=0.0, num_classes=num_class)
    elif name == 'wresnet28_10':
        model = WideResNet(28, 10, dropout_rate=0.0, num_classes=num_class)
    elif name == 'resnet34':
        model = ResNet34(num_class)   
        if conf['noise_type'] in ['clean', 'worst', 'aggre', 'rand1', 'rand2', 'rand3', 'clean100', 'noisy100']:
            noise_type_map = {'clean': 'clean_label', 'worst': 'worse_label', 'aggre': 'aggre_label',
                          'rand1': 'random_label1', 'rand2': 'random_label2', 'rand3': 'random_label3',
                          'clean100': 'clean_label', 'noisy100': 'noisy_label'}
            conf['noise_type'] = noise_type_map[conf['noise_type']]
        model_path = f"../../dividemix/dividemix/{conf['dataset']}_{conf['noise_type']}{conf['lambda_u']}best.pth.tar"
        state_dict = torch.load(model_path, map_location = "cpu")
        model.load_state_dict(state_dict['state_dict'])
        print(f'model loaded from {model_path}')

    elif name == 'preres18':
        model = PreActResNet18(num_class)
    else:
        raise NameError('no model named, %s' % name)

    if data_parallel:
        print('DEBUG: torch device count', torch.cuda.device_count())
        model = model.cuda()
        model = DataParallel(model, device_ids=devices)
    else:
        import horovod.torch as hvd
        device = torch.device('cuda', hvd.local_rank())
        model = model.to(device)
    cudnn.benchmark = True
    return model


def num_class(dataset):
    return {
        'cifar10': 10,
        'reduced_cifar10': 10,
        'cifar100': 100,
        'imagenet': 1000,
        'reduced_imagenet': 120,
    }[dataset]
