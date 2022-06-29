import numpy as np
import os
import torch
import torchvision.transforms.functional as TransF
from datetime import datetime
import scipy.io

from utils import loader_imgnet, model_imgnet, evaluate

def load_uap(path, negative=False):
    if path[-4:] == '.npy':
        pert = np.load(path)[0]
    elif path[-4:] == '.mat':
        pert = scipy.io.loadmat(path)['r']

    if negative:
        pert = np.where(pert <= 0, pert, 0)

    return TransF.to_tensor(pert/255).to(torch.float32)

dir_data = '../imagenet/val'

model_list = ['inception_v3', 'resnet152']
pert_path = {'inception_v3': 'universal.npy', 'resnet152': 'ResNet-152.mat'}

for model_name in model_list:

    loader = loader_imgnet(None, dir_data, 50000, 100, 224) # torch inceptionv3 uses adaptive pooling, not need to be (299, 299)

    model = model_imgnet(model_name)
    # uap = load_uap('perturbations/pre_computed/' + pert_path[model_name], True)
    uap = None
    top, top_probs, top1acc, top5acc, outputs, labels = evaluate(model, loader, uap = uap)
    accuracy = sum(outputs == labels) / len(labels)

    print(model_name)
    # print('Accuracy = {} | uap: min={:.8f} max={:.8f} mean={:.8f}'.format(
        # accuracy, torch.min(uap).item(), torch.max(uap).item(), torch.mean(uap).item()))
    print(f'Accuracy = {accuracy}')

# visualize UAP
# plt.imshow(np.transpose(((uap / eps) + 1) / 2, (1, 2, 0)))