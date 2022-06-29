# from locale import currency
# import matplotlib.pyplot as plt
import numpy as np
import os
# import sys
import torch
import logging
from datetime import datetime

# sys.path.append(os.path.realpath('..'))

from attacks import uap_sgd
from utils import loader_imgnet, model_imgnet, evaluate

def print_log(logger, message_):
    message = f'{datetime.now()} | '
    message += message_
    print(message)
    logger.info(message)

def generate(logger, model, fooling_rate):
    global current_epoch

    loader = loader_imgnet(current_epoch, dir_data, 2000, 100, 224)
    
    # clean accuracy
    top, top_probs, top1acc, top5acc, outputs, labels = evaluate(model, loader)
    accuracy = sum(outputs == labels) / len(labels)
    # print_log(logger, 'Epoch: {:2d}\ntop = {}, top_probs = {}, top1acc = {}, top5acc = {} accuracy = {}'.format(
        # current_epoch, top, top_probs, top1acc, top5acc, accuracy)
    print_log(logger, 'Epoch: {:2d} | accuracy = {}'.format(
        current_epoch, accuracy)
    )
    current_epoch += 1

    epoch_accuracy_worst = [(0, 1)]

    while 1 - fooling_rate < accuracy:
        loader = loader_imgnet(current_epoch, dir_data, 2000, 100, 224) # torch inceptionv3 uses adaptive pooling, not need to be (299, 299)

        uap, losses = uap_sgd(model, loader, nb_epoch, eps, beta, step_decay)

        # evaluate
        top, top_probs, top1acc, top5acc, outputs, labels = evaluate(model, loader, uap = uap)
        accuracy = sum(outputs == labels) / len(labels)
        
        print_log(logger, 
        # 'Epoch: {:2d}\ntop = {}, top_probs = {}, top1acc = {}, top5acc = {} accuracy = {}\nuap: min={:.5f} max={:.5f} mean={:.5f}\n'.format(
            # current_epoch, top, top_probs, top1acc, top5acc, accuracy, torch.min(uap).item(), torch.max(uap).item(), torch.mean(uap).item())
        'Epoch: {:2d} | accuracy = {} | uap: min={:.5f} max={:.5f} mean={:.5f}'.format(
            current_epoch, accuracy, torch.min(uap).item(), torch.max(uap).item(), torch.mean(uap).item())
        ) ### TODO: test

        current_epoch += 1

        # save best perturbation with lowest accuracy
        if accuracy < epoch_accuracy_worst[-1][1]:
            epoch_accuracy_worst.append((current_epoch, accuracy))
            uap_np = uap.detach().cpu().numpy()

            uap_np_name = pert_dir + f'pert_negative_sgd-uap_{model_name}_epoch{epoch_accuracy_worst[-2][0]}.npy'
            np.save(uap_np_name, uap_np)

            uap_np_name_update = uap_np_name.replace(f'_epoch{epoch_accuracy_worst[-2][0]}', f'_epoch{epoch_accuracy_worst[-1][0]}')
            os.rename(src=uap_np_name, dst=uap_np_name_update)
            print_log(logging, 'Saved best perturbation at ' + uap_np_name_update)

    else:
        print_log(logger, f'Perturbation achieved fooling rate {fooling_rate} after {current_epoch*len(loader)} images')

# def get_image_size(model_name):
#     if model_name == 'resnet152':
#         return 224
#     elif model_name == 'inception_v3':
#         return 299
#     else:
#         raise NotImplementedError

dir_data = '../imagenet/val'
current_epoch = 0

nb_epoch = 8
eps = 40 / 255
beta = 12
step_decay = 0.7

fooling_rate = 0.8
model_list = ['resnet152']#, 'inception_v3']

log_dir = './logs/'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

pert_dir = './perturbations/'
if not os.path.exists(pert_dir):
    os.makedirs(pert_dir)

for model_name in model_list:
    logging.basicConfig(filename=log_dir + f'log_sgd-uap_{model_name}.log',
                    format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]', level=logging.INFO, filemode='a',
                    datefmt='%Y-%m-%d %I:%M:%S %p')

    current_epoch = 0 # reset epoch count for new model

    # loader = loader_imgnet(dir_data, 2000, 100, 224) # torch inceptionv3 uses adaptive pooling, not need to be (299, 299)
    # print_log(logging, f'Dataloader: {len(loader)*100} images')

    model = model_imgnet(model_name)
    uap = generate(logging, model, fooling_rate)

    # except Keyboard...
    # print_log(logging, 'Keyboard Interrupt: save perturbation and exit.')
    # uap_np_name = pert_dir + f'pert_sgd-uap_{model_name}.npy'
    # np.save(uap_np_name, uap_np)
    # print_log(logging, 'Saved perturbation at', uap_np_name)
    # raise

# visualize UAP
# plt.imshow(np.transpose(((uap / eps) + 1) / 2, (1, 2, 0)))