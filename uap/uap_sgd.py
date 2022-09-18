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

def generate(model_dir, logger, model, target_acc, eps):
    global current_epoch

    # torch inceptionv3 uses adaptive pooling, not need to be (299, 299)
    loader = loader_imgnet(current_epoch, dir_data, nb_images, batch_size, img_size=224)

    current_epoch += 1
    uap, losses = uap_sgd(model, loader, nb_epoch, eps, beta, step_decay)
    top, top_probs, top1acc, top5acc, outputs, labels = evaluate(model, loader_test, uap = uap)
    accuracy = sum(outputs == labels) / len(labels)
    message = 'Epoch: {:2d} | accuracy = {} | uap: min={:.5f} max={:.5f} mean={:.5f}'.format(
        current_epoch, accuracy, torch.min(uap).item(), torch.max(uap).item(), torch.mean(uap).item())
    print_log(logger, message)
    uap_np = uap.detach().cpu().numpy()
    uap_np_name = model_dir + f'epoch{current_epoch}.npy'
    np.save(uap_np_name, uap_np)
    print_log(logger, 'Saved perturbation at ' + uap_np_name)

    current_epoch += 1
    # print(type(uap), uap.requires_grad)

    epoch_accuracy_worst = [(0, 1)]

    while current_epoch < term_epoch+1:
        
        uap, losses = uap_sgd(model, loader, nb_epoch, eps, beta, step_decay, uap_init=uap.detach().clone())

        # evaluate
        top, top_probs, top1acc, top5acc, outputs, labels = evaluate(model, loader_test, uap = uap)
        accuracy = sum(outputs == labels) / len(labels)
        
        message = 'Epoch: {:2d} | accuracy = {} | uap: min={:.5f} max={:.5f} mean={:.5f}'.format(
            current_epoch, accuracy, torch.min(uap).item(), torch.max(uap).item(), torch.mean(uap).item())
        print_log(logger, message)
        # 'Epoch: {:2d}\ntop = {}, top_probs = {}, top1acc = {}, top5acc = {} accuracy = {}\nuap: min={:.5f} max={:.5f} mean={:.5f}\n'.format(
            # current_epoch, top, top_probs, top1acc, top5acc, accuracy, torch.min(uap).item(), torch.max(uap).item(), torch.mean(uap).item())
         ### TODO: test

        current_epoch += 1

        # save best perturbation with lowest accuracy
        if accuracy < epoch_accuracy_worst[-1][1]:
            epoch_accuracy_worst.append((current_epoch, accuracy))
            uap_np = uap.detach().cpu().numpy()

            uap_np_name = model_dir + f'epoch{epoch_accuracy_worst[-2][0]}.npy'
            np.save(uap_np_name, uap_np)

            uap_np_name_update = uap_np_name.replace(f'_epoch{epoch_accuracy_worst[-2][0]}', f'_epoch{epoch_accuracy_worst[-1][0]}')
            os.rename(src=uap_np_name, dst=uap_np_name_update)
            print_log(logger, 'Saved best perturbation at ' + uap_np_name_update)

    else:
        if accuracy < target_acc:
            print_log(logger, f'Perturbation achieved fooling rate {target_acc} after {current_epoch*len(loader)} images')

# def get_image_size(model_name):
#     if model_name == 'resnet152':
#         return 224
#     elif model_name == 'inception_v3':
#         return 299
#     else:
#         raise NotImplementedError

nb_images = 50000
nb_epoch = 1
eps_list = [i for i in range(10, 50, 10)]

dir_data = '../imagenet/val'
current_epoch = 0
term_epoch = 10 # 50000 / nb_images

batch_size = 64
beta = 12
step_decay = 0.7

target_acc = 0.5
model_list = ['resnet152', 'inception_v3', 'resnext101_32x8d']

# log_dir = './logs/'
# if not os.path.exists(log_dir):
#     os.makedirs(log_dir)

# pert_dir = './perturbations/20220829/' # normal range [eps, -eps]
# nb_epoch = 8
# nb_images = 2000

pert_dir = './perturbations/20220829/'
if not os.path.exists(pert_dir):
    os.makedirs(pert_dir)

gpu_id = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = gpu_id
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
print(f'Using GPU {gpu_id}')

for model_name in model_list:
    model = model_imgnet(model_name)

    # clean accuracy
    loader_test = loader_imgnet(current_epoch, dir_data, 50000, batch_size, 224)
    top, top_probs, top1acc, top5acc, outputs, labels = evaluate(model, loader_test)
    accuracy = sum(outputs == labels) / len(labels)
    # print_log(logger, 'Epoch: {:2d}\ntop = {}, top_probs = {}, top1acc = {}, top5acc = {} accuracy = {}'.format(
        # current_epoch, top, top_probs, top1acc, top5acc, accuracy)
    print_log(logging, 'Epoch: {:2d} | accuracy = {}'.format(
        current_epoch, accuracy)
    )
    
    for eps in eps_list:
        model_dir = pert_dir + f'{model_name}_eps{eps}/'
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        print(model_dir)

        logging.basicConfig(filename=model_dir + f'hist.log',
                        format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]', level=logging.INFO, filemode='a',
                        datefmt='%Y-%m-%d %I:%M:%S %p')

        current_epoch = 0 # reset epoch count for new model

        # loader = loader_imgnet(dir_data, 2000, 100, 224) # torch inceptionv3 uses adaptive pooling, not need to be (299, 299)
        # print_log(logger, f'Dataloader: {len(loader)*100} images')
        
        generate(model_dir, logging, model, target_acc, eps / 255)

        # except Keyboard...
        # print_log(logger, 'Keyboard Interrupt: save perturbation and exit.')
        # uap_np_name = pert_dir + f'pert_sgd-uap_{model_name}.npy'
        # np.save(uap_np_name, uap_np)
        # print_log(logger, 'Saved perturbation at', uap_np_name)
        # raise

# visualize UAP
# plt.imshow(np.transpose(((uap / eps) + 1) / 2, (1, 2, 0)))