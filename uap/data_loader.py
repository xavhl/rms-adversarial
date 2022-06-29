import torch
import torchvision
import torchvision.transforms as transforms
import os

device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device(device)

def load_data():
    transform = transforms.Compose([
        transforms.ToTensor(),  # Transform to tensor
        transforms.Normalize((0.5,), (0.5,))  # Min-max scaling to [-1, 1]
    ])

    data_dir = os.path.join('/raid/home/yhyeung2/rms', 'imagenet')
    print('Data stored in %s' % data_dir)
    trainset = torchvision.datasets.ImageNet(root=data_dir, split='train', download=True, transform=transform)
    testset = torchvision.datasets.ImageNet(root=data_dir, split='val', download=True, transform=transform)

    # classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal',
    #            'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=8, pin_memory=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=5, shuffle=False, num_workers=8, pin_memory=True)
    return trainloader,testloader

