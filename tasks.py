import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import json
import numpy as np
import random
import os
# import cv2
import copy
# import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader,Dataset
from models import mnist_model,fashionmnist_model,emnist_model,cifar_model,cifar100_model,a9a_model,mimic_model,agnews_model,imagenet_model,wav2vec_model
from partymodel import ServerParty,ClientParty
from src.torchmimic.data import IHMDataset
from src.torchmimic.utils import pad_colalte
from torchtext.datasets import AG_NEWS
from torch.utils.data.dataset import random_split
from torchtext.data.functional import to_map_style_dataset
from transformers import DistilBertTokenizer,BertTokenizer,Wav2Vec2FeatureExtractor
from MLclf import MLclf
from models.wav2vec_model import TestModel
from torch.nn.utils.rnn import pad_sequence
from datasets import load_dataset, Audio
device = torch.device("cpu")

class A9ADataset(Dataset):
    def __init__(self,data,labels):     
        self.data = data
        self.labels = labels
        self.size = data.shape[0]

    def __getitem__(self, index):
        return self.data[index],self.labels[index]

    def __len__(self):
        return self.size
    
def set_seed(seed=0):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True
    #torch.use_deterministic_algorithms(True)

def load_a9a_data(path):
    data = []
    labels = []
    file = open(path,'r')
    file_data = file.readlines()
    for row in file_data:
        tmp_list = row.split(' ')
        labels.append(1 if int(tmp_list[0])==1 else 0)
        one_row = [0]*123
        for val in tmp_list[1:-1]:
            one_row[int(val.split(':')[0])-1] = 1
        data.append(one_row)
    
    data = torch.Tensor(data)
    labels = torch.Tensor(labels).long()
    return data,labels

def get_task_data(task_name,id=0,is_asyn=True,gpu=-1,use_concat=False,estimation=False,search=False):
    global device
    device = torch.device('cuda:{}'.format(gpu) if gpu!=-1 and torch.cuda.is_available() else 'cpu')
    set_seed()
    
    if task_name == 'mnist':
        return get_mnist_task_data(id,is_asyn,use_concat,estimation,search)
    elif task_name == 'fashionmnist':
        return get_fashionmnist_task_data(id,is_asyn,use_concat,estimation,search)
    elif task_name == 'emnist':
        return get_emnist_task_data(id,is_asyn,use_concat,estimation,search)
    elif task_name == 'cifar':
        return get_cifar_task_data(id,is_asyn,use_concat,estimation,search)
    elif task_name == 'cifar100':
        return get_cifar100_task_data(id,is_asyn,use_concat,estimation,search)
    elif task_name == 'a9a':
        return get_a9a_task_data(id,is_asyn,use_concat,estimation,search)
    elif task_name == 'agnews':
        return get_agnews_task_data(id,is_asyn,False,estimation,search)
    elif task_name == 'imagenet':
        return get_imagenet_task_data(id,is_asyn,True,estimation,search)
    elif task_name == 'mimic':
        return get_mimic_task_data(id,is_asyn,use_concat,estimation,search)
    elif task_name == 'audiomnist':
        return get_audiomnist_task_data(id,is_asyn,True,estimation,search)
    return -1

def reset_ps_party(task_name,id=0,lr=0.001):
    task_info = json.load(open('task_info.json','r',encoding='utf-8'))[task_name]
    n_local = task_info['n_local']
    div = task_info['div']

    if task_name == 'mnist':
        if id == 0:
            model = mnist_model.ServerNet().to(device)
            loss_func = nn.CrossEntropyLoss().to(device)
            optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
            party = ServerParty(model=model,loss_func=loss_func,optimizer=optimizer,n_iter=n_local[id])
        else:
            model = mnist_model.ClientNet().to(device)
            optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
            party = ClientParty(model=model,optimizer=optimizer,n_iter=n_local[id])
    if task_name == 'fashionmnist':
        if id == 0:
            model = fashionmnist_model.ServerNet().to(device)
            loss_func = nn.CrossEntropyLoss().to(device)
            optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
            party = ServerParty(model=model,loss_func=loss_func,optimizer=optimizer,n_iter=n_local[id])
        else:
            model = fashionmnist_model.ClientNet().to(device)
            optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
            party = ClientParty(model=model,optimizer=optimizer,n_iter=n_local[id])
    if task_name == 'emnist':
        if id == 0:
            model = emnist_model.ServerNet().to(device)
            loss_func = nn.CrossEntropyLoss().to(device)
            optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
            party = ServerParty(model=model,loss_func=loss_func,optimizer=optimizer,n_iter=n_local[id])
        else:
            model = emnist_model.ClientNet().to(device)
            optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
            party = ClientParty(model=model,optimizer=optimizer,n_iter=n_local[id])
    elif task_name == 'cifar':
        if id == 0:
            model = cifar_model.ServerNet().to(device)
            loss_func = nn.CrossEntropyLoss().to(device)
            optimizer = optim.SGD(model.parameters(), lr=lr)
            party = ServerParty(model=model,loss_func=loss_func,optimizer=optimizer,n_iter=n_local[id])
        else:
            model = cifar_model.ClientNet().to(device)
            optimizer = optim.SGD(model.parameters(), lr=lr)
            party = ClientParty(model=model,optimizer=optimizer,n_iter=n_local[id])
    elif task_name == 'cifar100':
        if id == 0:
            model = cifar100_model.ServerNet().to(device)
            loss_func = nn.CrossEntropyLoss().to(device)
            optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
            party = ServerParty(model=model,loss_func=loss_func,optimizer=optimizer,n_iter=n_local[id])
        else:
            model = cifar100_model.ClientNet().to(device)
            optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
            party = ClientParty(model=model,optimizer=optimizer,n_iter=n_local[id])
    elif task_name == 'a9a':
        if id == 0:
            model = a9a_model.ServerNet().to(device)
            loss_func = nn.CrossEntropyLoss().to(device)
            optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
            party = ServerParty(model=model,loss_func=loss_func,optimizer=optimizer,n_iter=n_local[id])
        else:
            model = a9a_model.ClientNet(div[id]-div[id-1]).to(device)
            optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
            party = ClientParty(model=model,optimizer=optimizer,n_iter=n_local[id])
    elif task_name == 'mimic':
        if id == 0:
            model = mimic_model.ServerNet(num_layers=2).to(device)
            loss_func = nn.BCELoss().to(device)
            optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
            party = ServerParty(model=model,loss_func=loss_func,optimizer=optimizer,n_iter=n_local[id])
        else:
            model = mimic_model.ClientNet(48*(div[id]-div[id-1])).to(device)
            optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
            party = ClientParty(model=model, optimizer=optimizer, n_iter=n_local[id])
    
    return party

def get_mnist_task_data(id,is_asyn,use_concat,estimation,search):
    task_info = json.load(open('task_info.json','r',encoding='utf-8'))['mnist']
    data_dir = task_info['data_dir']
    client_num = task_info['client_num']
    n_local = task_info['n_local']
    bounds = task_info['bounds'] if is_asyn else [0]*(client_num+1)
    train_batch_size = task_info['train_batch_size']
    test_batch_size = task_info['test_batch_size']
    epochs = task_info['epochs']
    div = task_info['div']
    lr = task_info['lr']
    delta_T = task_info['delta_T']
    CT = task_info['CT']
    Tw = task_info['Tw']
    c0 = task_info['c0']
    estimation_D = task_info['estimation_D']
    search_CT = task_info['search_CT']

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    g = torch.Generator()

    task_data = []


    if id == 0:
        train_dataset = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(data_dir, train=False, transform=transform)
        train_loader = DataLoader(train_dataset,batch_size=train_batch_size,drop_last=True,shuffle=True,generator=g)
        test_loader = DataLoader(test_dataset,batch_size=test_batch_size,drop_last=True)
        model = mnist_model.ServerNet().to(device)
        loss_func = nn.CrossEntropyLoss().to(device)
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        party = ServerParty(model=model,loss_func=loss_func,optimizer=optimizer,n_iter=n_local[id],use_concat=use_concat)

        task_data.append(party)
        task_data.append(train_loader)
        task_data.append(test_loader)
        task_data.append(epochs)
        task_data.append(bounds[id])
        task_data.append(lr)
        task_data.append(delta_T)
        task_data.append(CT)

        if estimation:
            task_data.append(Tw)
            task_data.append(c0)
            task_data.append(estimation_D)
        elif search:
            task_data.append(search_CT)
            task_data.append(c0)
    else:
        train_dataset = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(data_dir, train=False, transform=transform)
        train_dataset.data = train_dataset.data[:,:,div[id-1]:div[id]]
        test_dataset.data = test_dataset.data[:,:,div[id-1]:div[id]]
        train_loader = DataLoader(train_dataset,batch_size=train_batch_size,drop_last=True,shuffle=True,generator=g)
        test_loader = DataLoader(test_dataset,batch_size=test_batch_size,drop_last=True)
        model = mnist_model.ClientNet().to(device)
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        party = ClientParty(model=model,optimizer=optimizer,n_iter=n_local[id])

        task_data.append(party)
        task_data.append(train_loader)
        task_data.append(test_loader)
        task_data.append(epochs)
        task_data.append(bounds[id])

        if estimation:
            task_data.append(Tw)

    return task_data

def get_fashionmnist_task_data(id,is_asyn,use_concat,estimation,search):
    task_info = json.load(open('task_info.json','r',encoding='utf-8'))['fashionmnist']
    data_dir = task_info['data_dir']
    client_num = task_info['client_num']
    n_local = task_info['n_local']
    bounds = task_info['bounds'] if is_asyn else [0]*(client_num+1)
    train_batch_size = task_info['train_batch_size']
    test_batch_size = task_info['test_batch_size']
    epochs = task_info['epochs']
    div = task_info['div']
    lr = task_info['lr']
    delta_T = task_info['delta_T']
    CT = task_info['CT']
    Tw = task_info['Tw']
    c0 = task_info['c0']
    estimation_D = task_info['estimation_D']
    search_CT = task_info['search_CT']

    transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize((0.1307,), (0.3081,)),
    ])

    g = torch.Generator()

    task_data = []


    if id == 0:
        train_dataset = datasets.FashionMNIST(data_dir, train=True, download=True, transform=transform)
        test_dataset = datasets.FashionMNIST(data_dir, train=False, transform=transform)
        train_loader = DataLoader(train_dataset,batch_size=train_batch_size,drop_last=True,shuffle=True,generator=g)
        test_loader = DataLoader(test_dataset,batch_size=test_batch_size,drop_last=True)
        model = fashionmnist_model.ServerNet().to(device)
        loss_func = nn.CrossEntropyLoss().to(device)
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        party = ServerParty(model=model,loss_func=loss_func,optimizer=optimizer,n_iter=n_local[id],use_concat=use_concat)

        task_data.append(party)
        task_data.append(train_loader)
        task_data.append(test_loader)
        task_data.append(epochs)
        task_data.append(bounds[id])
        task_data.append(lr)
        task_data.append(delta_T)
        task_data.append(CT)

        if estimation:
            task_data.append(Tw)
            task_data.append(c0)
            task_data.append(estimation_D)
        elif search:
            task_data.append(search_CT)
            task_data.append(c0)
    else:
        train_dataset = datasets.FashionMNIST(data_dir, train=True, download=True, transform=transform)
        test_dataset = datasets.FashionMNIST(data_dir, train=False, transform=transform)
        train_dataset.data = train_dataset.data[:,:,div[id-1][0]:div[id-1][1]]
        test_dataset.data = test_dataset.data[:,:,div[id-1][0]:div[id-1][1]]

        train_loader = DataLoader(train_dataset,batch_size=train_batch_size,drop_last=True,shuffle=True,generator=g)
        test_loader = DataLoader(test_dataset,batch_size=test_batch_size,drop_last=True)
        model = fashionmnist_model.ClientNet().to(device)
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        party = ClientParty(model=model,optimizer=optimizer,n_iter=n_local[id])

        task_data.append(party)
        task_data.append(train_loader)
        task_data.append(test_loader)
        task_data.append(epochs)
        task_data.append(bounds[id])

        if estimation:
            task_data.append(Tw)

    return task_data

# def get_emnist_task_data(id,is_asyn,use_concat,estimation,search):
#     task_info = json.load(open('task_info.json','r',encoding='utf-8'))['emnist']
#     data_dir = task_info['data_dir']
#     client_num = task_info['client_num']
#     n_local = task_info['n_local']
#     bounds = task_info['bounds'] if is_asyn else [0]*(client_num+1)
#     train_batch_size = task_info['train_batch_size']
#     test_batch_size = task_info['test_batch_size']
#     epochs = task_info['epochs']
#     div = task_info['div']
#     lr = task_info['lr']
#     delta_T = task_info['delta_T']
#     CT = task_info['CT']
#     Tw = task_info['Tw']
#     c0 = task_info['c0']
#     estimation_D = task_info['estimation_D']
#     search_CT = task_info['search_CT']

#     transform = transforms.Compose([
#         transforms.ToTensor(),
#         # transforms.Normalize((0.1307,), (0.3081,)),
#     ])

#     g = torch.Generator()

#     task_data = []

#     if id == 0:
#         train_dataset = datasets.EMNIST(data_dir, train=True, download=True, transform=transform, split='byclass')
#         test_dataset = datasets.EMNIST(data_dir, train=False, transform=transform, split='byclass')
        
#         train_loader = DataLoader(train_dataset,batch_size=train_batch_size,drop_last=True,shuffle=True,generator=g)
#         test_loader = DataLoader(test_dataset,batch_size=test_batch_size,drop_last=True)
        
#         model = emnist_model.ServerNet().to(device)
#         loss_func = nn.CrossEntropyLoss().to(device)
#         optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
#         party = ServerParty(model=model,loss_func=loss_func,optimizer=optimizer,n_iter=n_local[id],use_concat=use_concat)

#         task_data.append(party)
#         task_data.append(train_loader)
#         task_data.append(test_loader)
#         task_data.append(epochs)
#         task_data.append(bounds[id])
#         task_data.append(lr)
#         task_data.append(delta_T)
#         task_data.append(CT)

#         if estimation:
#             task_data.append(Tw)
#             task_data.append(c0)
#             task_data.append(estimation_D)
#         elif search:
#             task_data.append(search_CT)
#             task_data.append(c0)
#     else:
#         train_dataset = datasets.EMNIST(data_dir, train=True, download=True, transform=transform, split='byclass')
#         test_dataset = datasets.EMNIST(data_dir, train=False, transform=transform, split='byclass')
        
#         train_dataset.data = train_dataset.data[:,:,div[id-1]:div[id]] # [697932, 84, 28]
#         test_dataset.data = test_dataset.data[:,:,div[id-1]:div[id]]

#         train_loader = DataLoader(train_dataset,batch_size=train_batch_size,drop_last=True,shuffle=True,generator=g)
#         test_loader = DataLoader(test_dataset,batch_size=test_batch_size,drop_last=True)

#         model = emnist_model.ClientNet().to(device)
#         optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
#         party = ClientParty(model=model,optimizer=optimizer,n_iter=n_local[id])

#         task_data.append(party)
#         task_data.append(train_loader)
#         task_data.append(test_loader)
#         task_data.append(epochs)
#         task_data.append(bounds[id])

#         if estimation:
#             task_data.append(Tw)

#     return task_data

# padding过的emnist
def get_emnist_task_data(id,is_asyn,use_concat,estimation,search):
    task_info = json.load(open('task_info.json','r',encoding='utf-8'))['emnist']
    data_dir = task_info['data_dir']
    client_num = task_info['client_num']
    n_local = task_info['n_local']
    bounds = task_info['bounds'] if is_asyn else [0]*(client_num+1)
    train_batch_size = task_info['train_batch_size']
    test_batch_size = task_info['test_batch_size']
    epochs = task_info['epochs']
    div = task_info['div']
    lr = task_info['lr']
    delta_T = task_info['delta_T']
    CT = task_info['CT']
    Tw = task_info['Tw']
    c0 = task_info['c0']
    estimation_D = task_info['estimation_D']
    search_CT = task_info['search_CT']

    transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize((0.1307,), (0.3081,)),
    ])

    g = torch.Generator()

    task_data = []

    if id == 0:
        train_dataset = datasets.EMNIST(data_dir, train=True, download=True, transform=transform, split='byclass')
        test_dataset = datasets.EMNIST(data_dir, train=False, transform=transform, split='byclass')

        train_dataset.data = torch.from_numpy(np.pad(train_dataset.data, ((0,0),(28,28),(28,28)), mode='constant', constant_values=0))
        test_dataset.data = torch.from_numpy(np.pad(test_dataset.data, ((0,0),(28,28),(28,28)), mode='constant', constant_values=0))
        
        train_loader = DataLoader(train_dataset,batch_size=train_batch_size,drop_last=True,shuffle=True,generator=g)
        test_loader = DataLoader(test_dataset,batch_size=test_batch_size,drop_last=True)

        model = emnist_model.ServerNet().to(device)
        loss_func = nn.CrossEntropyLoss().to(device)
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        party = ServerParty(model=model,loss_func=loss_func,optimizer=optimizer,n_iter=n_local[id],use_concat=use_concat)


        task_data.append(party)
        task_data.append(train_loader)
        task_data.append(test_loader)
        task_data.append(epochs)
        task_data.append(bounds[id])
        task_data.append(lr)
        task_data.append(delta_T)
        task_data.append(CT)

        if estimation:
            task_data.append(Tw)
            task_data.append(c0)
            task_data.append(estimation_D)
        elif search:
            task_data.append(search_CT)
            task_data.append(c0)
    else:
        train_dataset = datasets.EMNIST(data_dir, train=True, download=True, transform=transform, split='byclass')
        test_dataset = datasets.EMNIST(data_dir, train=False, transform=transform, split='byclass')

        train_dataset.data = torch.from_numpy(np.pad(train_dataset.data, ((0,0),(28,28),(28,28)), mode='constant', constant_values=0))[:,:,div[id-1]:div[id]] # [697932, 84, 28]
        test_dataset.data = torch.from_numpy(np.pad(test_dataset.data, ((0,0),(28,28),(28,28)), mode='constant', constant_values=0))[:,:,div[id-1]:div[id]]
        
        # train_dataset.data = train_dataset.data[:,:,div[id-1]:div[id]] # [697932, 84, 28]
        # test_dataset.data = test_dataset.data[:,:,div[id-1]:div[id]]

        train_loader = DataLoader(train_dataset,batch_size=train_batch_size,drop_last=True,shuffle=True,generator=g)
        test_loader = DataLoader(test_dataset,batch_size=test_batch_size,drop_last=True)

        model = emnist_model.ClientNet().to(device)
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        party = ClientParty(model=model,optimizer=optimizer,n_iter=n_local[id])

        task_data.append(party)
        task_data.append(train_loader)
        task_data.append(test_loader)
        task_data.append(epochs)
        task_data.append(bounds[id])

        if estimation:
            task_data.append(Tw)

    return task_data

def get_cifar_task_data(id,is_asyn,use_concat,estimation,search):
    task_info = json.load(open('task_info.json','r',encoding='utf-8'))['cifar']
    data_dir = task_info['data_dir']
    client_num = task_info['client_num']
    n_local = task_info['n_local']
    bounds = task_info['bounds'] if is_asyn else [0]*(client_num+1)
    train_batch_size = task_info['train_batch_size']
    test_batch_size = task_info['test_batch_size']
    epochs = task_info['epochs']
    div = task_info['div']
    lr = task_info['lr']
    delta_T = task_info['delta_T']
    CT = task_info['CT']
    Tw = task_info['Tw']
    c0 = task_info['c0']
    estimation_D = task_info['estimation_D']
    search_CT = task_info['search_CT']

    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    g = torch.Generator()

    task_data = []


    if id == 0:
        train_dataset = datasets.CIFAR10(data_dir, train=True, download=True, transform=transform_train)
        test_dataset = datasets.CIFAR10(data_dir, train=False, transform=transform_test)

        # train_dataset.data = np.pad(train_dataset.data, ((0,0),(32,32),(32,32),(0,0)), mode='constant', constant_values=0)
        # test_dataset.data = np.pad(test_dataset.data, ((0,0),(32,32),(32,32),(0,0)), mode='constant', constant_values=0)

        train_loader = DataLoader(train_dataset,batch_size=train_batch_size,drop_last=True,shuffle=True,generator=g)
        test_loader = DataLoader(test_dataset,batch_size=test_batch_size,drop_last=True)
        model = cifar_model.ServerNet().to(device)
        loss_func = nn.CrossEntropyLoss().to(device)
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        party = ServerParty(model=model,loss_func=loss_func,optimizer=optimizer,n_iter=n_local[id],use_concat=use_concat)
        
        task_data.append(party)
        task_data.append(train_loader)
        task_data.append(test_loader)
        task_data.append(epochs)
        task_data.append(bounds[id])
        task_data.append(lr)
        task_data.append(delta_T)
        task_data.append(CT)

        if estimation:
            task_data.append(Tw)
            task_data.append(c0)
            task_data.append(estimation_D)
        elif search:
            task_data.append(search_CT)
            task_data.append(c0)
    else:
        train_dataset = datasets.CIFAR10(data_dir, train=True, download=True, transform=transform_train)
        test_dataset = datasets.CIFAR10(data_dir, train=False, transform=transform_test)

        # padding = transforms.Pad(padding=32, fill=0)
        # train_dataset.data = np.pad(train_dataset.data, ((0,0),(32,32),(32,32),(0,0)), mode='constant', constant_values=0)
        # test_dataset.data = np.pad(test_dataset.data, ((0,0),(32,32),(32,32),(0,0)), mode='constant', constant_values=0)

        train_dataset.data = train_dataset.data[:,:,div[id-1][0]:div[id-1][1]]
        test_dataset.data = test_dataset.data[:,:,div[id-1][0]:div[id-1][1]]

        train_loader = DataLoader(train_dataset,batch_size=train_batch_size,drop_last=True,shuffle=True,generator=g)
        test_loader = DataLoader(test_dataset,batch_size=test_batch_size,drop_last=True)
        # model = cifar_model.ClientNet(n_dim=32*3*(div[id]-div[id-1])).to(device) # n_dim
        model = cifar_model.ClientNet().to(device)
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        party = ClientParty(model=model,optimizer=optimizer,n_iter=n_local[id])

        task_data.append(party)
        task_data.append(train_loader)
        task_data.append(test_loader)
        task_data.append(epochs)
        task_data.append(bounds[id])

        if estimation:
            task_data.append(Tw)

    return task_data

def get_cifar100_task_data(id,is_asyn,use_concat,estimation,search):
    task_info = json.load(open('task_info.json','r',encoding='utf-8'))['cifar100']
    data_dir = task_info['data_dir']
    client_num = task_info['client_num']
    n_local = task_info['n_local']
    bounds = task_info['bounds'] if is_asyn else [0]*(client_num+1)
    train_batch_size = task_info['train_batch_size']
    test_batch_size = task_info['test_batch_size']
    epochs = task_info['epochs']
    div = task_info['div']
    lr = task_info['lr']
    delta_T = task_info['delta_T']
    CT = task_info['CT']
    Tw = task_info['Tw']
    c0 = task_info['c0']
    estimation_D = task_info['estimation_D']
    search_CT = task_info['search_CT']

    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    g = torch.Generator()

    task_data = []


    if id == 0:
        train_dataset = datasets.CIFAR10(data_dir, train=True, download=True, transform=transform_train)
        # train_dataset = list(datasets.CIFAR100(data_dir, train=True, download=True, transform=transform_train))[:-256]
        # val_dataset = list(datasets.CIFAR100(data_dir, train=True, download=True, transform=transform_train))[-256:]
        test_dataset = datasets.CIFAR100(data_dir, train=False, transform=transform_test)
        train_loader = DataLoader(train_dataset,batch_size=train_batch_size,drop_last=True,shuffle=True,generator=g)
        # val_loader = DataLoader(val_dataset,batch_size=train_batch_size,drop_last=True,shuffle=True,generator=torch.Generator())
        test_loader = DataLoader(test_dataset,batch_size=test_batch_size,drop_last=True)
        model = cifar100_model.ServerNet().to(device)
        loss_func = nn.CrossEntropyLoss().to(device)
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        # party = ServerParty(model=model,loss_func=loss_func,optimizer=optimizer,val_loader=val_loader,n_iter=n_local[id],use_concat=use_concat)
        party = ServerParty(model=model,loss_func=loss_func,optimizer=optimizer,n_iter=n_local[id],use_concat=use_concat)
        
        task_data.append(party)
        task_data.append(train_loader)
        # task_data.append(val_loader)
        task_data.append(test_loader)
        task_data.append(epochs)
        task_data.append(bounds[id])
        task_data.append(lr)
        task_data.append(delta_T)
        task_data.append(CT)

        if estimation:
            task_data.append(Tw)
            task_data.append(c0)
            task_data.append(estimation_D)
        elif search:
            task_data.append(search_CT)
            task_data.append(c0)
    else:
        train_dataset = datasets.CIFAR100(data_dir, train=True, download=True, transform=transform_train)
        test_dataset = datasets.CIFAR100(data_dir, train=False, transform=transform_test)
        train_dataset.data = train_dataset.data[:,:,div[id-1]:div[id]]
        test_dataset.data = test_dataset.data[:,:,div[id-1]:div[id]]

        train_dataset = list(train_dataset)[:-256]
        # val_dataset = list(train_dataset)[45000:]

        train_loader = DataLoader(train_dataset,batch_size=train_batch_size,drop_last=True,shuffle=True,generator=g)
        # val_loader = DataLoader(val_dataset,batch_size=train_batch_size,drop_last=True,shuffle=True,generator=g)
        test_loader = DataLoader(test_dataset,batch_size=test_batch_size,drop_last=True)
        # model = cifar_model.ClientNet(n_dim=32*3*(div[id]-div[id-1])).to(device) # n_dim
        model = cifar100_model.ClientNet().to(device)
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        party = ClientParty(model=model,optimizer=optimizer,n_iter=n_local[id])

        task_data.append(party)
        task_data.append(train_loader)
        task_data.append(test_loader)
        task_data.append(epochs)
        task_data.append(bounds[id])

        if estimation:
            task_data.append(Tw)

    return task_data

def get_a9a_task_data(id,is_asyn,use_concat,estimation,search):
    task_info = json.load(open('task_info.json','r',encoding='utf-8'))['a9a']
    data_dir = task_info['data_dir']
    client_num = task_info['client_num']
    n_local = task_info['n_local']
    bounds = task_info['bounds'] if is_asyn else [0]*(client_num+1)
    train_batch_size = task_info['train_batch_size']
    test_batch_size = task_info['test_batch_size']
    epochs = task_info['epochs']
    div = task_info['div']
    lr = task_info['lr']
    delta_T = task_info['delta_T']
    CT = task_info['CT']
    Tw = task_info['Tw']
    c0 = task_info['c0']
    estimation_D = task_info['estimation_D']
    search_CT = task_info['search_CT']

    train_data,train_labels = load_a9a_data(data_dir+'/train.txt')
    test_data,test_labels = load_a9a_data(data_dir+'/test.txt')

    g = torch.Generator()

    task_data = []


    if id == 0:
        train_dataset = A9ADataset(train_data,train_labels)
        test_dataset = A9ADataset(test_data,test_labels)
        train_loader = DataLoader(train_dataset,train_batch_size,drop_last=True,shuffle=True,generator=g)
        test_loader = DataLoader(test_dataset,test_batch_size,drop_last=True)
        model = a9a_model.ServerNet().to(device)
        loss_func = nn.CrossEntropyLoss().to(device)
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        party = ServerParty(model=model,loss_func=loss_func,optimizer=optimizer,n_iter=n_local[id],use_concat=use_concat)
        
        task_data.append(party)
        task_data.append(train_loader)
        task_data.append(test_loader)
        task_data.append(epochs)
        task_data.append(bounds[id])
        task_data.append(lr)
        task_data.append(delta_T)
        task_data.append(CT)

        if estimation:
            task_data.append(Tw)
            task_data.append(c0)
            task_data.append(estimation_D)
        elif search:
            task_data.append(search_CT)
            task_data.append(c0)
    else:
        train_dataset = A9ADataset(train_data[:,div[id-1]:div[id]],train_labels)
        test_dataset = A9ADataset(test_data[:,div[id-1]:div[id]],test_labels)
        train_loader = DataLoader(train_dataset,train_batch_size,drop_last=True,shuffle=True,generator=g)
        test_loader = DataLoader(test_dataset,test_batch_size,drop_last=True)
        model = a9a_model.ClientNet(div[id]-div[id-1]).to(device)
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        party = ClientParty(model=model,optimizer=optimizer,n_iter=n_local[id])

        task_data.append(party)
        task_data.append(train_loader)
        task_data.append(test_loader)
        task_data.append(epochs)
        task_data.append(bounds[id])
    
        if estimation:
            task_data.append(Tw)

    return task_data


def get_mimic_task_data(id,is_asyn,use_concat,estimation,search):
    task_info = json.load(open('task_info.json', 'r', encoding='utf-8'))['mimic']
    data_dir = task_info['data_dir']
    client_num = task_info['client_num']
    n_local = task_info['n_local']
    bounds = task_info['bounds'] if is_asyn else [0]*(client_num+1)
    train_batch_size = task_info['train_batch_size']
    test_batch_size = task_info['test_batch_size']
    epochs = task_info['epochs']
    div = task_info['div']
    lr = task_info['lr']
    delta_T = task_info['delta_T']
    CT = task_info['CT']
    Tw = task_info['Tw']
    c0 = task_info['c0']
    estimation_D = task_info['estimation_D']
    search_CT = task_info['search_CT']

    train_dataset = IHMDataset(data_dir, train=True, n_samples=None)
    test_dataset = IHMDataset(data_dir, train=False, n_samples=None)
    # train_dataset.labels = train_dataset.labels[:, None]
    # test_dataset.labels = test_dataset.labels[:, None]
    train_dataset.labels = torch.zeros(len(train_dataset), 2).scatter_(1, train_dataset.labels[:, None].to(torch.int64), 1)
    test_dataset.labels = torch.zeros(len(test_dataset), 2).scatter_(1, test_dataset.labels[:, None].to(torch.int64), 1)
    kwargs = {"num_workers": 0, "pin_memory": True} if device else {}

    g = torch.Generator()

    task_data = []

    if id == 0:
        train_loader = DataLoader(train_dataset, batch_size=train_batch_size,shuffle=True,drop_last=True,generator=g,collate_fn=pad_colalte,**kwargs)
        test_loader = DataLoader(test_dataset, batch_size=test_batch_size,shuffle=False,drop_last=True,collate_fn=pad_colalte,**kwargs)
        model = mimic_model.ServerNet(num_layers=2).to(device)
        loss_func = nn.BCELoss().to(device)
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        party = ServerParty(model=model, loss_func=loss_func, optimizer=optimizer, n_iter=n_local[id],use_concat=use_concat)

        task_data.append(party)
        task_data.append(train_loader)
        task_data.append(test_loader)
        task_data.append(epochs)
        task_data.append(bounds[id])
        task_data.append(lr)
        task_data.append(delta_T)
        task_data.append(CT)

        if estimation:
            task_data.append(Tw)
            task_data.append(c0)
            task_data.append(estimation_D)
        elif search:
            task_data.append(search_CT)
            task_data.append(c0)
    else:
        train_dataset.data = [t[:, div[id - 1]:div[id]] for t in train_dataset.data]
        test_dataset.data = [t[:, div[id - 1]:div[id]] for t in test_dataset.data]
        train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True,drop_last=True,generator=g,collate_fn=pad_colalte,**kwargs)
        test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False,drop_last=True,collate_fn=pad_colalte,**kwargs)
        model = mimic_model.ClientNet(48*(div[id]-div[id-1])).to(device)
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        party = ClientParty(model=model, optimizer=optimizer, n_iter=n_local[id])

        task_data.append(party)
        task_data.append(train_loader)
        task_data.append(test_loader)
        task_data.append(epochs)
        task_data.append(bounds[id])

        if estimation:
            task_data.append(Tw)

    return task_data

def get_agnews_task_data(id,is_asyn,use_concat,estimation,search):
    task_info = json.load(open('task_info.json','r',encoding='utf-8'))['agnews']
    data_dir = task_info['data_dir']
    client_num = task_info['client_num']
    n_local = task_info['n_local']
    bounds = task_info['bounds'] if is_asyn else [0]*(client_num+1)
    train_batch_size = task_info['train_batch_size']
    test_batch_size = task_info['test_batch_size']
    epochs = task_info['epochs']
    div = task_info['div']
    lr = task_info['lr']
    delta_T = task_info['delta_T']
    CT = task_info['CT']
    Tw = task_info['Tw']
    c0 = task_info['c0']
    estimation_D = task_info['estimation_D']
    search_CT = task_info['search_CT']

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    text_pipeline = lambda x: tokenizer(
                            x,                      
                            add_special_tokens = True,  # 添加special tokens， 也就是CLS和SEP
                            max_length = 50,           # 设定最大文本长度
                            padding = 'max_length',   # pad到最大的长度  
                            return_tensors = 'pt',       # 返回的类型为pytorch tensor
                            truncation = True
                    )
    label_pipeline = lambda x: int(x) - 1

    def collate_batch(batch):
        label_list, text_list = [], []
        for _label, _text in batch:
            label_list.append(label_pipeline(_label))
            text_list.append(text_pipeline(_text))
        label_list = torch.tensor(label_list, dtype=torch.int64)
        text_list = torch.cat(
            [torch.cat([text['input_ids'] for text in text_list]).unsqueeze(1),
             torch.cat([text['attention_mask'] for text in text_list]).unsqueeze(1)],
             dim=1
        )

        return text_list, label_list
    
    g = torch.Generator()

    task_data = []


    if id == 0:
        train_iter, test_iter = AG_NEWS(root=data_dir)
        train_dataset = to_map_style_dataset(train_iter)
        test_dataset = to_map_style_dataset(test_iter)
        test_dataset, _ = random_split(test_dataset, [0.05, 0.95], generator=g)

        train_loader = DataLoader(train_dataset, batch_size=train_batch_size, drop_last=True, shuffle=True, collate_fn=collate_batch,generator=g)
        test_loader = DataLoader(test_dataset, batch_size=test_batch_size, drop_last=True, collate_fn=collate_batch)

        model = agnews_model.ServerNet(party_num=client_num if use_concat else 1,mask_dim=50,output_dim=4).to(device)
        loss_func = nn.CrossEntropyLoss().to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        party = ServerParty(model=model,loss_func=loss_func,optimizer=optimizer,n_iter=n_local[id],use_concat=use_concat)
        
        task_data.append(party)
        task_data.append(train_loader)
        task_data.append(test_loader)
        task_data.append(epochs)
        task_data.append(bounds[id])
        task_data.append(lr)
        task_data.append(delta_T)
        task_data.append(CT)

        if estimation:
            task_data.append(Tw)
            task_data.append(c0)
            task_data.append(estimation_D)
        elif search:
            task_data.append(search_CT)
            task_data.append(c0)
    else:
        train_iter, test_iter = AG_NEWS(root=data_dir)
        train_iter = [(label,text[int(len(text) * div[id-1]):int(len(text) * div[id])]) for label,text in train_iter]
        test_iter = [(label,text[int(len(text) * div[id-1]):int(len(text) * div[id])]) for label,text in test_iter]
        train_dataset = to_map_style_dataset(train_iter)
        test_dataset = to_map_style_dataset(test_iter)
        test_dataset, _ = random_split(test_dataset, [0.05, 0.95], generator=g)

        train_loader = DataLoader(train_dataset, batch_size=train_batch_size, drop_last=True, shuffle=True, collate_fn=collate_batch,generator=g)
        test_loader = DataLoader(test_dataset, batch_size=test_batch_size, drop_last=True, collate_fn=collate_batch)

        model = agnews_model.ClientNet().to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        party = ClientParty(model=model,optimizer=optimizer,n_iter=n_local[id])

        task_data.append(party)
        task_data.append(train_loader)
        task_data.append(test_loader)
        task_data.append(epochs)
        task_data.append(bounds[id])

        if estimation:
            task_data.append(Tw)

    return task_data


def get_imagenet_task_data(id,is_asyn,use_concat,estimation,search):
    task_info = json.load(open('task_info.json','r',encoding='utf-8'))['imagenet']
    data_dir = task_info['data_dir']
    client_num = task_info['client_num']
    n_local = task_info['n_local']
    bounds = task_info['bounds'] if is_asyn else [0]*(client_num+1)
    train_batch_size = task_info['train_batch_size']
    test_batch_size = task_info['test_batch_size']
    epochs = task_info['epochs']
    div = task_info['div']
    lr = task_info['lr']
    delta_T = task_info['delta_T']
    CT = task_info['CT']
    Tw = task_info['Tw']
    c0 = task_info['c0']
    estimation_D = task_info['estimation_D']
    search_CT = task_info['search_CT']

    class ImageNetDataset(Dataset):
        def __init__(self,data,labels):     
            self.data = data
            self.labels = labels
            self.size = data.shape[0]

        def __getitem__(self, index):
            return self.data[index],self.labels[index]

        def __len__(self):
            return self.size

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    
    g = torch.Generator()

    task_data = []


    if id == 0:
        train_dataset, validation_dataset, test_dataset = MLclf.miniimagenet_clf_dataset(data_dir=data_dir,ratio_train=0.6, ratio_val=0.2, seed_value=None, shuffle=True, transform=transform, save_clf_data=True)

        train_loader = DataLoader(train_dataset,batch_size=train_batch_size,drop_last=True,shuffle=True,generator=g,num_workers=2)
        test_loader = DataLoader(test_dataset,batch_size=test_batch_size,drop_last=True,num_workers=2)
        model = imagenet_model.ServerNet().to(device)
        loss_func = nn.CrossEntropyLoss().to(device)
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        party = ServerParty(model=model,loss_func=loss_func,optimizer=optimizer,n_iter=n_local[id],use_concat=use_concat,evaluate_func='top5')
        
        task_data.append(party)
        task_data.append(train_loader)
        task_data.append(test_loader)
        task_data.append(epochs)
        task_data.append(bounds[id])
        task_data.append(lr)
        task_data.append(delta_T)
        task_data.append(CT)

        if estimation:
            task_data.append(Tw)
            task_data.append(c0)
            task_data.append(estimation_D)
        elif search:
            task_data.append(search_CT)
            task_data.append(c0)
    else:
        train_dataset, validation_dataset, test_dataset = MLclf.miniimagenet_clf_dataset(data_dir=data_dir,ratio_train=0.6, ratio_val=0.2, seed_value=None, shuffle=True, transform=transform, save_clf_data=True)

        train_dataset = ImageNetDataset(train_dataset.tensors[0][:,:,div[id][0]:div[id][1]],train_dataset.tensors[1])
        test_dataset = ImageNetDataset(test_dataset.tensors[0][:,:,div[id][0]:div[id][1]],test_dataset.tensors[1])

        train_loader = DataLoader(train_dataset,batch_size=train_batch_size,drop_last=True,shuffle=True,generator=g,num_workers=2)
        test_loader = DataLoader(test_dataset,batch_size=test_batch_size,drop_last=True,num_workers=2)
        model = imagenet_model.ClientNet().to(device)
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        party = ClientParty(model=model,optimizer=optimizer,n_iter=n_local[id])

        task_data.append(party)
        task_data.append(train_loader)
        task_data.append(test_loader)
        task_data.append(epochs)
        task_data.append(bounds[id])

        if estimation:
            task_data.append(Tw)

    return task_data

def get_audiomnist_task_data(id,is_asyn,use_concat,estimation,search):
    task_info = json.load(open('task_info.json','r',encoding='utf-8'))['audiomnist']
    data_dir = task_info['data_dir']
    client_num = task_info['client_num']
    n_local = task_info['n_local']
    bounds = task_info['bounds'] if is_asyn else [0]*(client_num+1)
    train_batch_size = task_info['train_batch_size']
    test_batch_size = task_info['test_batch_size']
    epochs = task_info['epochs']
    div = task_info['div']
    lr = task_info['lr']
    delta_T = task_info['delta_T']
    CT = task_info['CT']
    Tw = task_info['Tw']
    c0 = task_info['c0']
    estimation_D = task_info['estimation_D']
    search_CT = task_info['search_CT']
    model_path = task_info['model_path']

    dataset = load_dataset(data_dir)
    def set_label(example):
        example['label'] = example['digit']  # digit
        return example
    dataset = dataset.map(set_label, num_proc=10)    # 设置label列
    num_labels = len(set(dataset['train']['label'])) # 获取标签类数 用于加载模型

    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_path)
    # 数据预处理 主要是从audio中提取特征
    def preprocess_function(examples):
        audio_arrays = [x["array"] for x in examples["audio"]]
        inputs = feature_extractor(
            audio_arrays, sampling_rate=16000, padding=True, truncation=True, max_length=16000
        )
        inputs['label'] = examples['label']
        return inputs
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000)) # 重采样音频到 16kHz
    dataset['train'] = dataset['train'].map(preprocess_function, batched=True, num_proc=10) # 提取特征
    dataset['test'] = dataset['test'].map(preprocess_function, batched=True, num_proc=10) # 提取特征
    dataset.set_format(type="torch", columns=["input_values", "label"]) # 将数据集转换为 PyTorch 格式
    
    g = torch.Generator()

    task_data = []


    if id == 0:
        # 填充输入 ... 就抽象 前面eature_extractor明明有截断和填充
        def custom_collate_fn(batch):
            input_values = [item['input_values'].squeeze(0) for item in batch]  # 去掉多余的维度
            labels = torch.tensor([item['label'] for item in batch])
            # 对 input_values 进行填充
            input_values = pad_sequence(input_values, batch_first=True, padding_value=0.0)
            padded_tensor = torch.zeros(input_values.shape[0],16000)
            padded_tensor[:, :input_values.shape[1]] = input_values
            input_values = padded_tensor
            return input_values,labels
        train_loader = DataLoader(dataset["train"], batch_size=train_batch_size, drop_last=True, shuffle=True, collate_fn=custom_collate_fn, generator=g) # 创建 DataLoader
        test_loader = DataLoader(dataset["test"], batch_size=test_batch_size, drop_last=True, collate_fn=custom_collate_fn) # 创建 DataLoader

        model = wav2vec_model.ServerNet(model_path,num_labels).to(device)
        loss_func = nn.CrossEntropyLoss().to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        party = ServerParty(model=model,loss_func=loss_func,optimizer=optimizer,n_iter=n_local[id],use_concat=use_concat)
        
        task_data.append(party)
        task_data.append(train_loader)
        task_data.append(test_loader)
        task_data.append(epochs)
        task_data.append(bounds[id])
        task_data.append(lr)
        task_data.append(delta_T)
        task_data.append(CT)

        if estimation:
            task_data.append(Tw)
            task_data.append(c0)
            task_data.append(estimation_D)
        elif search:
            task_data.append(search_CT)
            task_data.append(c0)
    else:
        def custom_collate_fn(batch):
            input_values = [item['input_values'].squeeze(0) for item in batch]  # 去掉多余的维度
            labels = torch.tensor([item['label'] for item in batch])
            # 对 input_values 进行填充
            input_values = pad_sequence(input_values, batch_first=True, padding_value=0.0)
            padded_tensor = torch.zeros(input_values.shape[0],16000)
            padded_tensor[:, :input_values.shape[1]] = input_values
            input_values = padded_tensor
            input_values = input_values[:,int(input_values.shape[1] * div[id-1]):int(input_values.shape[1] * div[id])]
            return input_values,labels
        train_loader = DataLoader(dataset["train"], batch_size=train_batch_size, drop_last=True, shuffle=True, collate_fn=custom_collate_fn, generator=g) # 创建 DataLoader
        test_loader = DataLoader(dataset["test"], batch_size=test_batch_size, drop_last=True, collate_fn=custom_collate_fn) # 创建 DataLoader

        model = wav2vec_model.ClientNet(model_path,num_labels).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        party = ClientParty(model=model,optimizer=optimizer,n_iter=n_local[id])

        task_data.append(party)
        task_data.append(train_loader)
        task_data.append(test_loader)
        task_data.append(epochs)
        task_data.append(bounds[id])

        if estimation:
            task_data.append(Tw)

    return task_data    