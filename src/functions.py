import numpy as np
import torch
import torch.nn.functional as F
import time
from typing import Dict


def get_device():
    if torch.cuda.is_available():
        DEVICE = 'cuda'
        # torch.set_default_tensor_type(torch.cuda.FloatTensor)
    elif torch.backends.mps.is_available():
        DEVICE = torch.device("mps")
    else:
        DEVICE = 'cpu'
    print('Using {}'.format(DEVICE))
    return DEVICE

def get_time() -> int:
    '''Returns current time in ms'''
    return int(round(time.time() * 1000))

class Timer:
    def __init__(self):
        self.reset()

    def lap(self):
        self.t_lap = get_time()

    def get(self):
        return get_time() - self.t_lap

    def reset(self):
        self.t_total = get_time()
        self.t_lap = get_time()

    def __str__(self):
        t = self.get()
        ms = t % 1000
        t = int(t / 1000)
        s = t % 60
        t = int(t / 60)
        m = t % 60
        if t == 0:
            return "{}.{:03}".format(s,ms)
        else:
            t = int(t / 60)
            h = t
            if t == 0:
                return "{}:{:02}.{:03}".format(m,s,ms)
            else:
                return "{}:{:02}:{:02}.{:03}".format(h,m,s,ms)

def append_dict(dict_a:Dict[str,np.ndarray], dict_b:Dict[str,np.ndarray]):
    for k, v in dict_b.items():
        dict_a[k] = np.concatenate((dict_a[k], v))

def L1Loss(x:torch.FloatTensor):
    # if x.dim() > 0:
    #     print(len(x[-1]))
    return torch.mean(torch.abs(x))

def L1LossLists(x):
    loss = 0
    divider = 0
    for i, elem in enumerate(x):
        if isinstance(elem, tuple):
            elem = elem[0]
        divider += elem.shape[1]
        if i == 0:
            # print(i, torch.mean(torch.abs(elem[:, :-2])).item())
            loss += torch.sum(torch.abs(elem[:, :-2]))
        else:
            # print(i, torch.mean(torch.abs(elem)).item())
            loss += torch.sum(torch.abs(elem))
    return loss / (divider * elem.shape[0])
    # return(torch.mean(torch.abs(torch.cat([elem.flatten() for elem in x]))))
    # return torch.sum(torch.tensor([torch.mean(torch.abs(elem)) for elem in x]))

def L1LossListsConv(x):
    loss = 0
    divider = 128*128 + 32*64*64 + 32*32*32
    for i, elem in enumerate(x):
        # print(f'{i}: {elem.shape}')
        #TODO change back to i == 0 and elem[:, 0] for conv and not hybrid
        if i == 0:
            # print(i, torch.mean(torch.abs(elem[:, 0])).item())
            # print(i, elem[:,0].shape)
            loss += torch.sum(torch.abs(elem[:, 0]))
        else:
            # print(i, torch.mean(torch.abs(elem)).item())
            # print(i, elem.shape)
            loss += torch.sum(torch.abs(elem))
    # print(loss / (divider * elem.shape[0]))
    return loss / (divider * elem.shape[0])

def L2Loss(x:torch.FloatTensor):
    
    return torch.mean(torch.pow(x, 2))

def Linear(x:torch.FloatTensor):
    return x

def parse_loss(args, terms):
    if args == None:
        return L1Loss, torch.tensor(0.0)
    
    pre, post, weights = terms
    arg1, arg2 = args.split('_')
    if arg1 == 'l1':
        if arg2 == 'all':
            loss_fn = L1LossLists
        elif arg2 == 'allconv':
            loss_fn = L1LossListsConv
        else:
            loss_fn = L1Loss
    else:
        loss_fn = L2Loss
        
    if arg2 == 'pre' or arg2 == 'preconv':
        loss_arg = pre
    elif arg2 == 'all' or arg2 == 'allconv':
        loss_arg = post
    else:
        loss_arg = weights
        
    return loss_fn, loss_arg
            
def init_params(size_x, size_y):
    return torch.normal(mean=torch.zeros((size_x, size_y)), std=torch.ones((size_x, size_y))*2) * (1. / size_x)

def normalize(x, p=2.0, dim=1):
    return F.normalize(x, p, dim)
