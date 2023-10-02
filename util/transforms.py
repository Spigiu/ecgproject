import torch
import numpy as np
from importlib import import_module
from torchvision.transforms import Compose


def get_transform(opt):
    t_list = []
    if not opt.keys():
        return torch.nn.Identity()
    for t in opt.keys():
        t_type = getattr(import_module('util.transforms'), t)
        t_list.append(t_type(*opt[t]))
    return Compose(t_list)


class RandomAmplitudeFlip(torch.nn.Module):
    def __init__(self, p=0.5, f=1):
        super(RandomAmplitudeFlip, self).__init__()
        self.p = p
        self.f = f

    def forward(self, data):
        if torch.rand(1) < self.p:
            n_ts = len(data)
            selection = np.random.choice(n_ts, self.f, replace=False)
            data_to_flip = data[selection, :]
            data[selection, :] = -data_to_flip
            return data
        return data
    
    
class RandomScaling(torch.nn.Module):
    def __init__ (self,p=0.5,lb=0.8,hb=1.2, f=1):
        super(RandomScaling, self).__init__()  
        self.p=p
        self.lb=lb
        self.hb=hb
        self.f=f
    
    def forward(self,data):
        if torch.rand(1) < self.p:
            n_ts=len(data)
            selection=np.random.choice(n_ts,self.f,replace=False)
            data_to_amplify= data[selection, :]
            factor= (self.hb - self.lb) * np.random.random_sample() + self.lb
            data[selection,:]= factor * data_to_amplify
            return data 
        return data 
        