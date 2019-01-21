import math
from torch import nn
from torch.autograd import Function
import torch
import numpy as np
import group_points_cuda
import time

def test():
    #B*C*M
    pts = torch.FloatTensor(128,3,128).normal_().to('cuda:0')
    #B*M*K
    indices = []
    for i in range(128*128*16):
        indices.append(np.random.choice(128))
    indices = np.array(indices).reshape(128,128,16)
    indices = torch.LongTensor(indices)
    
    indices = indices.to('cuda:0')
    #print(pts[0,:,:].transpose(0,1))
    #print(indices[0,:,:])
    print(torch.sum(indices[0,:,:]==1))
    
    st = time.time()
    x = group_points_cuda.forward(pts, indices)
    print(x.shape)
    print(x[1,:,12,:].transpose(0,1), time.time()-st)
    st = time.time()
    y = torch.stack([pts[k, :, index] for k, index in enumerate(torch.unbind(indices.long(), dim=0))], dim=0)
    print(y.shape)
    print(y[1,:,12,:].transpose(0,1), time.time()-st)
    print(torch.sum(x==y))
    grads = torch.ones(128,3,128,16).to('cuda:0')
    grad_x = group_points_cuda.backward(grads, indices, 128)
    print(grad_x[0,:,:])
    
    

test()