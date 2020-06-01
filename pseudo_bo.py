import torch
import torch.nn as nn
import torch.nn.functional as F
from models import MLP
from math import pi


class PseudoBO(nn.Module):
    def __init__(self, R, pretrained_net=None, x_dim=1, y_dim=1, h_dim=32, lr=0.1, bs=8, device='cpu'):
        super(PseudoBO, self).__init__()
        self.R = R
        self.device = device
        self.inn_lr = lr
        self.bs = bs

        self.net = MLP(x_dim, y_dim, h_dim) if pretrained_net is None else pretrained_net
        self.net.to(device)

        # init w_list
        self.inn_grad_step()

        # init D_0
        self.D = []
        s = torch.zeros(1, 1).to(self.device)
        self.D.append(s)


    def loss_inner(self, x):
        """
        x.shape = (1, 1)
        NOTE: R(x) \in R^d, d >= 1
        """
        return (self.R(x) - self.net.forward(x)).pow(2).sum()


    def loss_outer(self, x):
        """
        x.shape = (n, 1)
        """
        return (self.R(x) - self.forward(x)).pow(2).sum()


    def forward(self, x):
        return self.net.functional_forward(x, self.w_list)


    def inn_grad_step(self):
        self.w_list = []
        for l in self.net.model.children():
            l_dict = {}
            for k, v in l.named_parameters():
                if v.grad is None:
                    l_dict[k] = v # for init w_list
                else:
                    l_dict[k] = v - self.inn_lr * v.grad
            self.w_list.append(l_dict)


    def acquisition(self, n_steps, lr=0.1, debug=False):
        # TODO:
        # [x] 1) batch s;
        # 2) sample s_0 at high loss;
        # [x] 3) pretraining on D_0 (better fitted)
        s = torch.FloatTensor(self.bs, 1).uniform_(-pi, pi).to(self.device).requires_grad_()

        optimizer = torch.optim.Adam([s], lr=lr, weight_decay=1e-3)

        D_t = torch.cat(self.D, dim=0)

        for k in range(n_steps):
            self.net.update_w(self.w_list) # w(D_t) = w_list to sync

            self.net.zero_grad()
            l_inn = self.loss_inner(s) # L(s, w(D_t))
            l_inn.backward(retain_graph=True) # w(D_t).grad
            self.inn_grad_step() # w(s, D_t) = w(D_t) - inn_lr * w(D_t).grad

            optimizer.zero_grad() # zero out s.grad
            l_out = self.loss_outer(D_t) # L(D_t, w(s, D_t))
            l_out.backward()
            optimizer.step() # s = s - lr * s.grad

            if debug:
                print('Step %d: s = %.3f,  inn = %.3f,  out = %.3f' % (k, s, l_inn, l_out))

        self.D.append(s)




