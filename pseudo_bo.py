import torch
import torch.nn as nn
import torch.nn.functional as F
from models import MLP

class PseudoBO(nn.Module):

    def __init__(self, R, x_dim=1, y_dim=1, h_dim=32, lr=0.1, device='cpu'):
        super(PseudoBO, self).__init__()
        self.R = R
        self.device = device

        self.net = MLP(x_dim, y_dim, h_dim)

        self.w_list = self.grad_step()
        self.D = []


    def loss_inner(self, x):
        """
        x.shape = (1, 1)
        NOTE: R(x) \in R^d, d >= 1
        """
        return (self.R(x) - self.net.forward(x)).pow(2).sum()


    def loss_outer(self, x, w_list):
        """
        x.shape = (n, 1)
        """
        return (R(x) - self.net.functional_forward(x, w_list)).pow(2).sum()


    def grad_step(self, lr=0.1):
        w_list = []
        for l in self.net.model.children():
            l_dict = {}
            for k, v in l.named_parameters():
                if v.grad is None:
                    l_dict[k] = v
                else:
                    l_dict[k] = v - lr * v.grad
            w_list.append(l_dict)
        return w_list


    def acquisition(self, n_steps, lr=0.1, debug=False):
        s = torch.rand(1, 1).to(self.device).requires_grad_()
        optimizer = torch.optim.Adam([s], lr=lr, weight_decay=1e-6)

        D_t = torch.cat(self.D, dim=0)

        for k in range(n_steps):
            self.net.update_w(self.w_list) # w(D_t) = w_list

            self.net.zero_grad()
            l_inn = self.loss_inner(s) # L(s, w(D_t))
            l_inn.backward() # w(D_t).grad

            self.w_list = self.grad_step(lr) # w(s, D_t) = w(D_t) - lr * w(D_t).grad

            optimizer.zero_grad() # zero out s.grad
            l_out = self.loss_outer(D_t, self.w_list) # L(D_t, w(s, D_t))
            optimizer.step() # s = s - lr * s.grad

            if debug:
                print('Step %d: s = %.3f,  inn = %.3f,  out = %.3f' % (k, s, l_inn, l_out))

        self.D.append(s)




