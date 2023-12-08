import math
import numpy as np
from torch import nn
import torch



# ------------- Cost functionals ------------------------

class PowerCost:
    def __init__(self, p, h=1., case='static') -> None:
        self.p = p
        self.h = h
        self.case = case

    def _euclid_norm(self, u):
        return torch.pow(u, 2).sum(dim=-1).sqrt()

    def _cost_static(self, u):
        out = torch.pow(self._euclid_norm(u), self.p)
        return out.reshape(-1, 1)

    def _cost_first(self, u):
        out = self.h * torch.pow(self._euclid_norm(u) / self.h, self.p)
        return out.reshape(-1, 1)

    def _cost_second(self, u):
        out = self.h * torch.pow(self._euclid_norm(u) / math.sqrt(self.h), self.p)
        return out.reshape(-1, 1)
    
    def cost(self, u):
        if self.case == 'static': return self._cost_static(u)
        if self.case == 'first order': return self._cost_first(u)
        if self.case == 'second order': return self._cost_second(u)


# -------------- Neural Networks -------------------------

class FullConnNet(nn.Module):
    """
    Neural Network from R^(input_dim) to R^(output_dim) with activation function activ_func
    """
    def __init__(self, activ_func, input_dim, output_dim, width, depth=1):
        super(FullConnNet, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.width = width
        self.depth = depth
        self.linear_stack = nn.Sequential(
            nn.Linear(input_dim, width)
        )
        for i in range(depth - 1):
            self.linear_stack.append(activ_func())
            self.linear_stack.append(nn.Linear(width, width))
        self.linear_stack.append(activ_func())
        self.linear_stack.append(nn.Linear(width, output_dim))

    def forward(self, x):
        theta = self.linear_stack(x)
        return theta
    

# -------------- Risk measures -------------------------


class RiskMeasure1d(nn.Module):
    def __init__(self, func, cost, mu, activ_func, width, depth, maximize=True) -> None:
        super().__init__()
        self.func = func
        self.cost = cost
        self.mu = mu
        self.unet = FullConnNet(activ_func, 1, 1, width, depth)
        self.mc_err = 0.
        self.mult = 1.
        if maximize: self.mult = -1.

    def ccong(self, y):
        u_y = self.unet(y)
        return self.func(y + u_y) - self.cost(u_y)

    def forward(self, y):
        int_arg = self.ccong(y)
        if not self.training:
            self.mc_err = torch.std(int_arg) / math.sqrt(y.size(dim=0))
        risk_u = self.mult * torch.mean(int_arg)
        return risk_u
    


class RiskMeasure2d(nn.Module):
    def __init__(self, func, cost, mu, activ_func, width, depth, maximize=True) -> None:
        super().__init__()
        self.func = func
        self.cost = cost
        self.mu = mu
        self.unet = FullConnNet(activ_func, 2, 2, width, depth)
        self.mult = 1.
        if maximize: self.mult = -1.
        self.mc_err = None

    def ccong(self, y):
        u_y = self.unet(y)
        return self.func(y[:, 0] + u_y[:, 0], y[:, 1] + u_y[:, 1]) - self.cost(u_y)

    def forward(self, y):
        int_arg = self.ccong(y)
        if not self.training:
            self.mc_err = torch.std(int_arg) / math.sqrt(y.size(dim=0))
            return torch.mean(int_arg)
        return self.mult * torch.mean(int_arg)


    
class MartRiskMeasure1d(nn.Module):
    def __init__(self, func, cost, mu, activ_func, width, depth, maximize=True) -> None:
        super().__init__()
        self.func = func
        self.cost = cost
        self.mu = mu
        self.unet = FullConnNet(activ_func, 1, 2, width, depth)
        self.mult = 1.
        if maximize: self.mult = -1
        self.mc_err = None

    def _u_v_p(self, y):
        full_y = self.unet(y)
        u_y = full_y[:, 0].reshape(-1, 1)
        t_y = full_y[:, 1].reshape(-1, 1)
        p_y = 1. / (1. + torch.exp(- t_y))
        v_y = - torch.exp(t_y) * u_y
        return u_y, v_y, p_y

    def ccong(self, y):
        full_y = self.unet(y)
        u_y = full_y[:, 0].reshape(-1, 1)
        t_y = full_y[:, 1].reshape(-1, 1)
        p_y = 1. / (1. + torch.exp(- t_y))
        v_y = - torch.exp(t_y) * u_y
        return p_y * (self.func(y + u_y) - self.cost(u_y)) + (1 - p_y) * (self.func(y + v_y) - self.cost(v_y))

    def forward(self, y):
        int_arg = self.ccong(y)
        if not self.training:
            self.mc_err = torch.std(int_arg) / math.sqrt(y.size(dim=0))
            return self.mult * torch.mean(int_arg)
        risk_u = self.mult * torch.mean(int_arg)
        return risk_u
    


class MartRiskMeasureMulti(nn.Module):
    def __init__(self, func, cost, mu, activ_func, width, depth, d=1, maximize=True) -> None:
        super().__init__()
        self.func = func
        self.cost = cost
        self.mu = mu
        self.d = d
        self.unet = FullConnNet(activ_func, d, d + 1, width, depth)
        self.mult = 1.
        if maximize: self.mult = -1.
        self.mc_err = None
    
    def ccong(self, y):
        full_y = self.unet(y)
        u_y = full_y[..., 0:self.d]
        t_y = full_y[..., self.d:self.d + 1]
        p_y = 1. / (1. + torch.exp(- t_y))
        v_y = - torch.exp(t_y) * u_y
        return p_y * (self.func(y + u_y) - self.cost(u_y)) + (1 - p_y) * (self.func(y + v_y) - self.cost(v_y))

    def forward(self, y):
        int_arg = self.ccong(y)
        if not self.training:
            self.mc_err = torch.std(int_arg) / math.sqrt(y.size(dim=0))
            return torch.mean(int_arg)
        risk_u = self.mult * torch.mean(int_arg)
        return risk_u