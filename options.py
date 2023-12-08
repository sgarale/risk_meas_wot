import numpy as np
from scipy.stats import norm
import torch
from torch.distributions.multivariate_normal import MultivariateNormal



class MaxCall:
    def __init__(self, K) -> None:
        self.K = K

    def f(self, x):
        if x.ndim == 1:
            return torch.maximum(torch.max(x) - self.K, torch.tensor(0.))
        elif x.ndim == 2:
            return torch.maximum(torch.max(x, dim=1).values - self.K, torch.tensor(0.)).reshape(-1, 1)
        elif x.ndim > 2:
            return torch.maximum(torch.max(x, dim=-1).values - self.K, torch.tensor(0.))
        else:
            return ValueError("Inconsistent input dimension")
    


class MaxBull:
    def __init__(self, K1, K2) -> None:
        self.long_call = MaxCall(K1)
        self.short_call = MaxCall(K2)

    def f(self, x):
        return self.long_call.f(x) - self.short_call.f(x)



class BasketCall:
    def __init__(self, K) -> None:
        self.K = K

    def f(self, x):
        if x.ndim == 1:
            return torch.maximum(torch.mean(x) - self.K, torch.tensor(0.))
        elif x.ndim == 2:
            return torch.maximum(torch.mean(x, dim=1) - self.K, torch.tensor(0.)).reshape(-1, 1)
        elif x.ndim > 2:
            return torch.maximum(torch.mean(x, dim=-1) - self.K, torch.tensor(0.))
        else:
            return ValueError("Inconsistent input dimension")
        

    
class GeometricPut:
    def __init__(self, K) -> None:
        self.K = K

    def f(self, x):
        if x.ndim == 1:
            return torch.maximum(self.K - torch.pow(torch.prod(x), 1. / x.shape[0]), torch.tensor(0.))
        elif x.ndim == 2:
            return torch.maximum(self.K - torch.pow(torch.prod(x, dim=1), 1. / x.shape[1]), torch.tensor(0.)).reshape(-1, 1)
        elif x.ndim > 2:
            return torch.maximum(self.K - torch.pow(torch.prod(x, dim=-1), 1. / x.shape[2]), torch.tensor(0.))
        else:
            return ValueError("Inconsistent input dimension")
        


class MinPut:
    def __init__(self, K) -> None:
        self.K = K

    def f(self, x):
        if x.ndim == 1:
            return torch.maximum(self.K - torch.min(x), torch.tensor(0.))
        elif x.ndim == 2:
            return torch.maximum(self.K - torch.min(x, dim=1).values, torch.tensor(0.)).reshape(-1, 1)
        elif x.ndim > 2:
            return torch.maximum(self.K - torch.min(x, dim=-1).values, torch.tensor(0.))
        else:
            return ValueError("Inconsistent input dimension")