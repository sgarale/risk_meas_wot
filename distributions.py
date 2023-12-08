import torch
from torch.distributions.multivariate_normal import MultivariateNormal

# ---------------- distributions ----------------

class MultiLogNormal:
    def __init__(self, loc, covariance_matrix=None, precision_matrix=None, scale_tril=None, validate_args=None):
        self.multinormal = MultivariateNormal(loc, covariance_matrix, precision_matrix, scale_tril, validate_args)

    def sample(self, sample_shape=...):
        return torch.exp(self.multinormal.sample(sample_shape))
    
    def sample_n(self, n):
        return torch.exp(self.multinormal.sample_n(n))