import torch
import torch.nn as nn
from pipeline.utils import gaussian_prob_density


class EuclideanGMM(nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction


    def forward(self, w1, mu1, sigma1, w2, mu2, sigma2):
        density_1 = EuclideanGMM.gaussian_cross_product(mu1, sigma1, mu1, sigma1)
        density_2 = EuclideanGMM.gaussian_cross_product(mu2, sigma2, mu2, sigma2)
        density_x = EuclideanGMM.gaussian_cross_product(mu1, sigma1, mu2, sigma2)
        assert not torch.any(torch.logical_or(torch.isnan(density_1), torch.isinf(density_1)))
        assert not torch.any(torch.logical_or(torch.isnan(density_2), torch.isinf(density_2)))
        assert not torch.any(torch.logical_or(torch.isnan(density_x), torch.isinf(density_x)))

        w1_w1_cross = w1.unsqueeze(-1) @ w1.unsqueeze(-2)
        w2_w2_cross = w2.unsqueeze(-1) @ w2.unsqueeze(-2)
        w1_w2_cross = w1.unsqueeze(-1) @ w2.unsqueeze(-2)

        L2 = w1_w1_cross * density_1 + w2_w2_cross * density_2 - 2 * w1_w2_cross * density_x
        L2 = torch.sum(L2, dim=[-2, -1])
        assert torch.all(L2 >= 0)

        if self.reduction == 'sum':
            L2 = torch.sum(L2)
        elif self.reduction == 'mean':
            L2 = torch.mean(L2)
        return L2
    

    @staticmethod
    def gaussian_cross_product(mu1, sigma1, mu2, sigma2):
        n_gauss_1 = mu1.shape[-2]
        n_gauss_2 = mu2.shape[-2]
        batch_size = mu1.shape[:-2]
        mu1 = mu1.unsqueeze(-2).expand(*batch_size, -1, n_gauss_2, -1)
        mu2 = mu2.unsqueeze(-3).expand(*batch_size, n_gauss_1, -1, -1)
        sigma1 = sigma1.unsqueeze(-2).expand(*batch_size, -1, n_gauss_2, -1)
        sigma2 = sigma2.unsqueeze(-3).expand(*batch_size, n_gauss_1, -1, -1)
        return gaussian_prob_density(mu1, mu2, sigma1+sigma2)
