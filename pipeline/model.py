import torch
import torch.nn as nn
import pytorch_lightning as pl
import numpy as np
from scipy.stats import truncnorm
from pipeline.loss import EuclideanGMM


def truncated_normal(size, center, radius, dtype):
    values = truncnorm.rvs(-1, 1, size=size) * radius + center
    return torch.from_numpy(values.astype(dtype))


def DotProduct(tensor1, tensor2):
    tensor1 = tensor1.unsqueeze(-2)
    tensor2 = tensor2.unsqueeze(-1)
    return (tensor1 @ tensor2)[..., 0, 0]


class Word2Vec(pl.LightningModule):
    def __init__(self, dict_size, n_dims, lr=0.01, rate_adjust='CosineAnnealingLR', target_coef=1, **kwargs):
        super(Word2Vec, self).__init__()
        
        self.n_dims = n_dims
        self.embd_layers = nn.ModuleDict({
                'target': nn.Embedding(dict_size, n_dims),
                'context': nn.Embedding(dict_size, n_dims)
        })


        self.lr = lr
        self.rate_adjust = rate_adjust
        self.target_coef = target_coef
        
        self.distfunc = DotProduct
        self.lossfunc = nn.BCELoss(reduction='none')

        self.save_hyperparameters('dict_size', 'n_dims', 'lr', 'rate_adjust', 'target_coef')
    

    def forward(self, words):
        return self.embd_layers['target'](words).detach(),
    

    def training_step(self, batch, batch_idx):
        target, context, label = batch
        _, num_ns = context.shape
        target = target.expand(-1, num_ns)
        label = label.float()
        target_vec, context_vec = self.embd_layers['target'](target), self.embd_layers['context'](context)

        dist = self.distfunc(target_vec, context_vec)
        self.log('positive_samples_distance', torch.mean(dist[:, 0]))
        self.log('negative_samples_distance', torch.mean(dist[:, 1:]))

        prob = torch.sigmoid(dist)
        weight = torch.tensor([self.target_coef, 1, 1, 1, 1, 1], dtype=prob.dtype, device=prob.device).unsqueeze(0)
        loss = self.lossfunc(prob, label)
        loss = torch.mean(loss * weight)
        self.log('train_loss', loss)
        return loss


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        if self.rate_adjust == 'StepLR':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.75)
        elif self.rate_adjust == 'CosineAnnealingLR':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=self.lr/10)
        elif self.rate_adjust == 'CyclicLR':
            scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=self.lr/10, max_lr=self.lr, step_size_up=10, mode="triangular2", cycle_momentum=False)
        elif self.rate_adjust == 'none':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=1) # EQUAL TO NO SCHEDULER!
        return [optimizer], [scheduler]
    

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("Word2Vec")
        parser.add_argument('--n_dims', type=int, default=10)
        parser.add_argument('--lr', type=float, default=0.01)
        parser.add_argument('--rate_adjust', type=str, default='CosineAnnealingLR')
        parser.add_argument('--target_coef', type=int, default=1)
        return parent_parser


class Word2GMM(pl.LightningModule):
    def __init__(self, dict_size, n_gaussians, n_dims, center=1.2, radius=0.2, 
                 freeze_covar=True, anchors_embd=None, anchors_indices=None, anchoring='both',
                 lr=0.005, rate_adjust='StepLR', target_coef=1, **kwargs):
        super(Word2GMM, self).__init__()
        
        self.n_gaussians = n_gaussians
        self.n_dims = n_dims
        self.mu_layers = nn.ModuleDict({
                'target': nn.Embedding(dict_size, n_gaussians * n_dims),
                'context': nn.Embedding(dict_size, n_gaussians * n_dims)
        })

        self.w_layers = nn.ModuleDict({
                'target': nn.Sequential(nn.Embedding(dict_size, n_gaussians),
                                        nn.Softmax(dim=-1)),
                'context': nn.Sequential(nn.Embedding(dict_size, n_gaussians),
                                        nn.Softmax(dim=-1))
        })


        self.register_parameter('sigma', torch.nn.Parameter(truncated_normal([1, n_dims], center, radius, np.float32), requires_grad=not freeze_covar))

        self.register_buffer('anchors_embd', anchors_embd)
        self.register_buffer('anchors_indices', anchors_indices)

        self.lr = lr
        self.rate_adjust = rate_adjust
        self.target_coef = target_coef
        self.anchoring = anchoring
        
        self.distfunc = EuclideanGMM(reduction='none')
        self.lossfunc = nn.BCELoss(reduction='none')

        self.save_hyperparameters('dict_size', 'n_gaussians', 'n_dims', 'center', 
                                  'radius', 'freeze_covar', 'lr', 'rate_adjust', 
                                  'target_coef', 'anchoring')
    

    def forward(self, words):
        self.set_anchors()

        shape = words.shape
        words_mu = self.mu_layers['target'](words).reshape(*shape, self.n_gaussians, self.n_dims)
        words_w = self.w_layers['target'](words)
        words_sigma = self.sigma
        for _ in shape:
            words_sigma = words_sigma.unsqueeze(0)
        words_sigma = words_sigma.expand(*shape, self.n_gaussians, -1)
        return words_w.detach(), words_mu.detach(), words_sigma.detach()
    

    def training_step(self, batch, batch_idx):
        self.set_anchors()

        target, context, label = batch
        batch_sz, num_ns = context.shape
        target = target.expand(-1, num_ns)
        label = label.float()
        target_mu, context_mu = self.mu_layers['target'](target), self.mu_layers['context'](context)
        target_mu = target_mu.reshape(batch_sz, num_ns, self.n_gaussians, self.n_dims)
        context_mu = context_mu.reshape(batch_sz, num_ns, self.n_gaussians, self.n_dims)
        target_w, context_w = self.w_layers['target'](target), self.w_layers['context'](context)

        sigma_full = self.sigma.view(1, 1, 1, self.n_dims).expand(batch_sz, num_ns, self.n_gaussians, -1)
        dist = self.distfunc(target_w, target_mu, sigma_full, context_w, context_mu, sigma_full)
        self.log('positive_samples_distance', torch.mean(dist[:, 0]))
        self.log('negative_samples_distance', torch.mean(dist[:, 1:]))

        prob = torch.sigmoid(-dist)
        weight = torch.tensor([self.target_coef, 1, 1, 1, 1, 1], dtype=prob.dtype, device=prob.device).unsqueeze(0)
        loss = self.lossfunc(prob, label)
        loss = torch.mean(loss * weight)
        self.log('train_loss', loss)
        return loss


    def set_anchors(self):
        with torch.no_grad():
            anchoring_context = self.anchoring == 'both' or self.anchoring == 'context'
            anchoring_target = self.anchoring == 'both' or self.anchoring == 'target'
            if anchoring_context:
                dtype = self.w_layers['context'][0].weight.dtype
                device = self.w_layers['context'][0].weight.device
                self.mu_layers['context'].weight[self.anchors_indices] = self.anchors_embd.repeat(1, self.n_gaussians)
                self.w_layers['context'][0].weight[self.anchors_indices] = torch.ones([len(self.anchors_indices), self.n_gaussians], dtype=dtype, device=device) / self.n_gaussians
            
            if anchoring_target:
                dtype = self.w_layers['target'][0].weight.dtype
                device = self.w_layers['target'][0].weight.device
                self.mu_layers['target'].weight[self.anchors_indices] = self.anchors_embd.repeat(1, self.n_gaussians)
                self.w_layers['target'][0].weight[self.anchors_indices] = torch.ones([len(self.anchors_indices), self.n_gaussians], dtype=dtype, device=device) / self.n_gaussians


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        if self.rate_adjust == 'StepLR':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.75)
        elif self.rate_adjust == 'CosineAnnealingLR':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=self.lr/10)
        elif self.rate_adjust == 'CyclicLR':
            scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=self.lr/10, max_lr=self.lr, step_size_up=10, mode="triangular2", cycle_momentum=False)
        elif self.rate_adjust == 'none':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=1) # EQUAL TO NO SCHEDULER!
        return [optimizer], [scheduler]
    

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("Word2GMM")
        parser.add_argument('--n_gaussians', type=int, default=25)
        parser.add_argument('--n_dims', type=int, default=10)
        parser.add_argument('--center', type=float, default=1.2)
        parser.add_argument('--radius', type=float, default=0.2)
        parser.add_argument('--freeze_covar', action='store_true')
        parser.add_argument('--lr', type=float, default=0.005)
        parser.add_argument('--rate_adjust', type=str, default='StepLR')
        parser.add_argument('--target_coef', type=int, default=1)
        parser.add_argument('--anchoring', type=str, default='both')
        return parent_parser