import torch
import torch.nn as nn
from torch.nn.modules.batchnorm import _BatchNorm

# Multiple Loss Decomposition Optimizer
class MLDOptim(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, model, alpha, perturb_eps=1e-12, **kwargs):
        defaults = dict(**kwargs)
        super(MLDOptim, self).__init__(params, defaults)
        self.model = model
        self.base_optimizer = base_optimizer
        self.param_groups = self.base_optimizer.param_groups
        self.alpha = alpha
        self.perturb_eps = perturb_eps

    # get gradient of secondary_loss
    @torch.no_grad()
    def secondary_loss_backward(self, loss):
        loss.backward(retain_graph=True)
        for group in self.param_groups:
            for pItem in group["params"]:
                if pItem.grad is None: continue
                self.state[pItem]["old_g"] = pItem.grad.data.clone()
        self._disable_running_stats(self.model)

    # get gradient of primary_loss
    @torch.no_grad()
    def primary_loss_backward(self, loss):
        loss.backward(retain_graph=True)
        # calculate inner product
        inner_prod = 0.0
        for group in self.param_groups:
            for pItem in group['params']:
                if pItem.grad is None: continue
                try:
                    inner_prod += torch.sum(self.state[pItem]['old_g'] * pItem.grad.data)
                except:
                    aaa = 1

        if inner_prod > 0:
            # get norm
            new_grad_norm = self._grad_norm()
            old_grad_norm = self._grad_norm(by='old_g')

            # get cosine
            cosine = inner_prod / (new_grad_norm * old_grad_norm + self.perturb_eps)

            # gradient decomposition
            for group in self.param_groups:
                for pItem in group['params']:
                    if pItem.grad is None: continue
                    try:
                        vertical = self.state[pItem]['old_g'] - cosine * old_grad_norm * pItem.grad.data / (new_grad_norm + self.perturb_eps)
                        pItem.grad.data.add_( vertical, alpha=-self.alpha)
                    except:
                        aaa = 1

    @torch.no_grad()
    def optim_step(self):
        # update with new directions
        self.base_optimizer.step()

        # enable running stats
        self._enable_running_stats(self.model)


    @torch.no_grad()
    def _grad_norm(self, by=None):
        #shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        params = self.param_groups[0]["params"]
        if not by:
            array = []
            for pa in params:
                if pa.grad is not None:
                    array.append(pa.grad.norm(p=2))
            return torch.norm(torch.stack(array), p=2)
        else:
            array = []
            for pa in params:
                if pa.grad is not None:
                    try:
                        array.append(self.state[pa][by].norm(p=2))
                    except:
                        aaa = 1
            return torch.norm(torch.stack(array), p=2)

    def _disable_running_stats(self, model):
        def _disable(module):
            if isinstance(module, _BatchNorm):
                module.backup_momentum = module.momentum
                module.momentum = 0

        model.apply(_disable)

    def _enable_running_stats(self, model):
        def _enable(module):
            if isinstance(module, _BatchNorm) and hasattr(module, "backup_momentum"):
                module.momentum = module.backup_momentum

        model.apply(_enable)