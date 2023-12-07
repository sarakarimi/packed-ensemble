from torchmetrics import Metric
import torch
import torch.nn.functional as F


class NLL(Metric):
    def __init__(self, reduction="mean", **kwargs):
        super().__init__(**kwargs)
        self.reduction = reduction
        self.add_state("nlls", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, probs, target):

        self.nlls += F.nll_loss(torch.log(probs), target, reduction=self.reduction)
        self.total += target.size(0)

    def compute(self):
        return self.nlls.sum(dim=-1) / self.total
