import torch 
import numpy as np
import logging


class StepLR:
    def __init__(self, optimizer, learning_rate: float, total_epochs: int):
        self.optimizer = optimizer
        self.total_epochs = total_epochs
        self.base = learning_rate

    def step(self, epoch):        
        # let total_epochs = 200
        if epoch < self.total_epochs * 3/10: # 60
            lr = self.base
        elif epoch < self.total_epochs * 6/10: # 120
            lr = self.base * 0.2
        elif epoch < self.total_epochs * 8/10: # 160
            lr = self.base * 0.2 ** 2
        else:
            lr = self.base * 0.2 ** 3

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def lr(self) -> float:
        return self.optimizer.param_groups[0]["lr"]

def _cosine_annealing(step, total_steps, lr_max, lr_min):
    return lr_min + (lr_max -
                     lr_min) * 0.5 * (1 + np.cos(step / total_steps * np.pi))

def get_cosine_annealing_scheduler(optimizer, epochs, steps_per_epoch, base_lr):
    lr_min = 0.0
    total_steps = epochs * steps_per_epoch

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: _cosine_annealing(
            step,
            total_steps,
            1,  # since lr_lambda computes multiplicative factor
            lr_min / base_lr))

    return scheduler


class pyramidnet_scheduler:
    def __init__(self, optimizer, learning_rate: float, total_epochs: int):
        self.optimizer = optimizer
        self.total_epochs = total_epochs
        self.base_lr = learning_rate

    def step(self, epoch):
        lr = self.base_lr * (0.1 ** (epoch // (self.total_epochs*0.5))) * (0.1 ** (epoch // (self.total_epochs*0.75)))

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def lr(self) -> float:
        return self.optimizer.param_groups[0]["lr"]

