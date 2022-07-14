import numpy as np

class CosineInc:
    def __init__(self, std: float, num_epochs:int, steps_per_epoch: int, inc: int):
    	self.base = std
    	self.halfwavelength_steps = num_epochs * steps_per_epoch
    	self.inc = inc

    def __call__(self, step):
        scale_factor = -np.cos(step * np.pi / self.halfwavelength_steps) * 0.5 + 0.5
        self.current = self.base * (scale_factor * self.inc + 1)
        return self.current
