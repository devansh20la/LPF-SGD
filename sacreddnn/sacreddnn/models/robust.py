import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

class RobustNet(nn.Module):
    def __init__(self, Net, y=3, g=0., grate=1e-2, devices=None, gmax=float("inf"), use_center=False, Tmax=0, last_epoch=0):
            super(RobustNet, self).__init__()
            self.replicas = nn.ModuleList([Net() for a in range(y)])
            if devices is None:
                self.devices = [torch.device("cpu")]*(y+1)
            else:
                self.devices = devices
            self.distribute()
            self.master_device = self.devices[-1]
            self.g = g           # coupling
            self.grate = grate   # coupling increase rate

            self.gmax = gmax     # maximum coupling
            self.last_epoch = last_epoch
            self.Tmax = Tmax
            self.y = y           # number of replicas
            self.center = Net() if use_center else None
            self.Net = Net

    def distribute(self):
            for r, replica in enumerate(self.replicas):
                replica.to(self.devices[r])

    def forward(self, x, split_input=True, concatenate_output=True):
        if not isinstance(x, tuple) and not isinstance(x, list):
            if split_input:
                x = torch.chunk(x, self.y)
            else: # duplicate input
                x = tuple(x for a in range(self.y))

        x = [x[r].to(self.devices[r]) for r in range(len(x))]
        result = []
        for xr, replica in zip(x, self.replicas):
            result.append(replica(xr))

        result = [r.to(self.master_device) for r in result]

        if concatenate_output: # recompose
            return torch.cat(result)
        else:
            return result

    def has_center(self):
        return not self.center is None

    # num params per replica
    def num_params(self):
        return sum(p.numel() for p in self.replicas[0].parameters())

    def increase_g(self):
        self.g *= 1 + self.grate
        #self.g = min(self.g, self.gmax)
        
    def increase_g_linear(self):
        self.g += self.grate
        #self.g = min(self.g, self.gmax)

    def increase_g_cosine(self):
        self.g = -0.5 * self.gmax * (-1. + math.cos(math.pi * self.last_epoch / self.Tmax))
        self.last_epoch += 1
        
    def coupling_loss(self):
        return self.g * self.distance_loss()
    
    def distance_loss(self):
        return torch.mean(torch.stack(self.sqdistances()))

    # distances with the center of mass
    def sqdistances(self):
        dists = [0.0]*self.y
        if self.has_center():
            for a,r in enumerate(self.replicas):
                for wr, wc in zip(r.parameters(), self.center.parameters()):
                    dists[a] += F.mse_loss(wc, wr.to(self.master_device), reduction='sum')
        else:
            for wreplicas in zip(*(r.parameters() for r in self.replicas)):
                wreplicas_master = [w.to(self.master_device).detach() for w in wreplicas]
                wc = torch.mean(torch.stack(wreplicas_master).detach(), 0)
                for a, wr in enumerate(wreplicas):
                    dists[a] += F.mse_loss(wc.to(self.devices[a]), wr, reduction='sum').to(self.master_device)
        return dists

    def coupling_loss_d0(self, d0):
        return self.g * torch.mean(torch.stack(self.sqdistances_d0(d0)))
    
    # distances with the center of mass
    def sqdistances_d0(self, d0):
        dists = [0.0]*self.y
        if not self.has_center():
            for wreplicas in zip(*(r.parameters() for r in self.replicas)):
                wreplicas_master = [w.to(self.master_device).detach() for w in wreplicas]
                wc = torch.mean(torch.stack(wreplicas_master).detach(), 0)
                for a, wr in enumerate(wreplicas):
                    dists[a] += (F.mse_loss(wc.to(self.devices[a]), wr, reduction='sum').to(self.master_device) - d0)**2
        return dists

    
    def sqnorms(self):
        sqns = [0.0]*self.y
        for wreplicas in zip(*(s.parameters() for s in self.replicas)):
            for a, wr in enumerate(wreplicas):
                sqns[a] += wr.norm()**2
        return sqns

    def build_center_of_mass(self):
        center = self.Net()
        center.to(self.master_device)
        for wc, *wreplicas in zip(center.parameters(), *(r.parameters() for r in self.replicas)):
            wreplicas = [w.to(self.master_device) for w in wreplicas]
            wc.data = torch.mean(torch.stack(wreplicas), 0).data

        for bc, *breplicas in zip(center.buffers(), *(r.buffers() for r in self.replicas)):
            breplicas = [b.to(self.master_device) for b in breplicas]
            if breplicas[0].dtype == torch.long:
                bc.data = torch.ceil(torch.mean(torch.stack(breplicas).double())).long().data
            else:
                bc.data = torch.mean(torch.stack(breplicas), 0).data

        return center

    def get_or_build_center(self):
        if self.has_center():
            return self.center.to(self.master_device)
        else:
            return self.build_center_of_mass()


class RobustDataLoader():
    def __init__(self, dset, y, concatenate, M=-1, **kwargs):
        if M > 0:
            dset = Subset(dset, range(min(M, len(dset))))
        self.y = y
        self.dls = [DataLoader(dset, **kwargs) for a in range(y)]
        self.concatenate = concatenate
        self.dataset = dset

    def __iter__(self):
        if self.concatenate:
            for xs in zip(*self.dls):
                yield tuple(torch.cat([x[i] for x in xs]) for i in range(len(xs[0])))
        else:
            for xs in zip(*self.dls):
                yield xs

    def __len__(self):
        return len(self.dls[0])

    def single_loader(self):
        return self.dls[0]
