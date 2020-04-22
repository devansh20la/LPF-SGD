import torch
from sklearn.metrics import average_precision_score


def accuracy(output, target, topk=(1,)):
    if output.shape[1] > 1:
        """Computes the accuracy over the k top predictions for the specified values of k"""
        with torch.no_grad():
            maxk = max(topk)
            batch_size = target.size(0)
            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))

            res = []
            for k in topk:
                correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100.0 / batch_size))
        return res
    else:
        batch_size = target.size(0)
        pred = output
        pred[pred >= 0.5] = 1.0
        pred[pred < 0.5] = 0.0
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        correct = correct.view(-1).float().sum(0, keepdim=True)
        return (correct.mul_(100.0 / batch_size),)


def precision(output, target, topk=(1,)):
    out = output.clone()
    tar = target.clone()

    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        out = out.cpu().detach().numpy()
        tar = tar.cpu().detach().numpy()
        average_precision = average_precision_score(tar, out,average='micro')
        return torch.Tensor([average_precision])


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
