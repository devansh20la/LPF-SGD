from args import get_args
import torch
import torch.nn as nn
from torchvision import transforms, datasets
from models import conv_net
import time


def get_hessian(batch_loss, model):
    t = time.time()

    grads = torch.autograd.grad(batch_loss, model.parameters(), create_graph=True)
    grads = torch.cat([x.view(-1) for x in grads])
    print(grads.shape[0])
    hessian = torch.zeros((grads.shape[0], grads.shape[0]))

    # i = 0
    # for grad_t in grads:
    #     for g in grad_t.reshape(-1):
    #         hess = torch.autograd.grad(g, model.parameters(), retain_graph=True)
    #         hess = torch.cat([h.reshape(-1) for h in hess])
    #         hessian[i, :] = hess
    #         i += 1

    # print(f"Time elapsed:{time.time() - t}")
    # return hessian


def main(args):

    data_tranforms = {
         'train': transforms.Compose([
                      transforms.ToTensor(),
                      transforms.Normalize((0.1307,), (0.3081,))]),
         'val': transforms.Compose([
                      transforms.ToTensor(),
                      transforms.Normalize((0.1307,), (0.3081,))])}

    dsets = {
        'train': datasets.MNIST('data/', train=True, download=True,
                                transform=data_tranforms['train']),
        'val': datasets.MNIST('data/', train=False, download=True,
                              transform=data_tranforms['val'])}

    dset_loaders = {
        x: torch.utils.data.DataLoader(dsets[x], batch_size=128, shuffle=(x == 'train'))
        for x in ['train', 'val']}

    model = conv_net()
    criterion = nn.CrossEntropyLoss()

    if args.use_cuda:
        torch.backends.cudnn.benchmark = True

    model.load_state_dict(torch.load(args.saved_model)['model'])

    for batch_idx, inp_data in enumerate(dset_loaders['train'], 1):
        inputs, targets = inp_data

        if args.use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()

        with torch.set_grad_enabled(True):
            model.zero_grad()
            outputs = model(inputs)
            batch_loss = criterion(outputs, targets)
        break

    get_hessian(batch_loss, model)


if __name__ == '__main__':
    args = get_args()

    main(args)
