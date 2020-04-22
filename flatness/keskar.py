from args import get_args
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from models import LeNet
from functools import reduce
import copy
from utils import AverageMeter

from utils import vector_to_parameter_list as v2p
from torch.nn.utils import parameters_to_vector as p2v


def load_model_state(model, theta_star, matrix_A, y):

    t = matrix_A @ y
    A_y = v2p(t, model.parameters())

    for curr_state, t_star_state, new_state in zip(model.parameters(), theta_star, A_y):
        curr_state.data = t_star_state + new_state


def main(args):
    ep = 1e-3

    ################# DATA INITIALIZATION ######################
    data_tranforms = {
        x: transforms.Compose([transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))])
        for x in ['train', 'val']}

    dsets = {
        x: datasets.MNIST('data/', train=(x == 'train'), download=True,
                          transform=data_tranforms[x])
        for x in ['train', 'val']}

    dset_loaders = {
        x: torch.utils.data.DataLoader(dsets[x], batch_size=args.bs,
                                       shuffle=(x == 'train'))
        for x in ['train', 'val']}

    ############## MODEL INITIALIZATION ###########################
    model = LeNet()
    model.load_state_dict(torch.load(args.saved_model, map_location='cpu')['model'])

    criterion = nn.CrossEntropyLoss()

    if args.use_cuda:
        model.cuda()
        torch.backends.cudnn.benchmark = True

    theta_star = copy.deepcopy(list(model.parameters()))

    d = 0
    for param in model.parameters():
        d += reduce(lambda x, y: x * y, param.shape)

    p = 100

    if args.use_cuda:
        A_mtx = torch.randn((d, p)).cuda()
    else:
        A_mtx = torch.randn((d, p))

    b = p2v(theta_star).reshape(-1, 1)
    t, _ = torch.lstsq(b, A_mtx)
    t = t[:p]
    A_inv_theta = torch.abs(t) + 1

    if args.use_cuda:
        y = torch.randn((p, 1), requires_grad=True, device='cuda')
    else:
        y = torch.randn((p, 1), requires_grad=True)

    y.data = torch.max(torch.min(y.data, ep*A_inv_theta), -ep*A_inv_theta)

    optimizer = optim.LBFGS([y], lr=args.lr)
    phase = 'train'

    for epoch in range(10):
        loss = AverageMeter()

        for batch_idx, inp_data in enumerate(dset_loaders[phase], 1):
            inputs, targets = inp_data

            load_model_state(model, theta_star, A_mtx, y)

            if args.use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()

            with torch.set_grad_enabled(True):
                def closure():
                    if torch.is_grad_enabled():
                        optimizer.zero_grad()
                    output = model(inputs)
                    batch_loss = -1*criterion(output, targets)
                    if batch_loss.requires_grad:
                        batch_loss.backward()

                    return batch_loss

                optimizer.step(closure)

            # Compute loss again from saving purposes
            with torch.set_grad_enabled(False):
                batch_loss = closure()

            loss.update(-batch_loss.item(), inputs.size(0))

            # project y back to the constraint space
            with torch.set_grad_enabled(False):
                y.data = torch.max(torch.min(y.data, A_inv_theta*ep), -ep*A_inv_theta)

            if batch_idx % args.print_freq == 0:
                print("Phase:{0} -- Batch_idx:{1}/{2} -- Loss:{3:.6f}".format(
                      phase, batch_idx, len(dset_loaders[phase]), loss.avg))

        print('Train_Loss = {0}'.format(loss.avg))


if __name__ == '__main__':
    # args = get_args()
    args = get_args(["--saved_model", "checkpoints/mnist/run0/train_model_ep0.pth.tar", "--dtype", "mnist",
                     "--bs", "128", "--ms", "123"])

    main(args)
