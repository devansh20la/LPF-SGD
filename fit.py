import torch
import logging
import time
import utils as uf


def class_model_run(phase, loader, model, criterion, optimizer, args):
    """
        Function to forward pass through classification problem
    """
    logger = logging.getLogger('my_log')

    if phase == 'train':
        model.train()
    else:
        model.eval()

    loss = uf.AverageMeter()
    err1 = uf.AverageMeter()
    err5 = uf.AverageMeter()
    t = time.time()

    for batch_idx, inp_data in enumerate(loader[phase],1):

        inputs, targets = inp_data

        if args.use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()

        if phase == 'train':
            with torch.set_grad_enabled(True):
                # compute output
                outputs = model(inputs)
                # print(outputs.size(), targets.size())
                batch_loss = criterion(outputs, targets)

                # compute gradient and do SGD step
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()

        elif phase == 'val':
            with torch.no_grad():
                outputs = model(inputs)
                batch_loss = criterion(outputs, targets)
        else:
            logger.info('Define correct phase')
            quit()

        loss.update(batch_loss.item(), inputs.size(0))
        batch_err = uf.accuracy(outputs, targets, topk=(1,5))
        err1.update(float(100.0 - batch_err[0]), inputs.size(0))
        err5.update(float(100.0 - batch_err[1]), inputs.size(0))

        if batch_idx % args.print_freq == 0:
            logger.info("Phase:{0} -- Batch_idx:{1}/{2} -- {3:.2f} samples/sec"
                        "-- Loss:{4:.2f} -- Error1:{5:.2f}".format(
                          phase, batch_idx, len(loader[phase]),
                          err1.count / (time.time() - t), loss.avg, err1.avg))

    return loss.avg, err1.avg, err5.avg
