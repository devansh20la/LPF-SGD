import torchvision.models as models
import eagerpy as ep
from foolbox import PyTorchModel, accuracy, samples
import foolbox.attacks as fa
import numpy as np
from models import Wide_ResNet, ShakeShake, PyramidNet
import torch
import argparse
from utils import accuracy, AverageMeter
from cifar import Cifar
from tqdm import tqdm 
import os


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # model configs
    parser.add_argument("--mtype", default="wrn", type=str, help="Model Type")
    parser.add_argument("--depth", default=16, type=int, help="Number of layers.")
    parser.add_argument("--width_factor", default=8, type=int, help="How many times wider compared to normal ResNet.")

    # Data preprocessing and loading
    parser.add_argument("--dtype", default='cifar10', type=str, help="dtype")
    parser.add_argument("--batch_size", default=1500, type=int, help="batch size")
    parser.add_argument("--seed", default=0, type=int, help="seed")
    parser.add_argument("--model_path", requirede=True, type=str)


    args = parser.parse_args()

    # initialze num_classes
    if args.dtype == 'cifar10': 
        args.num_classes = 10
    else:
        args.num_classes = 100
    
    # instantiate a model (could also be a TensorFlow or JAX model)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if args.mtype == "wrn":
        model = Wide_ResNet(
            args.depth, 
            args.width_factor, 
            num_classes=args.num_classes).to(device)
    elif args.mtype == "shakeshake":
        model = ShakeShake(
            args.depth,
            args.width_factor,
            input_shape=(1, 3, 32, 32),
            num_classes=args.num_classes).to(device)
    elif args.mtype == "pyramidnet":
        model = PyramidNet(args.depth, args.width_factor, args.num_classes).to(device)
    else:
        logging.fatal("Select correct model type")
        sys.exit()

    model.eval()
    state = torch.load(f"{args.model_path}/best_model.pth.tar")
    if "state_dict" in state:
        model.load_state_dict(state["state_dict"])
    else:
        model.load_state_dict(state)


    ########################## CHECK####################################################
    # if os.path.isfile(f"{model_path}/attack_success.npy"):
    #     print("File Exists")
    #     quit()
        
    ############################ CLEAN ACCURACY #########################################
    print(args)
    print(model_path)
    if args.dtype == 'cifar10':
        dataset = Cifar(args.batch_size, 12)
    elif args.dtype == 'cifar100':
        dataset = Cifar(args.batch_size, 12, want_cifar100=True)
    else:
        print("Print correct data")
        quit()

    err1 = AverageMeter()

    with torch.no_grad():
        for batch in dataset.test:
            inputs, targets = (b.to(device) for b in batch)
            predictions = model(inputs)
            err1.update(float(100 - accuracy(predictions, targets, topk=(1,))[0]), inputs.shape[0])
    clean_err = err1.avg
    print(f"Clean error: {clean_err}")
    ######################## Robust Accuracy ################################################

    fmodel = PyTorchModel(model, bounds=(0, 1))
    attacks = [
        fa.FGM(),
        fa.FGSM(),
        fa.LinfPGD(),
        fa.LinfBasicIterativeAttack(),
        fa.LinfAdditiveUniformNoiseAttack(),
        fa.LinfDeepFoolAttack(),
    ]

    epsilons = [8/255]

    print("epsilons")
    print(epsilons)
    print("")

    attack_success = np.zeros((len(attacks), len(epsilons), len(dataset.test.dataset)), dtype=np.bool_)
    idx = 0
    for batch in tqdm(dataset.test):
        inputs, targets = (b.to(device) for b in batch)

        for i, attack in enumerate(attacks):
            _, _, success = attack(fmodel, inputs, targets, epsilons=epsilons)
            assert success.shape == (len(epsilons), inputs.shape[0])
            success_ = success.cpu().numpy()
            assert success_.dtype == np.bool_
            attack_success[i,:,idx:idx + inputs.shape[0]] = success_
        idx+=inputs.shape[0]


    # print("robust accuracy for perturbations with")
    np.save(f"{model_path}/epsilons.npy", np.array(epsilons))
    np.save(f"{model_path}/attack_success.npy", attack_success)

