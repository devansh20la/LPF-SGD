import csv
import pickle


with open('narrow_resnet_cifar10.pickle', 'rb') as handle:
    check_grid = pickle.load(handle)


print(check_grid)
with open('narrow_resnet_cifar10.csv', 'w') as f:
    # header
    f.write("hyper-parameter, good_exp, bad_exp \n")

    for k1, v1 in check_grid.items():
        for k2, v2 in v1.items():
            f.write(f"{k1}-{k2}, {v2[1]}, {v2[0]-v2[1]}\n")
