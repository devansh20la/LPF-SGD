import torch
import glob
from tqdm import tqdm
from sklearn.decomposition import PCA
from scipy import linalg
import numpy as np


def main():

    all_states = []
    for file in tqdm(glob.glob1("checkpoints/mnist/run0", "*.pth.tar")):
        state = torch.load(f"checkpoints/mnist/run0/{file}", map_location='cpu')
        state = state["grads"]
        all_states += state

    P = torch.empty((all_states[0].shape[0], len(all_states)))
    for k, up in enumerate(tqdm(all_states), 0):
        P[:, k] = up
    P = P.numpy().T

    S = linalg.svdvals(P)
    np.save("sig_val.pkl", S)
    print(S)
    # pca = PCA(n_components=0)
    # pca.fit(P)

    # print(pca.explained_variance_ratio_)
    # print(pca.singular_values_)


if __name__ == "__main__":
    main()