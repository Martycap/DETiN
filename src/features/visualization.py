import random, sys, numpy as np, os, torch

sys.path.append(os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")))
from data.DETiN_dataset import plot_image_noise_freq


def set_seed(seed=42):
    """
    Set seed for reproducibility across random, numpy, and torch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def visualize_sample(dataset, index=0):
    """
    Visualize a sample image, its noise, and frequency from the dataset.
    """
    x, _, _ = dataset[index]
    image = x[:3, :, :]
    noise = x[3:6, :, :]
    freq = x[6:9, :, :]
    plot_image_noise_freq(image, noise, freq)
