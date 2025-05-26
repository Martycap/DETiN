import torch

def extract_frequency(img_tensor):
    """
    Computes the frequency domain representation of an input RGB image tensor.

    For each channel, the function applies a 2D Fast Fourier Transform (FFT),
    shifts the zero-frequency component to the center, computes the logarithmic
    magnitude spectrum, and normalizes it to the [0, 1] range.
    """
    freq_channels = []
    for c in range(img_tensor.shape[0]):
        channel = img_tensor[c]
        fft = torch.fft.fft2(channel)
        fft_shifted = torch.fft.fftshift(fft)
        magnitude = torch.abs(fft_shifted)
        magnitude = torch.log1p(magnitude)
        magnitude = (magnitude - magnitude.min()) / (magnitude.max() - magnitude.min() + 1e-8)
        freq_channels.append(magnitude)
    freq = torch.stack(freq_channels)
    return freq

