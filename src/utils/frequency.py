import torch

def extract_frequency(img_tensor):
    freq_channels = []
    for c in range(img_tensor.shape[0]):
        channel = img_tensor[c]
        fft = torch.fft.fft2(channel)
        fft_shifted = torch.fft.fftshift(fft)
        magnitude = torch.abs(fft_shifted)
        magnitude = torch.log1p(magnitude)
        # normalizza per canale
        magnitude = (magnitude - magnitude.min()) / (magnitude.max() - magnitude.min() + 1e-8)
        freq_channels.append(magnitude)
    freq = torch.stack(freq_channels)
    return freq

