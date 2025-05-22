import os
from utils.dataset import create_triplets_from_tampered, show_triplet
from models.casia_dataset import CASIATransformerDataset, plot_image_noise_freq


authentic_dir = "data/raw/CASIA2/Authentic"
tampered_dir = "data/raw/CASIA2/Tampered"
mask_dir = "data/raw/CASIA2/Masks"


def main():
    print("Tampered images:", len(os.listdir(tampered_dir)))
    print("Mask images:", len(os.listdir(mask_dir)))
    print("Authentic images:", len(os.listdir(authentic_dir)))

    triplets = create_triplets_from_tampered(tampered_dir, mask_dir, authentic_dir)

    print(f"Totale triplets trovati: {len(triplets)}")

    if triplets:
        show_triplet(*triplets[0])
            
        dataset = CASIATransformerDataset(triplets)
        x, mask = dataset[0]  # x: [9,H,W] (img+noise+freq)
            
        # Split channel
        image = x[:3, :, :]
        noise = x[3:6, :, :]
        freq = x[6:9, :, :]
            
        # Plot image original + noise + frequency
        plot_image_noise_freq(image, noise, freq)
            
    else:
        print("Nessun triplet trovato.")

if __name__ == '__main__':
    main()
