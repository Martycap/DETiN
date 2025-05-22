import os
from utils.dataset import create_triplets_from_tampered


print("Tampered images:", len(os.listdir("data/raw/CASIA2/Tampered")))
print("Mask images:", len(os.listdir("data/raw/CASIA2/Masks")))
print("Authentic images:", len(os.listdir("data/raw/CASIA2/Authentic")))
triplets = create_triplets_from_tampered("data/raw/CASIA2/Tampered", "data/raw/CASIA2/Masks", "data/raw/CASIA2/Authentic")
print(f"Totale triplets trovati: {len(triplets)}")