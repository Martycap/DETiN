import os, cv2, pickle
import matplotlib.pyplot as plt


def create_triplets_from_tampered(inpaint_dir, mask_dir, authentic_dir):
    inpaint_files = {os.path.splitext(f)[0] for f in os.listdir(inpaint_dir)}
    mask_files = {os.path.splitext(f)[0] for f in os.listdir(mask_dir)}
    authentic_files = {os.path.splitext(f)[0] for f in os.listdir(authentic_dir)}
    
    common_files = inpaint_files & mask_files & authentic_files
    
    triplets = []
    for name in common_files:
        original = os.path.join(authentic_dir, name + ".jpg")
        inpaint = os.path.join(inpaint_dir, name + ".jpg")
        mask = os.path.join(mask_dir, name + ".jpg")
        triplets.append((original, inpaint, mask))

    return triplets
    
    
def create_pairs_from_tampered(inpaint_dir, mask_dir):
    inpaint_files = {os.path.splitext(f)[0] for f in os.listdir(inpaint_dir)}
    mask_files = {os.path.splitext(f)[0] for f in os.listdir(mask_dir)}
    
    common_files = inpaint_files & mask_files
    
    triplets = []
    for name in common_files:
        inpaint = os.path.join(inpaint_dir, name + ".jpg")
        mask = os.path.join(mask_dir, name + ".jpg")
        triplets.append((inpaint, mask))

    return triplets

def show_triplet(original_path, tampered_path, mask_path):
    original = cv2.imread(original_path)
    tampered = cv2.imread(tampered_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    tampered = cv2.cvtColor(tampered, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(15,5))
    plt.subplot(1,3,1)
    plt.imshow(original)
    plt.title("Originale")
    plt.axis('off')

    plt.subplot(1,3,2)
    plt.imshow(tampered)
    plt.title("Modificata")
    plt.axis('off')

    plt.subplot(1,3,3)
    plt.imshow(mask, cmap='gray')
    plt.title("Maschera")
    plt.axis('off')

    plt.show()