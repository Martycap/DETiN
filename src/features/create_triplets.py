import os, json, cv2, tqdm
import matplotlib.pyplot as plt

path_list_acronyms = "data/raw/CASIA2/list_acronyms.json"

def find_file_with_prefix(directory, prefix):
    """
    Searches the directory for a file that starts with prefix (before the extension)
    and returns the full path.
    Returns None if nothing is found.
    """
    for filename in os.listdir(directory):
        name, ext = os.path.splitext(filename)
        if name == prefix:
            return os.path.join(directory, filename)
    return None

def find_file_with_prefix_flexible(directory, prefix):
    """
    Searches for a file in a directory that starts with prefix 
    (the prefix can be the entire part without extension)
    and returns the first one found (any extension).
    """
    for filename in os.listdir(directory):
        name, ext = os.path.splitext(filename)
        if name == prefix:
            return os.path.join(directory, filename)
    return None

def create_triplets_from_tampered(tampered_dir, mask_dir, authentic_dir):
    triplets = []
    
    with open(path_list_acronyms, "r") as f:
        valid_acronyms = json.load(f)
        

    for tampered_file in tqdm.tqdm(os.listdir(tampered_dir), desc="Processing tampered files"):
        base_name = os.path.splitext(tampered_file)[0]

        mask_name = base_name + '_gt.png'
        mask_path = os.path.join(mask_dir, mask_name)
        if not os.path.exists(mask_path):
            continue  

        parts = base_name.split('_')
        original_code = None
        for part in parts:
            for acronym in valid_acronyms:
                if part.startswith(acronym):
                    original_code = part
                    break
            if original_code is not None:
                break
        if original_code is None:
            continue

        original_prefix = f"Au_{original_code[:3]}_{original_code[3:]}"
        original_path = find_file_with_prefix_flexible(authentic_dir, original_prefix)
        if original_path is None:
            continue  

        tampered_path = os.path.join(tampered_dir, tampered_file)
        triplets.append((original_path, tampered_path, mask_path))

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


