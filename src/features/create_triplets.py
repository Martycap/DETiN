import os, json, cv2, tqdm
import matplotlib.pyplot as plt

path_list_acronyms = "data/raw/CASIA2/list_acronyms.json"
path_tp_name = "tp_list.txt"

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


def extract_acronyms(filename):
    """
    Function that analyzes the path names of authentic photos 
    present in the CASIA2 directory and extracts the discriminating acronyms.
    """
    acronyms = set()
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            parts = line.split('_')
            if len(parts) >= 3:
                acronyms.add(parts[1])
    return list(acronyms)




def create_pairs_from_tp_list(tp_list_path, tampered_dir, mask_dir):
    """
    Crea una lista di tuple (tampered_image_path, mask_path) leggendo da tp_list.txt.
    Restituisce i path assoluti solo se entrambi i file esistono.

    Args:
        tp_list_path (str): path al file tp_list.txt
        tampered_dir (str): cartella contenente le immagini tampered
        mask_dir (str): cartella contenente le maschere

    Returns:
        List[Tuple[str, str]]: coppie (tampered_path, mask_path), entrambi assoluti
    """
    pairs = []

    with open(tp_list_path, 'r') as f:
        tampered_files = [line.strip() for line in f if line.strip()]

    for tampered_filename in tqdm.tqdm(tampered_files, desc="Creating tampered-mask pairs"):
        tampered_path = os.path.join(tampered_dir, tampered_filename)
        mask_filename = os.path.splitext(tampered_filename)[0] + '_gt.png'
        mask_path = os.path.join(mask_dir, mask_filename)

        if not os.path.exists(tampered_path):
            tqdm.tqdm.write(f"Immagine non trovata: {tampered_path}")
            continue

        if not os.path.exists(mask_path):
            tqdm.tqdm.write(f"Maschera non trovata: {mask_path}")
            continue

        pairs.append((os.path.abspath(tampered_path), os.path.abspath(mask_path)))

    return pairs



def create_pairs_COCO():
    """
    Crea una lista di tuple (tampered_image_path, mask_path) del dataset COCO.
    Returns:
        List[Tuple[str, str]]: coppie (tampered_path, mask_path), entrambi assoluti
    """
    modified_dirs_for_inference = [
        "data/processed/train/bbox_Kandinsky_random",
        "data/processed/train/bbox_Stable_Diffusion_random",
        "data/processed/train/bbox_Stable_Diffusion_realistic",
        "data/processed/train/random_box_Kandinsky_random",
        "data/processed/train/random_box_Stable_Diffusion_random",
        "data/processed/train/random_box_Stable_Diffusion_realistic",
        "data/processed/train/segmentation_Kandinsky_random",
        "data/processed/train/segmentation_Stable_Diffusion_random",
        "data/processed/train/segmentation_Stable_Diffusion_realistic"
    ]
    mask_dir = "data/processed/masks/CNN_masks"
    pairs = []

    for mod_dir in tqdm.tqdm(modified_dirs_for_inference):
        if not os.path.isdir(mod_dir):
            print(f"Directory non trovata: {mod_dir}")
            continue

        for file in os.listdir(mod_dir):
            if file.lower().endswith(".jpg"):
                tampered_path = os.path.abspath(os.path.join(mod_dir, file))
                base_name = os.path.splitext(file)[0]
                mask_filename = f"{base_name}.png"
                mask_path = os.path.abspath(os.path.join(mask_dir, mask_filename))

                if os.path.exists(mask_path):
                    pairs.append((tampered_path, mask_path))
                else:
                    print(f"Maschera non trovata per: {tampered_path}")
    
    return pairs

if __name__ == "__main__":
    input_filename = "data/raw/CASIA2/au_list.txt" 
    output_filename = "data/raw/CASIA2/list_acronyms.json"
    acronyms_list = extract_acronyms(input_filename)
    with open(output_filename, 'w') as f:
        json.dump(acronyms_list, f)
