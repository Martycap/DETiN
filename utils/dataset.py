import os
import matplotlib.pyplot as plt
import cv2
import tqdm


def find_file_with_prefix(directory, prefix):
    """
    Cerca nel directory un file che inizia con prefix (prima dell'estensione)
    e restituisce il path completo.
    Ritorna None se non trova nulla.
    """
    for filename in os.listdir(directory):
        name, ext = os.path.splitext(filename)
        if name == prefix:
            return os.path.join(directory, filename)
    return None

def find_file_with_prefix_flexible(directory, prefix):
    """
    Cerca un file in directory che inizia con prefix (il prefix può essere tutta la parte senza estensione)
    e ritorna il primo che trova (qualsiasi estensione).
    """
    for filename in os.listdir(directory):
        name, ext = os.path.splitext(filename)
        if name == prefix:
            return os.path.join(directory, filename)
    return None

def create_triplets_from_tampered(tampered_dir, mask_dir, authentic_dir):
    triplets = []
    valid_acronyms = {'nat', 'arc', 'pla', 'sec', 'ani', 'cha', 'ind', 'txt', 'art'}


    for tampered_file in tqdm.tqdm(os.listdir(tampered_dir), desc="Processing tampered files"):
        # Non filtri sull'estensione per permettere qualunque tipo di file
        base_name = os.path.splitext(tampered_file)[0]

        # Maschera con nome base + '_gt.png'
        mask_name = base_name + '_gt.png'
        mask_path = os.path.join(mask_dir, mask_name)
        if not os.path.exists(mask_path):
            continue  # salta se maschera non esiste

        # Estrai codice originale dal nome tampered (ani00018, art00076, ...)
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

        # Costruisci prefix originale: Au_ani_00018 (senza estensione)
        original_prefix = f"Au_{original_code[:3]}_{original_code[3:]}"
        original_path = find_file_with_prefix_flexible(authentic_dir, original_prefix)
        if original_path is None:
            continue  # se non trovato, salta

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


if __name__ == '__main__':
    authentic_dir = '/path/to/Authentic'
    tampered_dir = '/path/to/Tampered'
    mask_dir = '/path/to/Masks'

    triplets = create_triplets_from_tampered(tampered_dir, mask_dir, authentic_dir)

    print(f"Totale triplets trovati: {len(triplets)}")

    if triplets:
        original, tampered, mask = triplets[0]
        print("Primo esempio:")
        print(f"Immagine originale: {original}")
        print(f"Immagine modificata: {tampered}")
        print(f"Maschera: {mask}")
        show_triplet(original, tampered, mask)
    else:
        print("Nessun triplet trovato.")
