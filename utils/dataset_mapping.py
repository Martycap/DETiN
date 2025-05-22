import os
import matplotlib.pyplot as plt
import cv2


def create_triplets_from_tampered(tampered_dir, mask_dir, authentic_dir):
    triplets = []

    for tampered_file in os.listdir(tampered_dir):
        if not tampered_file.endswith('.tif'):
            continue

        # Nome base per maschera
        base_name = os.path.splitext(tampered_file)[0]
        mask_name = base_name + '_gt.png'
        mask_path = os.path.join(mask_dir, mask_name)

        if not os.path.exists(mask_path):
            continue  # skip se maschera non trovata

        # Estrai codice immagine originale dal nome tampered (es. ani00018 o art00076)
        parts = base_name.split('_')
        for part in parts:
            if part.startswith('ani') or part.startswith('art'):
                original_code = part
                break
        else:
            continue  # se non trova ani/art, salta

        # Costruisci nome originale: Au_ani_00018.jpg
        original_name = f"Au_{original_code[:3]}_{original_code[3:]}.jpg"
        original_path = os.path.join(authentic_dir, original_name)

        if not os.path.exists(original_path):
            continue  # skip se immagine reale mancante

        tampered_path = os.path.join(tampered_dir, tampered_file)
        triplets.append((original_path, tampered_path, mask_path))

    return triplets


def show_triplet(original_path, tampered_path, mask_path):
    # Leggi le immagini con OpenCV
    original = cv2.imread(original_path)
    tampered = cv2.imread(tampered_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # maschera bianco/nero

    # Converti BGR (OpenCV) in RGB per matplotlib
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
    print(f"Totale triplets trovati: {len(triplets)}")

    if triplets:
        original, tampered, mask = triplets[0]
        print("Primo esempio:")
        print(f"Immagine originale: {original}")
        print(f"Immagine modificata: {tampered}")
        print(f"Maschera: {mask}")
    else:
        print("Nessun triplet trovato.")