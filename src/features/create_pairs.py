import os, tqdm

def create_coco_pairs(inpaint_dir, mask_dir):
    """
    Creates a list of pairs for COCO dataset.
    """
    inpaint_files = {os.path.splitext(f)[0] for f in os.listdir(inpaint_dir)}
    mask_files = {os.path.splitext(f)[0] for f in os.listdir(mask_dir)}
    
    common_files = inpaint_files & mask_files
    
    pairs = []
    for name in common_files:
        inpaint = os.path.join(inpaint_dir, name + ".jpg")
        mask = os.path.join(mask_dir, name + ".jpg")
        pairs.append((inpaint, mask))

    return pairs

def create_casia_pairs(tp_list_path, tampered_dir, mask_dir):
    """
    Create a list of pairs for CASIA2 dataset.
    """
    
    pairs = []

    with open(tp_list_path, 'r') as f:
        tampered_files = [line.strip() for line in f if line.strip()]

    for tampered_filename in tqdm.tqdm(tampered_files, desc="Creating tampered-mask pairs"):
        tampered_path = os.path.join(tampered_dir, tampered_filename)
        mask_filename = os.path.splitext(tampered_filename)[0] + '_gt.png'
        mask_path = os.path.join(mask_dir, mask_filename)

        if not os.path.exists(tampered_path):
            continue

        if not os.path.exists(mask_path):
            continue

        pairs.append((os.path.abspath(tampered_path), os.path.abspath(mask_path)))

    return pairs