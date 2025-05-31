import random, sys, os, csv
sys.path.append(os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")))
from data.inpaint.inpaint_models import Inpaint
from data.inpaint.mask_generator import Mask
from data.inpaint.prompt_generator import load_prompts
from features.build_features import center_crop

from datetime import datetime
from pycocotools.coco import COCO
from PIL import Image


def image_loader(dataset: COCO, index, folder="./data/raw/COCO"):
    """
    Lazy image loader based on the IDs provided by COCO dataset.
    """

    for id in index:
        info = dataset.loadImgs([id])[0]
        file_name = info["file_name"]
        file_path = os.path.join(folder, file_name)
        try:
            img = Image.open(file_path).convert("RGB")
            resized = center_crop(img)
            yield (resized, id, file_name)
        except Exception as e:
            print(f"Error with image: {file_name} -> {e}")
    return


def save(image, directory, file_name):
    """
    Saves images and masks in the directory.
    """
    
    os.makedirs(directory, exist_ok=True)
    output_path = os.path.join(directory, file_name)
    image.save(output_path)
    return


def chunk_loader(lst, n):
    """
    Chunk loader for COCO index list.
    """
    
    avg = len(lst) / float(n)
    chunks = []
    last = 0
    
    while last < len(lst):
        chunks.append(lst[int(last):int(last + avg)])
        last += avg
    return chunks


def inpaint_pipeline():
    """
    Inpainting pipeline that automatically alternates between different 
    inpainting methods and mask types to generate the dataset.
    """
    
    dataset = COCO("./data/raw/annotations/instances_val2017.json")
    index = dataset.getImgIds()
    prompts = load_prompts()
    
    inpaint = Inpaint("./models/inpaint")
    inpaint_models = {
        "Stable_Diffusion": inpaint.Stable_Diffusion,
        "Kandinsky": inpaint.Kandinsky
    }
        
    mask_methods = {
        "segmentation": Mask.segmentation_mask,
        "bbox": Mask.bbox_mask,
        "random_box": lambda dataset, id: Mask.random_box_mask()
    }
    
    combinations = []
    for mask_name in mask_methods:
        for model_name in inpaint_models:
            if model_name == "Kandinsky":
                combinations.append((mask_name, model_name, "random"))
            else:
                combinations.append((mask_name, model_name, "random"))
                combinations.append((mask_name, model_name, "realistic"))

    batch = 555
    random.shuffle(index)
    chunks = chunk_loader(index, len(combinations))
    progress = {f"{m}_{n}_{p}": 0 for (m, n, p) in combinations}
    for (mask_type, model_name, prompt_type), ids in zip(combinations, chunks):
        
        for image, id, file_name in image_loader(dataset, ids):
            key = f"{mask_type}_{model_name}_{prompt_type}"
            
            if progress[key] >= batch:
                continue

            try:
                mask = mask_methods[mask_type](dataset, id)
                mask = Image.fromarray(mask).convert("L")

                prompt = (random.choice(prompts) if prompt_type == "random" 
                            else "Fill the missing region realistically")
                inpaint = inpaint_models[model_name]
                inpainted, gt_mask = inpaint(prompt, image, mask)
            
                save(inpainted, f"./data/processed/train/{key}", file_name)
                save(gt_mask, f"./data/processed/masks/{key}", file_name)

                now = datetime.now().strftime("%H:%M:%S")
                with open("./data/processed/inpaint_log.csv", "a", newline="", encoding="utf-8") as csv_file:
                    writer = csv.writer(csv_file)
                    writer.writerow([now, file_name, model_name, mask_type, prompt])

                progress[key] += 1
                if all(images >= batch for images in progress.values()):
                    exit()

            except Exception as e:
                print("\n\n --- ", e, " --- \n\n")

if __name__ == "__main__":
    inpaint_pipeline()