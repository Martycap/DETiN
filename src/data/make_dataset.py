import random, sys, os
sys.path.append(os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")))
from data.inpaint.inpaint_models import Inpaint
from data.inpaint.mask_generator import Mask
from data.inpaint.prompt_generator import load_prompts

from pycocotools.coco import COCO
from PIL import Image


def center_crop(image, size=256):
    """
    Crops the largest possible centered square from the image,
    then resizes it to (size, size) for CNN input.
    """
    width, height = image.size
    min_dim = min(width, height)

    left = (width - min_dim) // 2
    top = (height - min_dim) // 2
    right = left + min_dim
    bottom = top + min_dim

    cropped = image.crop((left, top, right, bottom))
    resized = cropped.resize((size, size), Image.LANCZOS)
    return resized


def image_loader(dataset: COCO, index, folder="./data/raw/val_images"):
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
    os.makedirs(directory, exist_ok=True)
    output_path = os.path.join(directory, file_name)
    image.save(output_path)
    return


def inpaint_pipeline():
    dataset = COCO("data/raw/annotations/instances_val2017.json")
    index = dataset.getImgIds()
    prompts = load_prompts()
    
    inpaint = Inpaint("./models/inpaint")
    mask_methods = {
        "segmentation": Mask.segmentation_mask,
        "bbox": Mask.bbox_mask,
        "random_box": lambda dataset, id: Mask.random_box_mask
    }
    inpaint_models = {
        "Stable_Diffusion": inpaint.Stable_Diffusion,
        "Kandinsky": inpaint.Kandinsky
    }
    
    combinations = []
    for mask_name in mask_methods:
        for model_name in inpaint_models:
            if model_name == "Kandinsky":
                combinations.append((mask_name, model_name, "random"))
            else:
                combinations.append((mask_name, model_name, "random"))
                combinations.append((mask_name, model_name, "realistic"))

    batch = 1000
    progress = {f"{m}_{n}_{p}": 0 for (m, n, p) in combinations}
    for image, id, file_name in image_loader(dataset, index):
        
        for mask_type, model_name, prompt_type in combinations:
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
                
                progress[key] += 1
                if all(images >= batch for images in progress.values()):
                    exit()

            except Exception as e:
                print(e)

if __name__ == "__main__":
    inpaint_pipeline()