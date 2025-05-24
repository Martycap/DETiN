import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from data.inpaint.inpaint_models import Inpaint
from data.inpaint.mask_generator import Mask
from pycocotools.coco import COCO
import random, cv2, os

def center_crop(image, size = 256):
    """
    Returns the image cropped to a centered square,
    then resized to preserve aspect ratio and avoid distortion.
    """
    
    height, width, _ = image.shape
    min_dim = min(height, width)

    top = (height - min_dim) // 2
    left = (width - min_dim) // 2
    cropped = image[top:top+min_dim, left:left+min_dim]

    resized = cv2.resize(cropped, (size, size), interpolation=cv2.INTER_LANCZOS4)
    return resized

def getImages():
    """
    Retrievs images from the folder.
    """
    
    folder = "./data/raw/val_images"
    images = []
    for image in os.listdir(folder):
        path = os.path.join(folder, image)
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        resized = center_crop(image)
        images.append(resized)
    return images

def prompt_generator(dataset, index, selections):
    parts = []
    for _ in range(selections - 1):
        id = random.choice(index)
        annotations = dataset.loadAnns(dataset.getAnnIds(imgIds=id))
        part = random.choice(annotations)
        parts.append(part['caption'].rstrip('. ').strip())
    return " and ".join(parts)


if __name__ == "__main__":
    dataset = COCO("data/raw/annotations/captions_val2017.json")
    index = dataset.getImgIds()
    
    prompt = prompt_generator(dataset, index, 3)
    print(prompt)
    images = getImages()
    
    import matplotlib.pyplot as plt
    plt.imshow(images[342])
    plt.axis('off')
    plt.show()
    
    mask = Mask.random_box_mask()
    
    models = Inpaint("./models/inpaint")
    
    from PIL import Image
    image = Image.fromarray(images[342])
    mask = Image.fromarray(mask).convert("L")
    plt.imshow(mask, cmap = "gray")
    plt.axis('off')
    plt.show()
    
    
    output = models.inference_kandinsky(prompt, images, mask)
    
    output.show()
    
    # img_id = random.choice(index)
    # img_info = dataset.loadImgs(img_id)[0]
    # print(img_info)

    # annotations = dataset.loadAnns(dataset.getAnnIds(imgIds=img_id))
    # for ann in annotations:
    #     print(ann)
    
    # img_path = os.path.join("data/raw/val_images", img_info["file_name"])
    # image = cv2.imread(img_path)
    # image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

