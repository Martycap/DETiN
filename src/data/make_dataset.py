import random, sys, os
sys.path.append(os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")))
from data.inpaint.inpaint_models import Inpaint
from data.inpaint.mask_generator import Mask
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

if __name__ == "__main__":
    dataset = COCO("data/raw/annotations/instances_val2017.json")
    index = dataset.getImgIds()

    random.shuffle(index)
    for image, id, file_name in image_loader(dataset, index):
        mask = Mask.segmentation_mask(dataset, id)
        mask_pil = Image.fromarray(mask).convert("L")
        
        image.show()
        mask_pil.show()

        break
    
    # images[256].show()
    
    # mask = Mask.random_box_mask()
    
    # models = Inpaint("./models/inpaint")
    
    # from PIL import Image
    # image = Image.fromarray(images[342])
    # mask = Image.fromarray(mask).convert("L")
    # plt.imshow(mask, cmap = "gray")
    # plt.axis('off')
    # plt.show()
    
    
    # output = models.inference_kandinsky(prompt, images, mask)
    
    # output.show()
    
    # img_id = random.choice(index)
    # img_info = dataset.loadImgs(img_id)[0]
    # print(img_info)

    # annotations = dataset.loadAnns(dataset.getAnnIds(imgIds=img_id))
    # for ann in annotations:
    #     print(ann)
    
    # img_path = os.path.join("data/raw/val_images", img_info["file_name"])
    # image = cv2.imread(img_path)
    # image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

