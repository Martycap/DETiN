import cv2, random, numpy as np

class Mask:
    def segmentation_mask(coco, id, size = (255, 255)):
        """
        Crea una maschera attraverso la segmentazione
        """
        ann_ids = coco.getAnnIds(imgIds = id)
        print(ann_ids)
        annotations = coco.loadAnns(ann_ids)
        mask = np.zeros(size, dtype=np.uint8)

        selected = annotations[:3]
        print(selected)
        for annotation in selected:
            rle = coco.annToMask(annotation)
            rle_resized = cv2.resize(rle.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST)
            mask = cv2.bitwise_or(mask, (rle_resized * 255).astype(np.uint8))
        
        check_ratio()
        return mask


    def bbox_mask(coco, image_id, size=(255, 255), num_boxes=3):
        """
        Crea una maschera binaria contenente i rettangoli delle bounding box per un'immagine
        """
        ann_ids = coco.getAnnIds(imgIds=image_id)
        anns = coco.loadAnns(ann_ids)
        mask = np.zeros(size, dtype=np.uint8)
        
        img_info = coco.loadImgs(image_id)[0]
        height, width = img_info['height'], img_info['width']

        selected = anns[:num_boxes]
        for ann in selected:
            x, y, w, h = ann['bbox']
            box_mask = np.zeros((height, width), dtype=np.uint8)
            x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
            cv2.rectangle(box_mask, (x1, y1), (x2, y2), 255, -1)
            box_mask_resized = cv2.resize(box_mask, size, interpolation=cv2.INTER_NEAREST)
            mask = cv2.bitwise_or(mask, box_mask_resized)
            
        check_ratio()
        return mask


    def random_box_mask(size=(512, 512)):
        """
        Genera una maschera contentente da 3 a 8 rettangoli casuali
        """
        mask = np.zeros(size, dtype=np.uint8)
        for _ in range(random.randint(3, 8)):
            x1 = random.randint(0, size[1] - 1)
            y1 = random.randint(0, size[0] - 1)
            x2 = random.randint(x1, size[1])
            y2 = random.randint(y1, size[0])
            cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
        
        check_ratio()
        return mask

# -------------------------------------------------------------------------
RATIO = 0.9
class WhiteMask(Exception):
    pass

def check_ratio(mask):
    """
    Se la maschera è troppo grande (=errore) allora lancia l'eccezione.
    """
    white_pixels = np.sum(mask == 255)
    total_pixels = mask.size
    white_ratio = white_pixels / total_pixels
    
    # print(f"White area: {white_ratio:.2%}")
    if white_ratio > RATIO:
        raise WhiteMask()
    return

# -------------------------------------------------------------------------
# from pycocotools.coco import COCO
# import os, matplotlib.pyplot as plt

# dataset = COCO("data/raw/annotations/instances_val2017.json")
# img_ids = dataset.getImgIds()
# random_id = random.choice(img_ids)
# img_info = dataset.loadImgs(random_id)[0]

# img_path = os.path.join("data/raw/val_images", img_info['file_name'])
# image = cv2.imread(img_path)
# image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# image_resized = cv2.resize(image_rgb, (255, 255), interpolation=cv2.INTER_AREA)

# mask = segmentation_mask(dataset, img_info["id"])

# plt.figure(figsize=(10, 5))
# plt.subplot(1, 2, 1)
# plt.title("Immagine COCO")
# plt.imshow(image_resized)
# plt.axis("off")

# plt.subplot(1, 2, 2)
# plt.title("Maschera di Segmentazione")
# plt.imshow(mask, cmap="gray")
# plt.axis("off")

# plt.tight_layout()
# plt.show()