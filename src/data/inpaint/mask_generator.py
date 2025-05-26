import cv2, random, numpy as np
from PIL import Image

class Mask:
    def segmentation_mask(coco, id, size = (256, 256)):
        """
        Generates a mask using segmentation info from annotations.
        """
        
        ann_ids = coco.getAnnIds(imgIds = id)
        print(ann_ids)
        annotations = coco.loadAnns(ann_ids)
        mask = np.zeros(size, dtype=np.uint8)

        selections = random.randint(1, 3)
        selected = annotations[:selections]
        print(selected)
        for annotation in selected:
            rle = coco.annToMask(annotation)
            rle_resized = cv2.resize(rle.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST)
            mask = cv2.bitwise_or(mask, (rle_resized * 255).astype(np.uint8))
        
        check_ratio(mask)
        return mask


    def bbox_mask(coco, image_id, size=(256, 256)):
        """
        Generates a binary mask with bounding box rectangles for a given image.
        """
        
        ann_ids = coco.getAnnIds(imgIds=image_id)
        anns = coco.loadAnns(ann_ids)
        mask = np.zeros(size, dtype=np.uint8)
        
        img_info = coco.loadImgs(image_id)[0]
        height, width = img_info['height'], img_info['width']

        selections = random.randint(1, 3)
        selected = anns[:selections]
        for ann in selected:
            x, y, w, h = ann['bbox']
            box_mask = np.zeros((height, width), dtype=np.uint8)
            x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
            cv2.rectangle(box_mask, (x1, y1), (x2, y2), 255, -1)
            box_mask_resized = cv2.resize(box_mask, size, interpolation=cv2.INTER_NEAREST)
            mask = cv2.bitwise_or(mask, box_mask_resized)
            
        check_ratio(mask)
        return mask


    def random_box_mask(size=(256, 256)):
        """
        Creates a mask with a random number of rectangles, ranging from 1 to 3.
        """
        
        mask = np.zeros(size, dtype=np.uint8)
        selections = random.randint(1, 3)
        for _ in range(selections):
            x1 = random.randint(0, size[1] - 1)
            y1 = random.randint(0, size[0] - 1)
            x2 = random.randint(x1, size[1])
            y2 = random.randint(y1, size[0])
            cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
        
        check_ratio(mask)
        return mask


    def compute_inpainted_mask(original, inpainted):
        """
        Generates the inpainted mask after inpainted is applied.
        """
        
        if inpainted.size != original.size:
            inpainted = inpainted.resize(original.size, Image.LANCZOS)
        
        original = np.array(original).astype(np.int16)
        inpainted = np.array(inpainted).astype(np.int16)
        differences = np.abs(inpainted - original).sum(axis = 2)
        
        kernel = np.ones((3,3), np.uint8)
        mask = (differences > 40).astype(np.uint8) * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        return Image.fromarray(mask, mode='L')
    
# -------------------------------------------------------------------------
RATIO = 0.9
class WhiteMask(Exception):
    pass

def check_ratio(mask):
    """
    Throws an exception if the mask exceeds the allowed size.
    """
    
    white_pixels = np.sum(mask == 256)
    total_pixels = mask.size
    white_ratio = white_pixels / total_pixels

    if white_ratio > RATIO:
        raise WhiteMask()
    return