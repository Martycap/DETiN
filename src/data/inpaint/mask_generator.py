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
    Generates the inpainted mask after inpainting is applied.
    """
    cropped_inpaint = inpainted.resize(original.size, Image.LANCZOS)
    
    orig_cv = np.array(original)
    inpaint_cv = np.array(cropped_inpaint)
    gray_orig = cv2.cvtColor(orig_cv, cv2.COLOR_RGB2GRAY)
    gray_inpaint = cv2.cvtColor(inpaint_cv, cv2.COLOR_RGB2GRAY)
    
    difference = cv2.absdiff(gray_orig, gray_inpaint)
    _, mask = cv2.threshold(difference, 30, 255, cv2.THRESH_BINARY)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.dilate(mask, kernel, iterations=1)
    mask = cv2.erode(mask, kernel, iterations=1)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(mask, contours, -1, color=255, thickness=cv2.FILLED)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return Image.fromarray(mask)
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