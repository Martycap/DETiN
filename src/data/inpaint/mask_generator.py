import cv2, random, numpy as np

class Mask:
    def segmentation_mask(coco, id, size = (256, 256)):
        """
        Crea una maschera attraverso la segmentazione
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
        Crea una maschera binaria contenente i rettangoli delle bounding box per un'immagine
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
        Genera una maschera contentente da 3 a 8 rettangoli casuali
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

# -------------------------------------------------------------------------
RATIO = 0.9
class WhiteMask(Exception):
    pass

def check_ratio(mask):
    """
    Se la maschera è troppo grande (=errore) allora lancia l'eccezione.
    """
    white_pixels = np.sum(mask == 256)
    total_pixels = mask.size
    white_ratio = white_pixels / total_pixels
    
    # print(f"White area: {white_ratio:.2%}")
    if white_ratio > RATIO:
        raise WhiteMask()
    return