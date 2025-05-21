from diffusers import DiffusionPipeline

pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-inpainting")

from pycocotools.coco import COCO
import cv2, os
from PIL import Image

coco = COCO("data/raw/annotations/instances_val2017.json")
img_ids = coco.getImgIds()
img_id = img_ids[0]
img_info = coco.loadImgs(img_id)[0]

img_path = os.path.join("data/raw/val_images", img_info["file_name"])
image = cv2.imread(img_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
import matplotlib.pyplot as plt
plt.imshow(image_rgb)
plt.axis('off')  # opzionale, per togliere gli assi
plt.show()
image_resized = cv2.resize(image_rgb, (128, 128))

from mask_generator import Mask
mask = Mask.segmentation_mask(coco, img_id, size=(128, 128))
image_pil = Image.fromarray(image_resized)
mask_pil = Image.fromarray(mask).convert("L")

# ---- Inpainting ----
prompt = "complete the missing regions realistically"
output = pipe(prompt=prompt, image=image_pil, mask_image=mask_pil).images[0]

# ---- Mostra risultato ----
output.show()