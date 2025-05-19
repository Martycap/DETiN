from diffusers import StableDiffusionInpaintPipeline
import torch

# Carica il modello

pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting",
    torch_dtype=torch.float16
).to("cuda")

from pycocotools.coco import COCO
import cv2, numpy as np, os, random
from PIL import Image

coco = COCO("data/raw/annotations/instances_val2017.json")
img_ids = coco.getImgIds()
img_id = random.choice(img_ids)
img_info = coco.loadImgs(img_id)[0]

img_path = os.path.join("data/raw/original_images", img_info["file_name"])
image = cv2.imread(img_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image_resized = cv2.resize(image_rgb, (512, 512))

from mask_generator import Mask
mask = Mask.segmentation_mask(coco, img_id, size=(512, 512))
image_pil = Image.fromarray(image_resized)
mask_pil = Image.fromarray(mask).convert("L")

# ---- Inpainting ----
prompt = "complete the missing regions realistically"
output = pipe(prompt=prompt, image=image_pil, mask_image=mask_pil).images[0]

# ---- Mostra risultato ----
output.show()