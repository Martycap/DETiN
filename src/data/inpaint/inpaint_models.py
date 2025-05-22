from diffusers import DiffusionPipeline
from pycocotools.coco import COCO
from mask_generator import Mask
import cv2, os
import torch
from PIL import Image

class Inpaint():
    def __init__(self, path):
        device = torch.device(
            "cuda" if not torch.cuda.is_available() else "cpu"
            )
        
        self.diffusion_pipe = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-inpainting",
            use_safetensors = True,
            cache_dir = path
            ).to(device)
        
        self.decoder = DiffusionPipeline.from_pretrained(
            "kandinsky-community/kandinsky-2-2-decoder-inpaint",
            torch_dtype=torch.float32,
            cache_dir=path
        ).to(device)
        
        self.prior = DiffusionPipeline.from_pretrained(
            "kandinsky-community/kandinsky-2-2-prior",
            torch_dtype=torch.float32,
            cache_dir=path
        ).to(device)
        return

    def inference_diffusion(self, prompt, image_pil, mask_pil):
        return self.diffusion_pipe(
            prompt = prompt,
            image = image_pil,
            mask_image = mask_pil
            ).images[0]
    
    def inference_kandinsky(self, prompt, image_pil, mask_pil):
        prior = self.prior(prompt=prompt, guidance_scale=1.0)
        image_embeds = prior.image_embeds
        negative_image_embeds = prior.negative_image_embeds

        output = self.decoder(
            prompt=prompt,
            image_embeds=image_embeds,
            negative_image_embeds=negative_image_embeds,
            image=image_pil,
            mask_image=mask_pil,
            guidance_scale=4.0,
            num_inference_steps=50
        ).images[0]

        return output



coco = COCO("data/raw/annotations/instances_val2017.json")
img_ids = coco.getImgIds()
img_id = img_ids[3000]
img_info = coco.loadImgs(img_id)[0]

img_path = os.path.join("data/raw/val_images", img_info["file_name"])
image = cv2.imread(img_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image_resized = cv2.resize(image_rgb, (256, 256))
mask = Mask.bbox_mask(coco, img_id, size=(256, 256))
image_pil = Image.fromarray(image_resized)
mask_pil = Image.fromarray(mask).convert("L")

import matplotlib.pyplot as plt
plt.imshow(image_rgb)
plt.axis('off')
plt.show()

#
result = cv2.inpaint(image_resized, mask, 3, cv2.INPAINT_NS)
plt.imshow(result)
plt.axis('off')
plt.show()

# # ---- Inpainting ----
# prompt = "a cat on the table and a dog with hat"
# models = Inpaint("./models/inpaint")
# output = models.inference_kandinsky(prompt, image_pil, mask_pil)

# # # ---- Mostra risultato ----
# output.show()