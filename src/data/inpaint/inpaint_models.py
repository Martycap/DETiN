from diffusers import DiffusionPipeline, StableDiffusionInpaintPipeline
from pycocotools.coco import COCO
from mask_generator import Mask
import cv2, os
from PIL import Image

class Inpaint():
    def __init__(self, path):
        self.diffusion_pipe = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-inpainting",
            use_safetensors = True,
            cache_dir = path
            )
        
        self.controlnet_pipe =  DiffusionPipeline.from_pretrained(
            "alimama-creative/SD3-Controlnet-Inpainting",
            use_safetensors = True,
            cache_dir = path
            ).to("cuda")
        return

    def inference_diffusion(self, prompt, image_pil, mask_pil):
        return self.diffusion_pipe(
            prompt = prompt,
            image = image_pil,
            mask_image = mask_pil
            ).images[0]
    
    def inference_controlnet(self):
        return self.controlnet_pipe(
            prompt = prompt,
            image = image_pil,
            mask_image = mask_pil,
            guidance_scale = 5.0
        ).images[0]

coco = COCO("data/raw/annotations/instances_val2017.json")
img_ids = coco.getImgIds()
img_id = img_ids[0]
img_info = coco.loadImgs(img_id)[0]

img_path = os.path.join("data/raw/val_images", img_info["file_name"])
image = cv2.imread(img_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
import matplotlib.pyplot as plt
plt.imshow(image_rgb)
plt.axis('off')
plt.show()
image_resized = cv2.resize(image_rgb, (256, 256))
mask = Mask.segmentation_mask(coco, img_id, size=(256, 256))
image_pil = Image.fromarray(image_resized)
plt.imshow(image_pil)
plt.axis('off')
plt.show()
mask_pil = Image.fromarray(mask).convert("L")
plt.imshow(mask_pil, cmap="grey")
plt.axis('off')
plt.show()

# ---- Inpainting ----
prompt = "complete the regions realistically"
models = Inpaint("./models/inpaint")
# output = models.inference_diffusion(prompt, image_pil, mask_pil)
# output = models.inference_controlnet(prompt, image_pil, mask_pil)

# # ---- Mostra risultato ----
# output.show()




from transformers import InpaintGenerator
from PIL import Image
import torch

# Carica il modello
generator = InpaintGenerator.from_pretrained("NimaBoscarino/aot-gan-places2")

# Carica immagine e maschera
image = Image.open("immagine.jpg").convert("RGB")         # immagine da inpaintare
mask = Image.open("maschera.png").convert("L")            # maschera bianca dove rimuovere (255), nera dove mantenere (0)

# Esegui l'inpainting
output = generator(image=image, mask_image=mask)

# Il risultato è un oggetto PIL.Image
output.save("output_inpainted.jpg")
output.show()