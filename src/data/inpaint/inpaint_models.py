from diffusers import DiffusionPipeline
from pycocotools.coco import COCO
from mask_generator import Mask
import cv2, os
import torch
from PIL import Image

class Inpaint():
    def __init__(self, path):
        # self.diffusion_pipe = DiffusionPipeline.from_pretrained(
        #     "stabilityai/stable-diffusion-2-inpainting",
        #     use_safetensors = True,
        #     cache_dir = path
        #     )
        
        self.prior =  DiffusionPipeline.from_pretrained(
            "kandinsky-community/kandinsky-2-2-decoder-inpaint",
            cache_dir = path
            )
        
        self.decoder = DiffusionPipeline.from_pretrained(
            "kandinsky-community/kandinsky-2-2-decoder-inpaint",
            torch_dtype=torch.float32,
            cache_dir=path
        )
        
        self.prior = DiffusionPipeline.from_pretrained(
            "kandinsky-community/kandinsky-2-2-prior",
            torch_dtype=torch.float32,
            cache_dir=path
        )
                
        return

    def inference_diffusion(self, prompt, image_pil, mask_pil):
        return self.diffusion_pipe(
            prompt = prompt,
            image = image_pil,
            mask_image = mask_pil
            ).images[0]
    
    def inference_kandinsky(self, prompt, image_pil, mask_pil):
        # Step 1: usa il prior per ottenere gli embedding
        prior_output = self.prior(prompt=prompt, guidance_scale=1.0)

        image_embeds = prior_output.image_embeds
        negative_image_embeds = prior_output.negative_image_embeds

        # Step 2: usa il decoder per generare l'immagine inpaintata
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
prompt = "a cat on the table and a showgirl"
models = Inpaint("./models/inpaint")
output = models.inference_kandinsky(prompt, image_pil, mask_pil)

# # ---- Mostra risultato ----
output.show()




# from transformers import InpaintGenerator
# from PIL import Image
# import torch

# # Carica il modello
# generator = InpaintGenerator.from_pretrained("NimaBoscarino/aot-gan-places2")

# # Carica immagine e maschera
# image = Image.open("immagine.jpg").convert("RGB")         # immagine da inpaintare
# mask = Image.open("maschera.png").convert("L")            # maschera bianca dove rimuovere (255), nera dove mantenere (0)

# # Esegui l'inpainting
# output = generator(image=image, mask_image=mask)

# # Il risultato è un oggetto PIL.Image
# output.save("output_inpainted.jpg")
# output.show()