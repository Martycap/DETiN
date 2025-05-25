from diffusers import DiffusionPipeline
import torch, numpy as np, cv2
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


    def Stable_Diffusion(self, prompt, image, mask):
        inpainted = self.diffusion_pipe(
            prompt = prompt,
            image = image,
            mask_image = mask
            ).images[0]
        
        gt = Inpaint.compute_inpainted_mask(image, inpainted)
        return inpainted, gt
    
    
    def Kandinsky(self, prompt, image, mask):
        prior = self.prior(prompt=prompt, guidance_scale=1.0)
        image_embeds = prior.image_embeds
        negative_image_embeds = prior.negative_image_embeds

        inpainted = self.decoder(
            prompt = prompt,
            image_embeds = image_embeds,
            negative_image_embeds = negative_image_embeds,
            image = image,
            mask_image = mask,
            guidance_scale = 4.0,
            num_inference_steps = 50
        ).images[0]
        
        gt = Inpaint.compute_inpainted_mask(image, inpainted)
        return inpainted, gt
    
    
    def compute_inpainted_mask(original, inpainted):
        original = np.array(original).astype(np.int16)
        inpainted = np.array(inpainted).astype(np.uint16)
        differences = np.abs(inpainted - original).sum(axis = 2)
        
        kernel = np.ones((3,3), np.uint8)
        mask = (differences > 0).astype(np.uint8) * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        return Image.fromarray(mask, mode='L')
    