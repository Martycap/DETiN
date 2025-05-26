from diffusers import DiffusionPipeline
from data.inpaint.mask_generator import Mask
import torch

class Inpaint():
    def __init__(self, path):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.diffusion_pipe = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-inpainting",
            torch_dtype=torch.float32,
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
        """
        Stable Diffusion model's inference.
        """
        
        inpainted = self.diffusion_pipe(
            prompt = prompt,
            image = image,
            mask_image = mask
            ).images[0]
        
        gt = Mask.compute_inpainted_mask(image, inpainted)
        return inpainted, gt
    
    
    def Kandinsky(self, prompt, image, mask):
        """
        Kandinsky model's inference.
        """
        
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
        
        gt = Mask.compute_inpainted_mask(image, inpainted)
        return inpainted, gt
