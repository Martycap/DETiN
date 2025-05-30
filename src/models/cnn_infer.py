import os
import tqdm
from PIL import Image
import torch


def infer_and_save_masks(model, dataloader, device, output_dir):
    """
    Runs inference on a dataset using a segmentation model and saves the predicted binary masks as PNG images.

    """
    os.makedirs(output_dir, exist_ok=True)
    model.eval()
    with torch.no_grad():
        for batch in tqdm.tqdm(dataloader):
            inputs, filenames = batch
            inputs = inputs.to(device)
            outputs = model(inputs)
            outputs = outputs.squeeze(1).cpu()  

            for i in range(outputs.size(0)):
                mask = outputs[i].numpy()
                mask_img = (mask > 0.5).astype('uint8') * 255  
                pil_mask = Image.fromarray(mask_img)
                filename_no_ext = os.path.splitext(filenames[i])[0]
                save_path = os.path.join(output_dir, filename_no_ext.replace(" ", "_") + ".png")
                pil_mask.save(save_path)
                