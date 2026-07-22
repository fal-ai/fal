# /// script
# dependencies = [
#   "diffusers",
#   "torch",
#   "torchvision"
# ]
# ///

import torch
from diffusers import AutoModel
from PIL import Image
import torchvision.transforms.functional as F


device = torch.device("mps")
tiny_vae = AutoModel.from_pretrained(
    "fal/FLUX.2-Tiny-AutoEncoder", trust_remote_code=True, torch_dtype=torch.bfloat16
).to(device)

image_path = "/Users/vedat/Downloads/fBDJwWVdQNHleNvDZ7QEx_QgwoQeBx.png"

pil_image = Image.open(image_path)
image_tensor = F.to_tensor(pil_image)
image_tensor = image_tensor.unsqueeze(0) * 2.0 - 1.0
image_tensor = image_tensor.to(device, dtype=tiny_vae.dtype)
with torch.inference_mode():
    latents = tiny_vae.encode(image_tensor, return_dict=False)
    recon = tiny_vae.decode(latents, return_dict=False)
    recon = recon.squeeze(0).clamp(-1, 1) / 2.0 + 0.5
    recon = recon.float().detach().cpu()
print(latents.shape)
torch.save(latents.clone(), "latents.pt")
recon_image = F.to_pil_image(recon)
recon_image.save("reconstituted.png")
