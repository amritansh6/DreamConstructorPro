import numpy as np
import torch
import open_clip
from diffusers import DDIMScheduler, StableDiffusionPipeline
import torchvision.transforms as T
from matplotlib import pyplot as plt


class SDS:
    def __init__(self, sd_version="2.1", device="cpu", t_range=[0.02, 0.98], output_dir="output"):
        self.device = device
        self.output_dir = output_dir
        self.sd_model_key = "stabilityai/stable-diffusion-2-1-base"

        self.sd_pipeline = StableDiffusionPipeline.from_pretrained(
                self.sd_model_key, torch_dtype=torch.float32
            ).to(device)

        self.initialize_model_components(t_range)

        print("[INFO] Loaded Stable Diffusion model successfully!")

    def initialize_model_components(self, t_range):
        self.H, self.W = 512, 512  # Default height and width for Stable Diffusion
        self.preprocess = T.Resize((self.H, self.W))
        self.vae = self.sd_pipeline.vae
        self.tokenizer = self.sd_pipeline.tokenizer
        self.text_encoder = self.sd_pipeline.text_encoder
        self.unet = self.sd_pipeline.unet
        self.scheduler = DDIMScheduler.from_pretrained(
            self.sd_model_key, subfolder="scheduler", torch_dtype=torch.float32
        )

        num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.min_step = int(num_train_timesteps * t_range[0])
        self.max_step = int(num_train_timesteps * t_range[1])
        self.alphas = self.scheduler.alphas_cumprod.to(self.device)

    @torch.no_grad()
    def get_text_embeddings(self, prompt):
        text_input = self.tokenizer(prompt, padding="max_length", max_length=self.tokenizer.model_max_length, return_tensors="pt")
        return self.text_encoder(text_input.input_ids.to(self.device))[0]

    def encode_imgs(self, img):
        assert img.shape[-2:] == (self.H, self.W), "Image shape should be 512x512"
        img = 2 * img - 1  # Normalize image from [0, 1] to [-1, 1]
        img = self.preprocess(img)
        latents = self.vae.encode(img).latent_dist.sample() * self.vae.config.scaling_factor
        return latents

    def decode_latents(self, latents):
        latents /= self.vae.config.scaling_factor
        imgs = self.vae.decode(latents).sample
        imgs = (imgs / 2 + 0.5).clamp(0, 1)  # Rescale images from [-1, 1] to [0, 1]
        imgs = imgs.detach().cpu().permute(0, 2, 3, 1).numpy()  # Convert from torch to numpy
        return (imgs * 255).round()[0] # Convert to uint8 format

    def compute_sds_loss(self, latents, text_embeddings, text_embeddings_uncond=None, guidance_scale=100, grad_scale=1):
        t = torch.randint(self.min_step, self.max_step + 1, (1,), dtype=torch.long, device=self.device)
        noise = torch.randn_like(latents)
        latents_noisy = self.scheduler.add_noise(latents, noise, t)
        noise_pred = self.unet(latents_noisy, t, text_embeddings).sample

        if text_embeddings_uncond is not None and guidance_scale != 1:
            noise_pred_uncond = self.unet(latents_noisy, t, text_embeddings_uncond).sample
            noise_pred += guidance_scale * (noise_pred - noise_pred_uncond)

        w = 1 - self.alphas[t]
        gradient = w * (noise_pred - noise)
        latents_target = latents - grad_scale * gradient #SDS Loss
        #img=self.decode_latents(latents_target)


        return ((latents_target - latents) ** 2).mean()

    def batch_sds_loss(self, latents, text_embeddings, text_embeddings_uncond=None, guidance_scale=100, grad_scale=1):
        total_loss = sum(self.compute_sds_loss(latents[i:i + 1], text_embeddings[i:i + 1], text_embeddings_uncond[i:i + 1] if text_embeddings_uncond is not None else None, guidance_scale, grad_scale) for i in range(len(latents)))
        return total_loss / len(latents) if latents.size(0) > 0 else 0
