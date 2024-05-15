import torch
import torch.nn.functional as F
from diffusers import DDIMScheduler, StableDiffusionPipeline
import open_clip
import torchvision.transforms as T

class CLIP:

    def __init__(
        self,
        device="cpu",
        output_dir="output",
    ):
        self.H = 224
        self.W = 224
        self.output_dir = output_dir
        self.device = device
        model, _, _ = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
        self.preprocess = T.Compose([T.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
                                     T.Resize((self.H, self.W))])
        self.tokenizer = open_clip.get_tokenizer('ViT-B-32')
        self.model = model.to(device)

        print(f"[INFO] loaded OpenClip!")

    @torch.no_grad()
    def get_text_embeddings(self, prompt):
        return self.model.encode_text(self.tokenizer(prompt).to(self.device))

    def encode_imgs(self, image):
        image = self.preprocess(image)
        image_embeddings = self.model.encode_image(image)
        return image_embeddings 

    def clip_loss(
        self,
        imgs,
        text_embeddings,
        text_embeddings_uncond=None
    ):
        image_embeddings = self.encode_imgs(imgs)
        image_embeddings = image_embeddings / image_embeddings.norm(dim=-1, keepdim=True)
        text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)

        if text_embeddings_uncond is not None:
            text_embeddings_uncond = text_embeddings_uncond / text_embeddings_uncond.norm(dim=-1, keepdim=True)
            text_embeddings = torch.cat([text_embeddings, text_embeddings_uncond])
            text_probs = (image_embeddings @ text_embeddings.T).mean(0)
            loss = -text_probs[0] + text_probs[1:].mean()
        else:
            text_probs = (image_embeddings @ text_embeddings.T).mean(0)
            loss = -text_probs[0]
            
        return loss

    def clip_score(
        self,
        imgs,
        text_embeddings
    ):
        image_embeddings = self.encode_imgs(imgs)
        image_embeddings = image_embeddings / image_embeddings.norm(dim=-1, keepdim=True)
        text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)

        text_probs = (image_embeddings @ text_embeddings.T).mean(0)
            
        return text_probs