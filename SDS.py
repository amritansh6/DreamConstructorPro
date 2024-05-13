import torch
import torch.nn.functional as F
from diffusers import DDIMScheduler, StableDiffusionPipeline
import open_clip
import torchvision.transforms as T

class CLIP:
    """
    Class to implement the SDS loss function.
    """

    def __init__(
        self,
        device="cpu",
        output_dir="output",
    ):
        """
        Load the Stable Diffusion model and set the parameters.

        Args:
            sd_version (str): version for stable diffusion model
            device (_type_): _description_
        """

        # Set parameters
        self.H = 224  # default height of CLIP
        self.W = 224  # default width of CLIP
        self.output_dir = output_dir
        self.device = device

        # Set the open_clip model key based on the version
        model, _, _ = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
        self.preprocess = T.Compose([T.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
                                     T.Resize((self.H, self.W))])
        self.tokenizer = open_clip.get_tokenizer('ViT-B-32')
        self.model = model.to(device)

        print(f"[INFO] loaded OpenClip!")

    @torch.no_grad()
    def get_text_embeddings(self, prompt):
        """
        Get the text embeddings for the prompt.

        Args:
            prompt (list of string): text prompt to encode.
        """
        return self.model.encode_text(self.tokenizer(prompt).to(self.device))

    def encode_imgs(self, image):
        """
        Encode images to latent representation.

        Args:
            img (tensor): image to encode. shape (N, 3, H, W), range [0, 1]

        Returns:
            latents (tensor): latent representation. shape (N, 512)
        """

        image = self.preprocess(image)

        # Encode the rendered image to latents
        image_embeddings = self.model.encode_image(image)

        return image_embeddings 

    def clip_loss(
        self,
        imgs,
        text_embeddings,
        text_embeddings_uncond=None
    ):
        """
        Compute the SDS loss.

        Args:
            imgs (tensor): input latents, shape [N, H, W, 3]
            text_embeddings (tensor): conditional text embedding (for positive prompt), shape [1, 77, 1024]
            text_embeddings_uncond (tensor, optional): unconditional text embedding (for negative prompt), shape [1, 77, 1024]. Defaults to None.

        Returns:
            loss (tensor): CLIP loss
        """
        image_embeddings = self.encode_imgs(imgs)
        # Compute the loss
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
        """
        Compute the SDS loss.

        Args:
            imgs (tensor): input latents, shape [N, H, W, 3]
            text_embeddings (tensor): conditional text embedding (for positive prompt), shape [1, 77, 1024]
            text_embeddings_uncond (tensor, optional): unconditional text embedding (for negative prompt), shape [1, 77, 1024]. Defaults to None.

        Returns:
            loss (tensor): CLIP loss
        """
        image_embeddings = self.encode_imgs(imgs)
        # Compute the loss
        image_embeddings = image_embeddings / image_embeddings.norm(dim=-1, keepdim=True)
        text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)

        text_probs = (image_embeddings @ text_embeddings.T).mean(0)
            
        return text_probs