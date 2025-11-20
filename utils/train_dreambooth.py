import os
import gc
import shutil
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from torch import nn
from PIL import Image
from transformers import AutoProcessor, BlipForConditionalGeneration

from diffusers import StableDiffusionImg2ImgPipeline

# For inpainting we try the legacy pipeline if available, otherwise fall back to the new one.
try:
    from diffusers import StableDiffusionInpaintPipelineLegacy as StableDiffusionInpaintPipeline
except ImportError:
    from diffusers import StableDiffusionInpaintPipeline  # type: ignore

from peft import PeftModel, LoraConfig
from tqdm import tqdm
import matplotlib.pyplot as plt


from utils.perceptual import PerceptualLoss


# -------------------------------
#  Helper functions
# -------------------------------
def resize_shape( img, short=True):
    device = img.device
    si = 512
    if device.type == 'cpu':
        img = img.to('cuda')
    if short:
        if img.shape[2] < img.shape[3]:
            img = torch.nn.functional.interpolate(img, size=(si, int(si*img.shape[3]/img.shape[2])), mode='bilinear', align_corners=False)
        else:
            img = torch.nn.functional.interpolate(img, size=(int(si*img.shape[2]/img.shape[3]), si), mode='bilinear', align_corners=False)
    else:
        if img.shape[2] < img.shape[3]:
            img = torch.nn.functional.interpolate(img, size=(int(img.shape[2]/img.shape[3]*si), si), mode='bilinear', align_corners=False)
        else:
            img = torch.nn.functional.interpolate(img, size=(si, int(img.shape[3]/img.shape[2]*si)), mode='bilinear', align_corners=False)

    if device.type == 'cpu':
        img = img.to('cpu')
    return img

def image_grid(imgs: List[Image.Image], rows: int, cols: int, resize: int = 256) -> Image.Image:
    """Make a simple image grid (for quick visualization / debugging)."""
    if resize is not None:
        imgs = [img.resize((resize, resize)) for img in imgs]
    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


def load_images(img_folder: str) -> List[Image.Image]:
    """Load all .png images in a folder as RGB PIL Images."""
    img_paths = [
        os.path.join(img_folder, img)
        for img in os.listdir(img_folder)
        if img.lower().endswith(".png")
    ]
    imgs = [Image.open(p).convert("RGB") for p in img_paths]
    return imgs


def caption_images(imgs: List[Image.Image]) -> List[str]:
    """
    Caption a list of PIL images using BLIP.
    Used to auto-generate the DreamBooth instance/class prompts.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    generated_captions: List[str] = []

    blip_processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    blip_model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    ).to(device)

    for input_image in imgs:
        inputs = blip_processor(images=input_image, return_tensors="pt").to(
            device,
            torch.float16 if device == "cuda" else torch.float32,
        )
        pixel_values = inputs.pixel_values

        generated_ids = blip_model.generate(pixel_values=pixel_values, max_length=50)
        generated_caption = blip_processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0]
        generated_captions.append(generated_caption)

    # free VRAM
    del blip_processor, blip_model
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    return generated_captions


# -------------------------------
#  Custom SD Img2Img pipeline (feature access)
# -------------------------------

class CustomStableDiffusion(StableDiffusionImg2ImgPipeline):
    """
    Extension of StableDiffusionImg2ImgPipeline that exposes a get_img_feature()
    method used for similarity-based editing (cal_sim / cal_p_ssim).
    """

    @torch.no_grad()
    def get_img_feature(
        self,
        prompt: Union[str, List[str]] = None,
        image: Any = None,
        strength: float = 0.01,
        num_inference_steps: Optional[int] = 50,
        timesteps: List[int] = None,
        sigmas: List[float] = None,
        guidance_scale: Optional[float] = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: Optional[float] = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        ip_adapter_image: Any = None,
        ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        clip_skip: int = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        **kwargs,
    ):
        """
        Returns the UNet noise prediction (feature-like tensor) at the first denoising step.
        This is used as a pseudo feature map for similarity computations.
        """

        self._guidance_scale = guidance_scale
        self._clip_skip = clip_skip
        self._cross_attention_kwargs = cross_attention_kwargs
        self._interrupt = False

        # 1. batch size
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        # 2. Encode input prompt
        text_encoder_lora_scale = (
            self.cross_attention_kwargs.get("scale", None)
            if self.cross_attention_kwargs is not None
            else None
        )

        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            self.do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=text_encoder_lora_scale,
            clip_skip=self.clip_skip,
        )

        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        # 3. IP-Adapter image embeds (if any)
        if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
            image_embeds = self.prepare_ip_adapter_image_embeds(
                ip_adapter_image,
                ip_adapter_image_embeds,
                device,
                batch_size * num_images_per_prompt,
                self.do_classifier_free_guidance,
            )
        else:
            image_embeds = None

        # 4. Preprocess image to latents
        image = self.image_processor.preprocess(image)

        # 5. Set timesteps using the pipeline’s built-in helper
        timesteps, num_inference_steps = self.get_timesteps(
            num_inference_steps, strength, device
        )
        latent_timestep = timesteps[:1].repeat(batch_size * num_images_per_prompt)

        # 6. Prepare latents
        latents = self.prepare_latents(
            image,
            latent_timestep,
            batch_size,
            num_images_per_prompt,
            prompt_embeds.dtype,
            device,
            generator,
        )

        # 7. Extra scheduler kwargs
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 8. Add IP-Adapter cond kwargs
        added_cond_kwargs = (
            {"image_embeds": image_embeds}
            if image_embeds is not None
            else None
        )

        # 9. Optional timestep_cond (for time_cond_proj_dim)
        timestep_cond = None
        if self.unet.config.time_cond_proj_dim is not None:
            guidance_scale_tensor = torch.tensor(self.guidance_scale - 1).repeat(
                batch_size * num_images_per_prompt
            )
            timestep_cond = self.get_guidance_scale_embedding(
                guidance_scale_tensor,
                embedding_dim=self.unet.config.time_cond_proj_dim,
            ).to(device=device, dtype=latents.dtype)

        # We only really need the first denoising step's prediction as a “feature”.
        t = timesteps[0]

        # expand latents for CFG
        latent_model_input = (
            torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
        )
        latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

        # UNet noise prediction
        noise_pred = self.unet(
            latent_model_input,
            t,
            encoder_hidden_states=prompt_embeds,
            timestep_cond=timestep_cond,
            cross_attention_kwargs=self.cross_attention_kwargs,
            added_cond_kwargs=added_cond_kwargs,
            return_dict=False,
        )[0]

        return noise_pred


# -------------------------------
#  LoRA + SD pipeline loader
# -------------------------------

def get_lora_sd_pipeline(
    ckpt_dir: str,
    base_model_name_or_path: Optional[str] = None,
    dtype: torch.dtype = torch.float16,
    device: str = "cuda",
    adapter_name: str = "default",
    mask: bool = False,
):
    """
    Load a Stable Diffusion (img2img or inpaint) pipeline and apply LoRA adapters
    from a checkpoint directory created by the DreamBooth LoRA training script.
    """
    unet_sub_dir = os.path.join(ckpt_dir, "unet")
    text_encoder_sub_dir = os.path.join(ckpt_dir, "text_encoder")

    if os.path.exists(text_encoder_sub_dir) and base_model_name_or_path is None:
        config = LoraConfig.from_pretrained(text_encoder_sub_dir)
        base_model_name_or_path = config.base_model_name_or_path

    if base_model_name_or_path is None:
        raise ValueError("Please specify the base model name or path")

    # Use img2img or inpaint variant
    if not mask:
        pipe = CustomStableDiffusion.from_pretrained(
            base_model_name_or_path,
            torch_dtype=dtype,
            safety_checker=None,
        ).to(device)
    else:
        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            base_model_name_or_path,
            torch_dtype=dtype,
            safety_checker=None,
        ).to(device)

    # Load LoRA on UNet
    pipe.unet = PeftModel.from_pretrained(pipe.unet, unet_sub_dir, adapter_name=adapter_name)

    # Optional LoRA on text encoder
    if os.path.exists(text_encoder_sub_dir) and hasattr(pipe, "text_encoder"):
        pipe.text_encoder = PeftModel.from_pretrained(
            pipe.text_encoder, text_encoder_sub_dir, adapter_name=adapter_name
        )

    if dtype in (torch.float16, torch.bfloat16):
        pipe.unet.half()
        if hasattr(pipe, "text_encoder") and pipe.text_encoder is not None:
            pipe.text_encoder.half()

    pipe.to(device)
    return pipe


# -------------------------------
#  DreamBooth wrapper used by train_drag.py
# -------------------------------

class DreamBooth:
    """
    Minimal DreamBooth wrapper:

    - train_dreambooth(img_folder): run LoRA DreamBooth (via accelerate).
    - img2img(...): apply the trained LoRA SD pipeline as an editor.
    """

    def __init__(self):
        self.init = False
        self.caption: Optional[str] = None
        self.save_dir: Optional[str] = None
        self.p_ssim: Optional[nn.Module] = None

    # -------- DreamBooth training --------
    def train_dreambooth(self, img_folder: str):
        save_root = "./dreambooth"
        os.makedirs(save_root, exist_ok=True)
        imgs = load_images(img_folder)

        if len(imgs) == 0:
            raise ValueError(f"No .png images found in {img_folder}")

        # seed everything
        torch.manual_seed(0)
        np.random.seed(0)

        # Use BLIP caption on the first image to define the concept
        self.caption = caption_images([imgs[0]])[0]
        self.save_dir = os.path.join(save_root, self.caption)

        unet_dir = os.path.join(self.save_dir, "unet")
        # print(f"Training DreamBooth LoRA for prompt: {self.caption}")
        if not os.path.exists(unet_dir):
            class_images_dir = os.path.join(self.save_dir, "class_images")
            instance_prompt = f"{self.caption} sks"
            class_prompt = f"{self.caption} with diverse situation, shape and  motion"

            cmd = (
                f'accelerate launch utils/train_dreambooth_lora.py '
                f'--pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" '
                f'--instance_data_dir="{img_folder}" '
                f'--output_dir="{self.save_dir}" '
                f'--mixed_precision="fp16" '
                f'--train_text_encoder '
                f'--instance_prompt="{instance_prompt}" '
                f'--class_prompt="{class_prompt}" '
                f'--num_dataloader_workers=4 '
                f'--resolution=512 '
                f'--train_batch_size=1 '
                f'--lr_scheduler="constant" '
                f'--lr_warmup_steps=0 '
                f'--lora_text_encoder_r 16 '
                f'--lora_text_encoder_alpha 17 '
                f'--learning_rate=1e-4 '
                f'--use_8bit_adam '
                f'--max_train_steps=200 '
                f'--num_class_images=100 '
                f'--class_data_dir="{class_images_dir}" '
                f'--with_prior_preservation '
                f'--use_lora '
                f'--seed="0"'
            )
            print(f"Running DreamBooth training with command: {cmd}")
            os.system(cmd)

        # Mirror to ./tmp for later use
        if os.path.exists("./tmp"):
            shutil.rmtree("./tmp")
        shutil.copytree(save_root, "./tmp/")

        self.save_dir = os.path.join("./tmp", self.caption)
        self.p_ssim = PerceptualLoss().eval().to("cuda")

    # -------- Editing used in train_drag.py --------
    @torch.no_grad()
    def img2img(
        self,
        imgs: torch.Tensor,
        strength: float = 0.5,
        guidance_scale: float = 7.5,
        start: Optional[List[torch.Tensor]] = None,
        end: Optional[List[torch.Tensor]] = None,
        imgs2: Optional[torch.Tensor] = None,
        add_prompt: str = "",
        mask: Optional[List[torch.Tensor]] = None,
    ):
        """
        Core editing function used by train_drag.py.

        Args:
            imgs: Tensor (B, C, H, W) in [0,1].
            strength, guidance_scale: SD img2img parameters.
            start, end, imgs2: optional patch-based similarity inputs.
            add_prompt: extra text appended to the base caption.
            mask: optional list of [H,W] tensors; if given, we use inpaint pipeline.
        """
        assert self.save_dir is not None, "Call train_dreambooth() first."
        assert self.caption is not None, "Caption not initialized."
        assert self.p_ssim is not None, "PerceptualLoss not initialized."

        original_shape = imgs.shape  # (B, C, H, W)

        # Prepare mask to SD resolution if present
        if mask is not None:
            mask_tensor = torch.stack(mask, 0).squeeze().unsqueeze(1).float()
            mask_tensor = resize_shape(mask_tensor, True)
        else:
            mask_tensor = None

        # Resize imgs to SD resolution
        imgs_resized = resize_shape(imgs, True).cpu()
        img_list: List[torch.Tensor] = []
        weight_list: List[float] = []

        # Load LoRA SD pipeline
        pipe = get_lora_sd_pipeline(
            self.save_dir,
            mask=True if mask_tensor is not None else False,
        )
        pipe.set_progress_bar_config(disable=True)

        prompt = f"{self.caption} sks"
        if add_prompt:
            prompt = prompt + " " + add_prompt
        negative_prompt = (
            "low quality, blurry, unfinished, artifacts, wired shapes, wired human bodies"
        )

        for i, img in tqdm(enumerate(imgs_resized), desc="Editing images", total=len(imgs_resized)):
            # (C,H,W) -> PIL
            img_np = img.numpy().transpose(1, 2, 0)
            img_np = np.clip(img_np, 0, 1)
            pil_img = Image.fromarray((img_np * 255).astype(np.uint8))

            if mask_tensor is None:
                out = pipe(
                    prompt=prompt,
                    image=pil_img,
                    strength=strength,
                    guidance_scale=guidance_scale,
                    negative_prompt=negative_prompt,
                    num_inference_steps=50,
                )
                out_img = out.images[0]
            else:
                t_mask = mask_tensor[i].squeeze()[None, None, :, :].float()
                t_mask = 1 - t_mask  # invert mask as in original code
                out = pipe(
                    prompt=prompt,
                    image=pil_img,
                    strength=strength,
                    guidance_scale=guidance_scale,
                    negative_prompt=negative_prompt,
                    num_inference_steps=30,
                    mask_image=t_mask,
                )
                out_img = out.images[0]

            # Resize back to original resolution
            out_img = out_img.resize((original_shape[3], original_shape[2]))
            out_tensor = torch.tensor(np.array(out_img)).permute(2, 0, 1) / 255.0

            # Optional similarity-based weight
            if start is not None and end is not None and imgs2 is not None:
                img2 = imgs2[i]
                sim = self.cal_p_ssim(img2, out_tensor, start[i], end[i])
                weight_list.append(sim)

            img_list.append(out_tensor)

        if start is not None and end is not None and len(weight_list) > 0:
            weight_tensor = torch.tensor(weight_list)
            weight_tensor = -weight_tensor
            pos_mask = weight_tensor > 0
            if pos_mask.any():
                max_val = weight_tensor[pos_mask].max()
                weight_tensor[~pos_mask] = -1000
                weight_tensor[pos_mask] = weight_tensor[pos_mask] / max_val
                weight_tensor = weight_tensor * 2 - 1
                weight_tensor = 1 / (1 + torch.exp(-weight_tensor))
            else:
                weight_tensor = torch.ones_like(weight_tensor)
        else:
            weight_tensor = torch.ones(len(img_list))

        imgs_out = torch.stack(img_list, 0).cuda()

        del pipe
        torch.cuda.empty_cache()

        return imgs_out.cpu(), weight_tensor.cuda()

    # -------- Patch-wise perceptual similarity --------
    def cal_p_ssim(
        self,
        img1: torch.Tensor,
        img2: torch.Tensor,
        coords1: torch.Tensor,
        coords2: torch.Tensor,
    ) -> float:
        """
        Patch-wise perceptual similarity difference between two images,
        around corresponding keypoints coords1 / coords2.
        """
        assert self.p_ssim is not None

        sims: List[float] = []

        img1 = img1.cuda()
        img2 = img2.cuda()

        patch_size = 32

        for coord1, coord2 in zip(coords1, coords2):
            x1, y1 = int(coord1[0]), int(coord1[1])
            x2, y2 = int(coord2[0]), int(coord2[1])

            def out_of_bounds(x, y, img):
                return (
                    x - patch_size < 0
                    or x + patch_size > img.shape[2]
                    or y - patch_size < 0
                    or y + patch_size > img.shape[1]
                )

            if out_of_bounds(x1, y1, img1) or out_of_bounds(x2, y2, img2):
                # Large penalty if any patch goes out of bounds
                return 100.0

            patch1 = img1[:, y1 - patch_size : y1 + patch_size, x1 - patch_size : x1 + patch_size]
            patch2 = img2[:, y2 - patch_size : y2 + patch_size, x2 - patch_size : x2 + patch_size]
            patch3 = img1[:, y2 - patch_size : y2 + patch_size, x2 - patch_size : x2 + patch_size]

            sim = self.p_ssim(patch1.unsqueeze(0), patch2.unsqueeze(0))
            sim2 = self.p_ssim(patch1.unsqueeze(0), patch3.unsqueeze(0))
            sim = sim - sim2
            sims.append(sim.item())

        final_sim = float(np.mean(sims))
        return final_sim
