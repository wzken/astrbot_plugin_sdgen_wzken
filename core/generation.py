# astrbot_plugin_sdgen_wzken/core/generation.py

from typing import Dict, List, Any
import base64

from .client import SDAPIClient
from ..utils.sd_utils import SDUtils

class GenerationManager:
    DEFAULT_UPSCALE_FACTOR = 2.0
    DEFAULT_UPSCALER = "Latent"

    def __init__(self, config: dict, client: SDAPIClient, sd_utils: SDUtils):
        self.config = config
        self.client = client
        self.sd_utils = sd_utils

    async def generate_txt2img(self, prompt: str, negative_prompt: str) -> List[str]:
        """Generates an image from text and returns a list of base64 encoded images."""
        payload = self.sd_utils.prepare_txt2img_payload(prompt, negative_prompt)
        response = await self.client.txt2img(payload)
        return response.get("images", [])

    async def generate_img2img(self, image_info: Dict[str, Any], prompt: str, negative_prompt: str) -> List[str]:
        """Generates an image from an image and text, returns a list of base64 encoded images."""
        payload = self.sd_utils.prepare_img2img_payload(
            image_data=image_info["b64"],
            prompt=prompt,
            original_width=image_info["width"],
            original_height=image_info["height"],
            negative_prompt=negative_prompt
        )
        response = await self.client.img2img(payload)
        return response.get("images", [])

    async def process_and_upscale_images(self, images_b64: List[str]) -> List[str]:
        """Upscales a list of base64 encoded images if post-processing upscale is enabled."""
        # Only perform upscaling if the main toggle is on AND the mode is 'post'
        if not self.config.get("enable_upscale", False) or self.config.get("upscaling_mode") != "post":
            return images_b64

        upscaled_images = []
        upscale_params = self.config.get("upscale_params", {})
        for image_b64 in images_b64:
            payload = {
                "image": image_b64,
                "upscaling_resize": upscale_params.get("upscale_factor", self.DEFAULT_UPSCALE_FACTOR),
                "upscaler_1": upscale_params.get("upscaler", self.DEFAULT_UPSCALER),
                "resize_mode": 0,
            }
            response = await self.client.extra_single_image(payload)
            upscaled_images.append(response.get("image", image_b64))
        
        return upscaled_images
