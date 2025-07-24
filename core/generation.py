# astrbot_plugin_sdgen_v2/core/generation.py

from typing import Dict, List, Any
import base64

from .client import SDAPIClient

class GenerationManager:
    # Default prompts can be defined as class constants for clarity
    DEFAULT_POSITIVE_PROMPT_WHITELIST = "masterpiece, best quality"
    DEFAULT_NEGATIVE_PROMPT_GLOBAL = "(worst quality, low quality:1.4)"
    DEFAULT_UPSCALE_FACTOR = 2.0
    DEFAULT_UPSCALER = "Latent"

    def __init__(self, config: dict, client: SDAPIClient):
        self.config = config
        self.client = client

    def _build_base_payload(self, params: dict, prompt: str, negative_prompt: str, group_id: str) -> Dict[str, Any]:
        """Builds a base payload, applying group-specific prompt rules."""
        payload = params.copy()
        
        whitelist_groups = self.config.get("whitelist_groups", [])
        if group_id in whitelist_groups:
            positive_prefix = self.config.get("positive_prompt_whitelist", self.DEFAULT_POSITIVE_PROMPT_WHITELIST)
            negative_prefix = self.config.get("negative_prompt_whitelist", "")
        else:
            positive_prefix = self.config.get("positive_prompt_global", "")
            negative_prefix = self.config.get("negative_prompt_global", self.DEFAULT_NEGATIVE_PROMPT_GLOBAL)

        # A more robust way to combine prompts
        final_positive_parts = [p for p in [positive_prefix, prompt] if p]
        final_negative_parts = [p for p in [negative_prefix, negative_prompt] if p]

        payload["prompt"] = ", ".join(final_positive_parts)
        payload["negative_prompt"] = ", ".join(final_negative_parts)
        
        return payload

    def build_txt2img_payload(self, prompt: str, group_id: str, negative_prompt: str = "") -> Dict[str, Any]:
        """Builds the payload for a txt2img request."""
        params = self.config.get("default_params", {})
        payload = self._build_base_payload(params, prompt, negative_prompt, group_id)
        
        # Add common parameters
        payload["seed"] = params.get("seed", -1)
        payload["restore_faces"] = params.get("restore_faces", False)
        payload["tiling"] = params.get("tiling", False)

        # Add High-res fix parameters if enabled
        if params.get("enable_hr", False):
            payload["enable_hr"] = True
            payload["hr_scale"] = params.get("hr_scale", 2.0)
            payload["hr_upscaler"] = params.get("hr_upscaler", "Latent")
            payload["hr_second_pass_steps"] = params.get("hr_second_pass_steps", 0)
        else:
            payload["enable_hr"] = False # Explicitly set to false if not enabled

        return payload

    def build_img2img_payload(self, init_image_b64: str, prompt: str, group_id: str, negative_prompt: str = "") -> Dict[str, Any]:
        """Builds the payload for an img2img request."""
        params = self.config.get("img2img_params", {})
        payload = self._build_base_payload(params, prompt, negative_prompt, group_id)
        payload["init_images"] = [init_image_b64]
        
        # Add common parameters
        payload["seed"] = params.get("seed", -1)
        payload["restore_faces"] = params.get("restore_faces", False)
        payload["tiling"] = params.get("tiling", False)
        
        # Add img2img specific parameters
        payload["image_cfg_scale"] = params.get("image_cfg_scale", 0)
        payload["include_init_images"] = params.get("include_init_images", False)

        return payload

    async def generate_txt2img(self, prompt: str, group_id: str, negative_prompt: str = "") -> List[str]:
        """Generates an image from text and returns a list of base64 encoded images."""
        payload = self.build_txt2img_payload(prompt, group_id, negative_prompt)
        response = await self.client.txt2img(payload)
        return response.get("images", [])

    async def generate_img2img(self, init_image_b64: str, prompt: str, group_id: str, negative_prompt: str = "") -> List[str]:
        """Generates an image from an image and text, returns a list of base64 encoded images."""
        payload = self.build_img2img_payload(init_image_b64, prompt, group_id, negative_prompt)
        response = await self.client.img2img(payload)
        return response.get("images", [])

    async def process_and_upscale_images(self, images_b64: List[str]) -> List[str]:
        """Upscales a list of base64 encoded images if upscaling is enabled."""
        if not self.config.get("enable_upscale", False):
            return images_b64

        upscaled_images = []
        # Ensure upscale_params is a dictionary, even if missing from config
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
