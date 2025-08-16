import re
import io
import math
from PIL import Image
from astrbot.api.all import logger, AstrBotConfig
from astrbot.api.event import AstrMessageEvent
import astrbot.api.message_components as Comp
from typing import Tuple, List, Any
from ..static import messages

class SDUtils:
    def __init__(self, config: AstrBotConfig, context):
        self.config = config
        self.context = context

    def validate_resolution(self, width: int, height: int) -> bool:
        """Validates if the given width and height are valid SD resolutions."""
        return width > 0 and height > 0 and width % 64 == 0 and height % 64 == 0

    def _get_closest_resolution(self, original_width: int, original_height: int) -> Tuple[int, int]:
        """
        根据原始图片尺寸，保持横纵比，并调整到不超过 1920x1920，且为 64 的倍数，且尽可能大。
        """
        max_dim = 1920
        min_dim = 64
        
        if original_height == 0:
            aspect_ratio = 1.0
        else:
            aspect_ratio = original_width / original_height

        if original_width >= original_height:
            target_width = max_dim
            target_height = int(max_dim / aspect_ratio)
        else:
            target_height = max_dim
            target_width = int(max_dim * aspect_ratio)

        target_width = max(min_dim, math.floor(target_width / 64) * 64)
        target_height = max(min_dim, math.floor(target_height / 64) * 64)

        if target_width > max_dim or target_height > max_dim:
            scale_down_factor = min(max_dim / target_width, max_dim / target_height)
            target_width = max(min_dim, math.floor(target_width * scale_down_factor / 64) * 64)
            target_height = max(min_dim, math.floor(target_height * scale_down_factor / 64) * 64)

        return target_width, target_height

    def is_positive_int(self, value: Any) -> bool:
        """Validates if the value is a positive integer."""
        return isinstance(value, int) and value > 0

    def is_non_negative_int(self, value: Any) -> bool:
        """Validates if the value is a non-negative integer."""
        return isinstance(value, int) and value >= 0

    def is_valid_denoising_strength(self, value: Any) -> bool:
        """Validates if the value is a valid denoising strength (0.0 to 1.0)."""
        return isinstance(value, (int, float)) and 0.0 <= value <= 1.0

    def is_valid_hr_denoising_strength(self, value: Any) -> bool:
        """Validates if the value is a valid HR denoising strength (0.0 to 1.0)."""
        return isinstance(value, (int, float)) and 0.0 <= value <= 1.0

    def is_valid_image_cfg_scale(self, value: Any) -> bool:
        """Validates if the value is a valid image CFG scale (>= 0)."""
        return isinstance(value, (int, float)) and value >= 0

    def is_valid_seed(self, value: Any) -> bool:
        """Validates if the value is a valid seed."""
        return isinstance(value, int)

    def prepare_txt2img_payload(self, prompt: str, negative_prompt: str) -> dict:
        """构建生成参数"""
        params = self.config.get("default_params", {})
        
        payload = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "width": params.get("width", 1024),
            "height": params.get("height", 1024),
            "steps": params.get("steps", 20),
            "sampler_name": params.get("sampler_name", "DPM++ 2M Karras"),
            "scheduler": params.get("scheduler", "Karras"),
            "cfg_scale": params.get("cfg_scale", 7),
            "batch_size": params.get("batch_size", 1),
            "n_iter": params.get("n_iter", 1),
            "seed": params.get("seed", -1),
            "restore_faces": params.get("restore_faces", False),
            "tiling": params.get("tiling", False),
            "enable_hr": params.get("enable_hr", False),
        }

        if payload["enable_hr"]:
            hr_params = {
                "hr_scale": params.get("hr_scale", 2.0),
                "hr_upscaler": params.get("hr_upscaler", "Latent"),
                "hr_second_pass_steps": params.get("hr_second_pass_steps", 0),
                "denoising_strength": params.get("hr_denoising_strength", 0.7),
                "hr_sampler_name": params.get("sampler_name"),
                "hr_scheduler": params.get("scheduler"),
            }
            payload.update(hr_params)

        return payload

    def prepare_img2img_payload(self, image_data: str, prompt: str, original_width: int, original_height: int, negative_prompt: str) -> dict:
        """构建图生图生成参数"""
        params = self.config.get("img2img_params", {})

        # Use configured resolution if available, otherwise calculate it
        target_width = params.get("width")
        target_height = params.get("height")
        if not target_width or not target_height:
            target_width, target_height = self._get_closest_resolution(original_width, original_height)

        return {
            "init_images": [image_data],
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "width": target_width,
            "height": target_height,
            "steps": params.get("steps", 20),
            "sampler_name": params.get("sampler_name", "DPM++ 2M Karras"),
            "scheduler": params.get("scheduler", "Karras"),
            "cfg_scale": params.get("cfg_scale", 7),
            "denoising_strength": params.get("denoising_strength", 0.75),
            "batch_size": params.get("batch_size", 1),
            "n_iter": params.get("n_iter", 1),
            "seed": params.get("seed", -1),
            "restore_faces": params.get("restore_faces", False),
            "tiling": params.get("tiling", False),
            "image_cfg_scale": params.get("image_cfg_scale"),
            "include_init_images": params.get("include_init_images", False),
        }

    def get_full_prompts(self, base_prompt: str, group_id: str, is_i2i: bool, is_native: bool) -> Tuple[str, str]:
        """Constructs the full positive and negative prompts based on configuration."""
        if is_native:
            full_positive_prompt = base_prompt
        else:
            positive_prefix = self.config.get("positive_prompt_i2i" if is_i2i else "positive_prompt_global", "")
            
            whitelist_groups = self.config.get("whitelist_groups", [])
            if group_id and group_id in whitelist_groups:
                positive_prefix = self.config.get("positive_prompt_whitelist", "masterpiece, best quality")
            
            full_positive_prompt = ", ".join(filter(None, [positive_prefix, base_prompt]))

        negative_prefix = self.config.get("negative_prompt_global", "(worst quality, low quality:1.4)")
        full_negative_prompt = negative_prefix
        
        return full_positive_prompt, full_negative_prompt

    async def send_image_results(self, event: AstrMessageEvent, images_b64: List[str], full_positive_prompt: str):
        """Sends the generated images and prompt to the user based on configuration."""
        image_components = [Comp.Image.fromBase64(img) for img in images_b64]
        
        if self.config.get("enable_show_positive_prompt", True):
            image_components.insert(0, Comp.Plain(f"{messages.MSG_PROMPT_DISPLAY}: {full_positive_prompt}"))

        send_private = self.config.get("enable_private_message", False)
        use_forward_reply = self.config.get("enable_forward_reply", False)
        user_id = event.get_sender_id()

        if send_private:
            await event.send(event.send_result(image_components, user_id=user_id))
        elif use_forward_reply:
            node = Comp.Node(uin=event.message_obj.self_id, name="SD 生成结果", content=[Comp.Plain(text="SD 生成结果"), *image_components]) # Added Plain component for node title
            await event.send(event.chain_result([node]))
        else:
            await event.send(event.chain_result(image_components))
 
    def get_generation_params_str(self) -> str:
        """获取当前图像生成的参数"""
        positive_prompt_global = self.config.get("positive_prompt_global", "")
        negative_prompt_global = self.config.get("negative_prompt_global", "")

        params = self.config.get("default_params", {})
        width = params.get("width") or messages.MSG_NOT_SET
        height = params.get("height") or messages.MSG_NOT_SET
        steps = params.get("steps") or messages.MSG_NOT_SET
        sampler = params.get("sampler_name") or messages.MSG_NOT_SET
        scheduler = params.get("scheduler") or messages.MSG_NOT_SET
        cfg_scale = params.get("cfg_scale") or messages.MSG_NOT_SET
        batch_size = params.get("batch_size") or messages.MSG_NOT_SET
        n_iter = params.get("n_iter") or messages.MSG_NOT_SET

        base_model = self.config.get("base_model").strip() or messages.MSG_NOT_SET

        hr_denoising_strength = params.get("hr_denoising_strength") or messages.MSG_NOT_SET

        return (
            f"{messages.MSG_GLOBAL_POSITIVE_PROMPT}: {positive_prompt_global}\n"
            f"{messages.MSG_GLOBAL_NEGATIVE_PROMPT}: {negative_prompt_global}\n"
            f"{messages.MSG_BASE_MODEL}: {base_model}\n"
            f"{messages.MSG_IMAGE_DIMENSIONS}: {width}x{height}\n"
            f"{messages.MSG_STEPS}: {steps}\n"
            f"{messages.MSG_SAMPLER}: {sampler}\n"
            f"{messages.MSG_SCHEDULER}: {scheduler}\n"
            f"{messages.MSG_CFG_SCALE}: {cfg_scale}\n"
            f"{messages.MSG_BATCH_SIZE}: {batch_size}\n"
            f"{messages.MSG_ITERATIONS}: {n_iter}\n"
            f"{messages.MSG_HIRES_DENOISING}: {hr_denoising_strength}"
        )

    def get_img2img_params_str(self) -> str:
        """获取当前图生图的参数"""
        params = self.config.get("img2img_params", {})
        denoising_strength = params.get("denoising_strength") or messages.MSG_NOT_SET
        steps = params.get("steps") or messages.MSG_NOT_SET
        sampler = params.get("sampler_name") or messages.MSG_NOT_SET
        scheduler = params.get("scheduler") or messages.MSG_NOT_SET
        cfg_scale = params.get("cfg_scale") or messages.MSG_NOT_SET
        batch_size = params.get("batch_size") or messages.MSG_NOT_SET
        n_iter = params.get("n_iter") or messages.MSG_NOT_SET
        image_cfg_scale = params.get("image_cfg_scale") or messages.MSG_NOT_SET

        return (
            f"{messages.MSG_DENOISING_STRENGTH}: {denoising_strength}\n"
            f"{messages.MSG_STEPS}: {steps}\n"
            f"{messages.MSG_SAMPLER}: {sampler}\n"
            f"{messages.MSG_SCHEDULER}: {scheduler}\n"
            f"{messages.MSG_CFG_SCALE}: {cfg_scale}\n"
            f"{messages.MSG_BATCH_SIZE}: {batch_size}\n"
            f"{messages.MSG_ITERATIONS}: {n_iter}\n"
            f"{messages.MSG_IMAGE_CFG_SCALE}: {image_cfg_scale}"
        )

    def get_upscale_params_str(self) -> str:
        """获取当前放大处理的参数"""
        params = self.config.get("upscale_params", {})
        upscaler = params.get("upscaler") or messages.MSG_NOT_SET
        factor = params.get("upscale_factor") or messages.MSG_NOT_SET
        
        return (
            f"{messages.MSG_UPSCALER}: {upscaler}\n"
            f"{messages.MSG_UPSCALE_FACTOR}: {factor}"
        )
