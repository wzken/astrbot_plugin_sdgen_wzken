import re
import io
import math
from PIL import Image
from astrbot.api.all import logger, AstrBotConfig
from astrbot.api.event import AstrMessageEvent
from ..static import messages

class SDUtils:
    def __init__(self, config: AstrBotConfig, context):
        self.config = config
        self.context = context

    def _get_closest_resolution(self, original_width: int, original_height: int) -> tuple[int, int]:
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

    async def generate_payload(self, prompt: str, negative_prompt: str) -> dict:
        """构建生成参数"""
        params = self.config["default_params"]

        return {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "width": params["width"],
            "height": params["height"],
            "steps": params["steps"],
            "sampler_name": params["sampler"],
            "scheduler": params["scheduler"],
            "cfg_scale": params["cfg_scale"],
            "batch_size": params["batch_size"],
            "n_iter": params["n_iter"],
        }

    async def generate_img2img_payload(self, image_data: str, prompt: str, original_width: int, original_height: int, negative_prompt: str) -> dict:
        """构建图生图生成参数"""
        params = self.config["img2img_params"]
        
        target_width, target_height = self._get_closest_resolution(original_width, original_height)

        sampler_name = params.get("sampler", "Euler a")
        scheduler_name = params.get("scheduler", "DPM++ 2M Karras")

        return {
            "init_images": [image_data],
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "width": target_width,
            "height": target_height,
            "steps": params["steps"],
            "sampler_name": sampler_name,
            "scheduler": scheduler_name,
            "cfg_scale": params["cfg_scale"],
            "denoising_strength": params["denoising_strength"],
            "batch_size": params["batch_size"],
            "n_iter": params["n_iter"],
        }

    def trans_prompt(self, prompt: str) -> str:
        """
        替换提示词中的所有下划线为空格，并自动加上敏感词括号说明
        """
        prompt = prompt.replace("_", " ")
        prompt_with_notice = f"{prompt}{self.config.get('llm_prompt_suffix', '')}"
        return prompt_with_notice

    async def generate_prompt_with_llm(self, event: AstrMessageEvent, prompt: str) -> str:
        provider = self.context.get_using_provider()
        if provider:
            llm_prompt_prefix = self.config.get("LLM_PROMPT_PREFIX", messages.MSG_DEFAULT_LLM_PROMPT_PREFIX)
            
            cleaned_user_prompt = prompt
            
            group_id = event.get_group_id()
            whitelist_groups = self.config.get("whitelist_groups", [])
            
            final_prompt_description = cleaned_user_prompt
            prompt_guidelines = self.config.get("prompt_guidelines", "")

            if group_id and group_id in whitelist_groups:
                final_prompt_description = f"{cleaned_user_prompt}{messages.MSG_LLM_PROMPT_NOTICE}"
                prompt_guidelines = ""
            
            full_prompt = f"{llm_prompt_prefix}\n{prompt_guidelines}\n描述：{final_prompt_description}".strip()

            response = await provider.text_chat(full_prompt, session_id=None)
            if response.completion_text:
                generated_prompt = re.sub(r"<think>[\s\S]*</think>", "", response.completion_text).strip()
                logger.info(f"{messages.MSG_LLM_RETURNED_TAG}: {generated_prompt}")
                return generated_prompt

        return ""

    def get_generation_params_str(self) -> str:
        """获取当前图像生成的参数"""
        positive_prompt_global = self.config.get("positive_prompt_global", "")
        negative_prompt_global = self.config.get("negative_prompt_global", "")

        params = self.config.get("default_params", {})
        width = params.get("width") or messages.MSG_NOT_SET
        height = params.get("height") or messages.MSG_NOT_SET
        steps = params.get("steps") or messages.MSG_NOT_SET
        sampler = params.get("sampler") or messages.MSG_NOT_SET
        scheduler = params.get("scheduler") or messages.MSG_NOT_SET
        cfg_scale = params.get("cfg_scale") or messages.MSG_NOT_SET
        batch_size = params.get("batch_size") or messages.MSG_NOT_SET
        n_iter = params.get("n_iter") or messages.MSG_NOT_SET

        base_model = self.config.get("base_model").strip() or messages.MSG_NOT_SET

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
            f"{messages.MSG_N_ITER}: {n_iter}"
        )

    def get_upscale_params_str(self) -> str:
        """获取当前图像增强（超分辨率放大）参数"""
        params = self.config["default_params"]
        upscale_factor = params.get("upscale_factor", "2")
        upscaler = params.get("upscaler", messages.MSG_NOT_SET)

        return (
            f"{messages.MSG_UPSCALE_FACTOR}: {upscale_factor}\n"
            f"{messages.MSG_UPSCALER_ALGORITHM}: {upscaler}"
        )

    def get_img2img_params_str(self) -> str:
        """获取当前图生图的参数"""
        img2img_params = self.config.get("img2img_params", {})
        denoising_strength = img2img_params.get("denoising_strength") or messages.MSG_NOT_SET
        steps = img2img_params.get("steps") or messages.MSG_NOT_SET
        sampler = img2img_params.get("sampler") or messages.MSG_NOT_SET
        scheduler = img2img_params.get("scheduler") or messages.MSG_NOT_SET
        cfg_scale = img2img_params.get("cfg_scale") or messages.MSG_NOT_SET
        batch_size = img2img_params.get("batch_size") or messages.MSG_NOT_SET
        n_iter = img2img_params.get("n_iter") or messages.MSG_NOT_SET

        return (
            f"{messages.MSG_DENOISING_STRENGTH}: {denoising_strength}\n"
            f"{messages.MSG_STEPS}: {steps}\n"
            f"{messages.MSG_SAMPLER}: {sampler}\n"
            f"{messages.MSG_SCHEDULER}: {scheduler}\n"
            f"{messages.MSG_CFG_SCALE}: {cfg_scale}\n"
            f"{messages.MSG_BATCH_SIZE}: {batch_size}\n"
            f"{messages.MSG_N_ITER}: {n_iter}\n"
            f"{messages.MSG_IMG2IMG_RESOLUTION_AUTO_SET.format(width='自动', height='自动')}"
        )
