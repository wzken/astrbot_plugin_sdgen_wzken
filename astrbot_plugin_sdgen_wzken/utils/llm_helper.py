# astrbot_plugin_sdgen_v2/utils/llm_helper.py

from typing import Optional, Any

from astrbot.api.event import AstrMessageEvent
from astrbot.api.all import Context

class LLMHelper:
    def __init__(self, context: Context):
        self.context = context

    async def generate_text_prompt(self, base_prompt: str, guidelines: str, prefix: str) -> str:
        """
        Generates a refined text prompt using a text-only LLM.
        """
        provider = self.context.get_using_provider()
        if not provider:
            return base_prompt

        full_prompt = f"{prefix}\n{guidelines}\n描述：{base_prompt}".strip()
        
        try:
            response = await provider.text_chat(full_prompt, session_id=None)
            if response and response.completion_text:
                # Basic cleaning of potential LLM-internal tags
                return response.completion_text.strip()
        except Exception:
            # logger.error(f"Error during LLM text generation: {e}")
            # In case of error, return the original prompt
            return base_prompt
        
        return base_prompt

    async def generate_prompt_from_image(
        self, 
        image_b64: str, 
        user_instruction: str, 
        guidelines: str, 
        prefix: str
    ) -> Optional[str]:
        """
        Generates a prompt from an image and a text instruction using a multi-modal LLM.
        This is the core of the 'inspire' feature.
        Returns the generated prompt, or None if the operation fails.
        """
        provider = self.context.get_using_provider()
        if not provider or not hasattr(provider, 'image_chat'):
            # Provider does not support multi-modal chat, trigger graceful degradation
            return None

        # Construct the prompt for the multi-modal LLM
        # The exact format may depend on the provider's API
        full_prompt = f"{prefix}\n{guidelines}\n"
        full_prompt += f"Based on the image provided and the following instruction: '{user_instruction}', "
        full_prompt += "generate a detailed and creative prompt for Stable Diffusion."

        try:
            # This assumes the provider's image_chat method can take a base64 image string
            # The actual implementation might need adjustment based on AstrBot's provider API
            response = await provider.image_chat(
                prompt=full_prompt,
                image=image_b64, # Assuming the API takes base64 directly
                session_id=None
            )
            if response and response.completion_text:
                return response.completion_text.strip()
        except Exception:
            # logger.error(f"Error during multi-modal LLM generation: {e}")
            return None
        
        return None
