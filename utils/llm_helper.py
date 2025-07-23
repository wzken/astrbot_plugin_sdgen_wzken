# astrbot_plugin_sdgen_v2/utils/llm_helper.py

from typing import Optional

from astrbot.api.all import Context, logger

class LLMHelper:
    INSPIRE_PROMPT_TEMPLATE = (
        "{prefix}\n{guidelines}\n"
        "Based on the image provided and the following instruction: '{user_instruction}', "
        "generate a detailed and creative prompt for Stable Diffusion."
    )

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
                return response.completion_text.strip()
        except Exception as e:
            logger.error(f"Error during text_chat with LLM provider: {e}")
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
        """
        provider = self.context.get_using_provider()
        if not provider or not hasattr(provider, 'image_chat'):
            return None

        full_prompt = self.INSPIRE_PROMPT_TEMPLATE.format(
            prefix=prefix,
            guidelines=guidelines,
            user_instruction=user_instruction
        )

        try:
            response = await provider.image_chat(
                prompt=full_prompt,
                image=image_b64,
                session_id=None
            )
            if response and response.completion_text:
                return response.completion_text.strip()
        except Exception as e:
            logger.error(f"Error during image_chat with LLM provider: {e}")
            return None
        
        return None
