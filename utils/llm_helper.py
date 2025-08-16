# astrbot_plugin_sdgen_wzken/utils/llm_helper.py

import json
from astrbot.api.star import Context
from ..static import messages

class LLMHelper:
    def __init__(self, context: Context):
        self.context = context

    async def generate_text_prompt(self, base_prompt: str, guidelines: str = "", prefix: str = "") -> str:
        """
        Uses the LLM to generate a detailed English prompt for image generation.
        """
        fallback_prompt = f"{prefix}, {base_prompt}" if prefix else base_prompt
        provider = self.context.get_using_provider()

        if not provider:
            return fallback_prompt

        # Construct a prompt for the LLM to generate a better prompt
        generation_instruction = messages.SYSTEM_PROMPT_SD_PROMPT_GENERATION.format(
            guidelines=guidelines,
            prefix=prefix,
            base_prompt=base_prompt
        )

        try:
            llm_response = await provider.text_chat(
                prompt=generation_instruction,
                contexts=[],
            )

            if llm_response and llm_response.completion_text:
                # Clean up the response, remove potential markdown code blocks
                generated_prompt = llm_response.completion_text.strip()
                if generated_prompt.startswith("```") and generated_prompt.endswith("```"):
                    generated_prompt = generated_prompt.strip("`").strip()
                return generated_prompt
            else:
                # Fallback if LLM fails
                return fallback_prompt

        except Exception:
            # Fallback in case of any API error
            return fallback_prompt
