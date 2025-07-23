# astrbot_plugin_sdgen_v2/commands/inspire.py

from astrbot.api.star import Context
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.all import logger, Image as MessageImage, Plain as MessageText

from ..core.generation import GenerationManager
from ..utils.tag_manager import TagManager
from ..utils.llm_helper import LLMHelper
from ..static import messages

class InspireCommands:
    def __init__(self, context: Context, generator: GenerationManager, tag_manager: TagManager, llm_helper: LLMHelper):
        self.context = context
        self.generator = generator
        self.tag_manager = tag_manager
        self.llm_helper = llm_helper

    @filter.command("inspire")
    async def handle_inspire(self, event: AstrMessageEvent):
        """Handles the /inspire command for vision-based txt2img generation."""
        
        # 0. Permission Check
        group_id = event.get_group_id()
        blacklist_groups = self.context._config.get("blacklist_groups", [])
        if group_id in blacklist_groups:
            return # Silently ignore if in blacklist

        # 1. Extract image and text from the event
        image_b64 = None
        prompt_text = ""

        if event.message_obj and event.message_obj.message:
            for comp in event.message_obj.message:
                if isinstance(comp, MessageImage) and hasattr(comp, 'url') and comp.url:
                    try:
                        image_b64 = await self.generator.client.download_image_as_base64(comp.url)
                    except ConnectionError:
                        yield event.plain_result(messages.MSG_IMG_DOWNLOAD_FAILED)
                        return
                elif isinstance(comp, MessageText):
                    prompt_text += comp.text + " "
        
        prompt_text = prompt_text.strip()

        if not image_b64:
            yield event.plain_result(messages.MSG_NO_IMAGE_PROVIDED)
            return

        await event.send(event.plain_result(messages.MSG_GENERATING))

        # 2. Process prompt: replace tags, then use multi-modal LLM
        prompt_text, _ = self.tag_manager.replace(prompt_text)
        
        final_prompt = await self.llm_helper.generate_prompt_from_image(
            image_b64=image_b64,
            user_instruction=prompt_text,
            guidelines=self.context._config.get("prompt_guidelines", ""),
            prefix=self.context._config.get("llm_prompt_prefix", "")
        )

        # Graceful Degradation
        if final_prompt is None:
            yield event.plain_result(messages.MSG_VISION_FALLBACK)
            # Fallback to text-only generation
            final_prompt = await self.llm_helper.generate_text_prompt(
                base_prompt=prompt_text,
                guidelines=self.context._config.get("prompt_guidelines", ""),
                prefix=self.context._config.get("llm_prompt_prefix", "")
            )

        # 3. Generate image using txt2img
        try:
            generated_images = await self.generator.generate_txt2img(final_prompt, group_id)
            
            if not generated_images:
                yield event.plain_result(messages.MSG_API_ERROR)
                return

            # 4. Process and send results
            processed_images = await self.generator.process_and_upscale_images(generated_images)
            
            image_components = [MessageImage.from_base64(img) for img in processed_images]
            
            if self.context._config.get("enable_show_positive_prompt", True):
                yield event.plain_result(f"{messages.MSG_PROMPT_DISPLAY}: {final_prompt}")

            # Check sending preference
            if self.context._config.get("enable_forward_message", False):
                yield event.send_result(image_components)
            else:
                yield event.chain_result(image_components)

        except Exception as e:
            logger.error(f"An error occurred in inspire command: {e}")
            yield event.plain_result(messages.MSG_UNKNOWN_ERROR)

def register_inspire_commands(star_instance):
    """Registers the inspire command handlers to the main star instance."""
    inspire_handler = InspireCommands(star_instance.context, star_instance.generator, star_instance.tag_manager, star_instance.llm_helper)
    star_instance.handle_inspire = inspire_handler.handle_inspire
