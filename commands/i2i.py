# astrbot_plugin_sdgen_v2/commands/i2i.py

from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.message import MessageImage, MessageText
from astrbot.api.all import logger

from ..core.generation import GenerationManager
from ..utils.tag_manager import TagManager
from ..utils.llm_helper import LLMHelper
from ..static import messages

class I2ICommands:
    def __init__(self, generator: GenerationManager, tag_manager: TagManager, llm_helper: LLMHelper):
        self.generator = generator
        self.tag_manager = tag_manager
        self.llm_helper = llm_helper

    @filter.command("i2i")
    async def handle_i2i(self, event: AstrMessageEvent):
        """Handles the /i2i command for image-to-image generation."""
        
        # 0. Permission Check
        group_id = event.get_group_id()
        if group_id in self.generator.config.blacklist_groups:
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

        await event.send(event.plain_result(messages.MSG_IMG2IMG_GENERATING))

        # 2. Process prompt: replace tags and optionally use LLM
        prompt_text, _ = self.tag_manager.replace(prompt_text)
        
        if self.generator.config.enable_llm_prompt_generation:
            final_prompt = await self.llm_helper.generate_text_prompt(
                base_prompt=prompt_text,
                guidelines=self.generator.config.prompt_guidelines,
                prefix=self.generator.config.llm_prompt_prefix
            )
        else:
            final_prompt = prompt_text

        # 3. Generate image
        try:
            generated_images = await self.generator.generate_img2img(image_b64, final_prompt, group_id)
            
            if not generated_images:
                yield event.plain_result(messages.MSG_API_ERROR)
                return

            # 4. Process and send results
            processed_images = await self.generator.process_and_upscale_images(generated_images)
            
            image_components = [MessageImage.fromBase64(img) for img in processed_images]
            
            if self.generator.config.enable_show_positive_prompt:
                yield event.plain_result(f"{messages.MSG_PROMPT_DISPLAY}: {final_prompt}")

            # Check sending preference
            if self.generator.config.enable_forward_message:
                yield event.send_result(image_components)
            else:
                yield event.chain_result(image_components)

        except Exception as e:
            logger.error(f"An error occurred in i2i command: {e}")
            yield event.plain_result(messages.MSG_UNKNOWN_ERROR)

def register_i2i_commands(star_instance):
    """Registers the i2i command handlers to the main star instance."""
    i2i_handler = I2ICommands(star_instance.generator, star_instance.tag_manager, star_instance.llm_helper)
    
    # Manually add the method to the star instance so AstrBot can discover it
    star_instance.handle_i2i = i2i_handler.handle_i2i
