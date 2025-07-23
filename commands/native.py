# astrbot_plugin_sdgen_v2/commands/native.py

from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.message import MessageImage, MessageText
from astrbot.api.all import logger

from ..core.generation import GenerationManager
from ..utils.tag_manager import TagManager
from ..static import messages

class NativeCommands:
    def __init__(self, generator: GenerationManager, tag_manager: TagManager):
        self.generator = generator
        self.tag_manager = tag_manager

    @filter.command("原生画", "native")
    async def handle_native(self, event: AstrMessageEvent):
        """Handles the /原生画 command for direct txt2img generation."""
        
        # 0. Permission Check
        group_id = event.get_group_id()
        if group_id in self.generator.config.blacklist_groups:
            return # Silently ignore if in blacklist

        # 1. Extract text from the event
        prompt_text = event.get_plain_text().strip()

        if not prompt_text:
            yield event.plain_result(messages.MSG_NO_PROMPT_PROVIDED)
            return

        await event.send(event.plain_result(messages.MSG_GENERATING))

        # 2. Process prompt: only replace tags
        final_prompt, _ = self.tag_manager.replace(prompt_text)
        
        # 3. Generate image
        try:
            generated_images = await self.generator.generate_txt2img(final_prompt, group_id)
            
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
            logger.error(f"An error occurred in native command: {e}")
            yield event.plain_result(messages.MSG_UNKNOWN_ERROR)

def register_native_commands(star_instance):
    """Registers the native command handlers to the main star instance."""
    native_handler = NativeCommands(star_instance.generator, star_instance.tag_manager)
    
    star_instance.handle_native = native_handler.handle_native
