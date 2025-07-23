# astrbot_plugin_sdgen_v2/commands/native.py

from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.all import logger, Image as MessageImage, Plain as MessageText

from ..core.generation import GenerationManager
from ..utils.tag_manager import TagManager
from ..static import messages

class NativeCommands:
    def __init__(self, context: Context, generator: GenerationManager, tag_manager: TagManager):
        self.context = context
        self.generator = generator
        self.tag_manager = tag_manager

    @filter.command("原生画", "native")
    async def handle_native(self, event: AstrMessageEvent):
        """Handles the /原生画 command for direct txt2img generation."""
        
        # 0. Permission Check
        group_id = event.get_group_id()
        blacklist_groups = self.context._config.get("blacklist_groups", [])
        if group_id in blacklist_groups:
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
            
            image_components = [MessageImage.from_base64(img) for img in processed_images]
            
            if self.context._config.get("enable_show_positive_prompt", True):
                yield event.plain_result(f"{messages.MSG_PROMPT_DISPLAY}: {final_prompt}")
            
            # Check sending preference
            if self.context._config.get("enable_forward_message", False):
                yield event.send_result(image_components)
            else:
                yield event.chain_result(image_components)

        except Exception as e:
            logger.error(f"An error occurred in native command: {e}")
            yield event.plain_result(messages.MSG_UNKNOWN_ERROR)

def register_native_commands(star_instance):
    """Registers the native command handlers to the main star instance."""
    native_handler = NativeCommands(star_instance.context, star_instance.generator, star_instance.tag_manager)
    
    star_instance.handle_native = native_handler.handle_native
