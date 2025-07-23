# astrbot_plugin_sdgen_wzken/main.py

"""
Plugin Name: SDGen_wzken
Author: wzken
Version: 2.0.0
Description: A smarter and more powerful image generation plugin for AstrBot using Stable Diffusion.
"""

import json
from pathlib import Path

import astrbot.api.message_components as Comp
from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.star import Context, Star, register

from .core.client import SDAPIClient
from .core.generation import GenerationManager
from .utils.tag_manager import TagManager
from .utils.llm_helper import LLMHelper
from .static import messages

@register("SDGen_wzken", "wzken", "A smarter and more powerful image generation plugin for AstrBot using Stable Diffusion.", "2.0.0")
class SDGeneratorWzken(Star):
    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.config = config
        
        logger.info("SDGen_wzken plugin loaded. Initializing services...")
        self._initialize_services()
        self._register_commands()
        self._register_llm_tools()
        logger.info("SDGen_wzken plugin initialization complete.")

    def _initialize_services(self):
        """Initializes all core services and managers using dependency injection."""
        plugin_dir = Path(__file__).parent.resolve()
        
        # 1. API Client
        self.api_client = SDAPIClient(self.config)
        
        # 2. Generation Manager
        self.generator = GenerationManager(self.config, self.api_client)
        
        # 3. Tag Manager
        tags_file = plugin_dir / "data" / "tags.json"
        tags_file.parent.mkdir(exist_ok=True) # Ensure data directory exists
        self.tag_manager = TagManager(str(tags_file))

        # 5. LLM Helper
        self.llm_helper = LLMHelper(self.context)

    def _register_commands(self):
        """
        Registers all command handlers.
        NOTE: All commands are now defined as methods within this class, decorated
        with @filter.command or @command_group. The framework discovers them
        automatically, so this method can remain empty.
        """
        pass

    def _register_llm_tools(self):
        """Defines and registers LLM function tools."""
        # This method is now empty as LLM tools are registered via @llm_tool decorator.
        pass

    @filter.llm_tool("generate_sd_prompt")
    async def llm_generate_sd_prompt(self, event: AstrMessageEvent, prompt: str):
        """根据用户提供的提示词，生成一个优化过的、更详细的 Stable Diffusion 提示词。

        Args:
            prompt (str): 用户的基本提示词。
        """
        logger.info(f"LLM tool 'generate_sd_prompt' called with prompt: {prompt}")
        final_prompt = await self.llm_helper.generate_text_prompt(
            base_prompt=prompt or "",
            guidelines=self.config.get("prompt_guidelines", ""),
            prefix=self.config.get("llm_prompt_prefix", "")
        )
        return f"已为“{prompt}”生成优化后的提示词：{final_prompt}"

    # --- Settings Commands ---
    @filter.command_group("sd")
    def sd_group(self):
        pass

    @sd_group.command("tag")
    async def handle_tag(self, event: AstrMessageEvent):
        """Manages local tags. Usage: /sd tag add <name> <prompt>, /sd tag del <name>, /sd tag list, /sd tag import {...}"""
        text = event.message_str.strip()
        
        if text == "list":
            all_tags = self.tag_manager.get_all()
            if not all_tags:
                yield event.plain_result("当前没有设置任何本地标签。")
                return
            
            tag_list_str = "\n".join([f"- `{key}`: `{value}`" for key, value in all_tags.items()])
            yield event.plain_result(f"已保存的本地标签：\n{tag_list_str}")
            return

        if text.startswith("del "):
            key_to_del = text[4:].strip()
            if self.tag_manager.del_tag(key_to_del):
                yield event.plain_result(messages.MSG_TAG_DELETED.format(key=key_to_del))
            else:
                yield event.plain_result(f"❌ 未找到名为 '{key_to_del}' 的标签。")
            return

        if text.startswith("add "):
            parts = text[4:].strip().split(maxsplit=1)
            if len(parts) == 2:
                key, value = parts
                self.tag_manager.set_tag(key, value)
                yield event.plain_result(messages.MSG_TAG_SET.format(key=key))
            else:
                yield event.plain_result("❌ 格式错误，请输入 `/sd tag add <名字> <提示词>`。")
            return

        if text.startswith("import "):
            json_str = text[7:].strip()
            try:
                new_tags = json.loads(json_str)
                if not isinstance(new_tags, dict):
                    raise json.JSONDecodeError("Input is not a JSON object.", json_str, 0)
                self.tag_manager.import_tags(new_tags)
                yield event.plain_result(messages.MSG_TAGS_IMPORTED)
            except json.JSONDecodeError:
                yield event.plain_result("❌ 导入失败：输入的不是有效的JSON格式。")
            return

        if ":" in text:
            key, value = text.split(":", 1)
            key = key.strip()
            value = value.strip()
            if key and value:
                self.tag_manager.set_tag(key, value)
                yield event.plain_result(messages.MSG_TAG_SET.format(key=key))
            else:
                yield event.plain_result("❌ 格式错误，请输入 `/sd tag 关键词:替换内容`。")
            return
            
        yield event.plain_result("使用方法：\n- `/sd tag list`\n- `/sd tag add <名字> <提示词>`\n- `/sd tag 关键词:内容`\n- `/sd tag del 关键词`\n- `/sd tag import {...}`")

    @sd_group.command("check")
    async def handle_check(self, event: AstrMessageEvent):
        """Checks the connection status to the Stable Diffusion WebUI."""
        if await self.api_client.check_availability():
            yield event.plain_result(messages.MSG_WEBUI_AVAILABLE)
        else:
            yield event.plain_result(messages.MSG_WEBUI_UNAVAILABLE)

    @sd_group.command("conf")
    async def handle_conf(self, event: AstrMessageEvent):
        """Displays the current plugin configuration."""
        try:
            conf_str = json.dumps(self.config, indent=2, ensure_ascii=False)
            yield event.plain_result(f"Current Configuration:\n{conf_str}")
        except Exception as e:
            logger.error(f"Failed to display configuration: {e}")
            yield event.plain_result("Failed to display configuration.")
            
    @sd_group.command("verbose")
    async def handle_verbose(self, event: AstrMessageEvent):
        """Toggles verbose mode."""
        current_status = self.config.get("enable_show_positive_prompt", True)
        new_status = not current_status
        self.config["enable_show_positive_prompt"] = new_status
        feedback = "详细模式已开启。" if new_status else "详细模式已关闭。"
        yield event.plain_result(feedback)

    @sd_group.command("forward")
    async def handle_forward_toggle(self, event: AstrMessageEvent):
        """Toggles whether to send images as a new message or as a reply."""
        current_status = self.config.get("enable_forward_message", False)
        new_status = not current_status
        self.config["enable_forward_message"] = new_status
        feedback = "发送模式已切换为：新消息（不回复）。" if new_status else "发送模式已切换为：回复原消息。"
        yield event.plain_result(feedback)

    async def _permission_check(self, event: AstrMessageEvent) -> bool:
        """Checks if the command can be executed in the current context."""
        group_id = event.get_group_id()
        blacklist_groups = self.config.get("blacklist_groups", [])
        if group_id in blacklist_groups:
            logger.info(f"Command ignored in blacklisted group: {group_id}")
            return False
        return True

    async def _generate_and_send(self, event: AstrMessageEvent, final_prompt: str, image_b64: str = None, is_inspire: bool = False):
        """A unified helper to generate and send images."""
        try:
            group_id = event.get_group_id()
            
            # Determine generation method
            if is_inspire or image_b64 is None:
                # Inspire and Native commands use txt2img
                generated_images = await self.generator.generate_txt2img(final_prompt, group_id)
            else:
                # I2I command uses img2img
                generated_images = await self.generator.generate_img2img(image_b64, final_prompt, group_id)

            if not generated_images:
                yield event.plain_result(messages.MSG_API_ERROR)
                return

            # Process and send results
            processed_images = await self.generator.process_and_upscale_images(generated_images)
            image_components = [Comp.Image.from_base64(img) for img in processed_images]
            
            if self.config.get("enable_show_positive_prompt", True):
                yield event.plain_result(f"{messages.MSG_PROMPT_DISPLAY}: {final_prompt}")
            
            if self.config.get("enable_forward_message", False):
                yield event.send_result(image_components)
            else:
                yield event.chain_result(image_components)

        except Exception as e:
            logger.error(f"An error occurred during image generation/sending: {e}")
            yield event.plain_result(messages.MSG_UNKNOWN_ERROR)

    # --- Native Command ---
    @filter.command("原生画", alias={"native"})
    async def handle_native(self, event: AstrMessageEvent):
        """Handles the /原生画 command for direct txt2img generation."""
        if not await self._permission_check(event):
            return

        prompt_text = event.message_str.strip()
        if not prompt_text:
            yield event.plain_result(messages.MSG_NO_PROMPT_PROVIDED)
            return

        await event.send(event.plain_result(messages.MSG_GENERATING))
        final_prompt, _ = self.tag_manager.replace(prompt_text)
        
        async for result in self._generate_and_send(event, final_prompt):
            yield result

    # --- I2I Command ---
    @filter.command("i2i")
    async def handle_i2i(self, event: AstrMessageEvent):
        """Handles the /i2i command for image-to-image generation."""
        if not await self._permission_check(event):
            return

        try:
            image_b64, prompt_text = await self._extract_image_and_text(event)
        except ConnectionError:
            yield event.plain_result(messages.MSG_IMG_DOWNLOAD_FAILED)
            return

        if not image_b64:
            yield event.plain_result(messages.MSG_NO_IMAGE_PROVIDED)
            return

        await event.send(event.plain_result(messages.MSG_IMG2IMG_GENERATING))
        prompt_text, _ = self.tag_manager.replace(prompt_text)
        
        if self.config.get("enable_llm_prompt_generation", True):
            final_prompt = await self.llm_helper.generate_text_prompt(
                base_prompt=prompt_text,
                guidelines=self.config.get("prompt_guidelines", ""),
                prefix=self.config.get("llm_prompt_prefix", "")
            )
        else:
            final_prompt = prompt_text
        
        async for result in self._generate_and_send(event, final_prompt, image_b64=image_b64):
            yield result

    async def _extract_image_and_text(self, event: AstrMessageEvent) -> (str, str):
        """
        Extracts base64 image and text from a message event.
        Raises ConnectionError if image download fails.
        """
        image_b64 = None
        prompt_text = ""
        if event.message_obj and event.message_obj.message:
            for comp in event.message_obj.message:
                if isinstance(comp, Comp.Image) and hasattr(comp, 'url') and comp.url:
                    # Let the ConnectionError propagate up to the handler
                    image_b64 = await self.api_client.download_image_as_base64(comp.url)
                elif isinstance(comp, Comp.Plain):
                    prompt_text += comp.text + " "
        return image_b64, prompt_text.strip()

    # --- Inspire Command ---
    @filter.command("inspire")
    async def handle_inspire(self, event: AstrMessageEvent):
        """Handles the /inspire command for vision-based txt2img generation."""
        if not await self._permission_check(event):
            return

        try:
            image_b64, prompt_text = await self._extract_image_and_text(event)
        except ConnectionError:
            yield event.plain_result(messages.MSG_IMG_DOWNLOAD_FAILED)
            return

        if not image_b64:
            yield event.plain_result(messages.MSG_NO_IMAGE_PROVIDED)
            return

        await event.send(event.plain_result(messages.MSG_GENERATING))
        prompt_text, _ = self.tag_manager.replace(prompt_text)
        
        final_prompt = await self.llm_helper.generate_prompt_from_image(
            image_b64=image_b64,
            user_instruction=prompt_text,
            guidelines=self.config.get("prompt_guidelines", ""),
            prefix=self.config.get("llm_prompt_prefix", "")
        )

        if final_prompt is None:
            yield event.plain_result(messages.MSG_VISION_FALLBACK)
            final_prompt = await self.llm_helper.generate_text_prompt(
                base_prompt=prompt_text,
                guidelines=self.config.get("prompt_guidelines", ""),
                prefix=self.config.get("llm_prompt_prefix", "")
            )
        
        async for result in self._generate_and_send(event, final_prompt, is_inspire=True):
            yield result

    async def terminate(self):
        """Called when the plugin is unloaded/disabled to clean up resources."""
        if self.api_client:
            await self.api_client.close()
        logger.info("SDGen_wzken plugin terminated and resources cleaned up.")
