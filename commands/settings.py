# astrbot_plugin_sdgen_v2/commands/settings.py

import json
from astrbot.api.star import Context
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.all import command_group, logger

from ..core.client import SDAPIClient
from ..utils.tag_manager import TagManager
from ..static import messages

class SettingsCommands:
    def __init__(self, context: Context, client: SDAPIClient, tag_manager: TagManager):
        self.context = context
        self.client = client
        self.tag_manager = tag_manager

    @command_group("sd")
    def sd_group(self):
        pass

    @sd_group.command("tag")
    async def handle_tag(self, event: AstrMessageEvent):
        """Manages local tags. Usage: /sd tag key:value, /sd tag del key, /sd tag list, /sd tag import {...}"""
        text = event.get_plain_text().strip()
        
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
            
        yield event.plain_result("使用方法：\n- `/sd tag list`\n- `/sd tag 关键词:内容`\n- `/sd tag del 关键词`\n- `/sd tag import {...}`")

    @sd_group.command("check")
    async def handle_check(self, event: AstrMessageEvent):
        """Checks the connection status to the Stable Diffusion WebUI."""
        if await self.client.check_availability():
            yield event.plain_result(messages.MSG_WEBUI_AVAILABLE)
        else:
            yield event.plain_result(messages.MSG_WEBUI_UNAVAILABLE)

    @sd_group.command("conf")
    async def handle_conf(self, event: AstrMessageEvent):
        """Displays the current plugin configuration."""
        try:
            # Accessing config directly from the context
            conf_str = json.dumps(self.context._config, indent=2, ensure_ascii=False)
            yield event.plain_result(f"Current Configuration:\n{conf_str}")
        except Exception as e:
            logger.error(f"Failed to display configuration: {e}")
            yield event.plain_result("Failed to display configuration.")
            
    @sd_group.command("verbose")
    async def handle_verbose(self, event: AstrMessageEvent):
        """Toggles verbose mode."""
        current_status = self.context._config.get("enable_show_positive_prompt", True)
        new_status = not current_status
        self.context._config["enable_show_positive_prompt"] = new_status
        # Note: AstrBot should handle saving the config automatically.
        feedback = "详细模式已开启。" if new_status else "详细模式已关闭。"
        yield event.plain_result(feedback)

    @sd_group.command("forward")
    async def handle_forward_toggle(self, event: AstrMessageEvent):
        """Toggles whether to send images as a new message or as a reply."""
        current_status = self.context._config.get("enable_forward_message", False)
        new_status = not current_status
        self.context._config["enable_forward_message"] = new_status
        feedback = "发送模式已切换为：新消息（不回复）。" if new_status else "发送模式已切换为：回复原消息。"
        yield event.plain_result(feedback)


def register_settings_commands(star_instance):
    """Registers the settings command handlers to the main star instance."""
    settings_handler = SettingsCommands(star_instance.context, star_instance.api_client, star_instance.tag_manager)
    
    star_instance.sd_group = settings_handler.sd_group
    star_instance.sd_group.add_command(settings_handler.handle_tag)
    star_instance.sd_group.add_command(settings_handler.handle_check)
    star_instance.sd_group.add_command(settings_handler.handle_conf)
    star_instance.sd_group.add_command(settings_handler.handle_verbose)
    star_instance.sd_group.add_command(settings_handler.handle_forward_toggle)
