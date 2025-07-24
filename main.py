# astrbot_plugin_sdgen_wzken/main.py

"""
Plugin Name: SDGen_wzken
Author: wzken
Version: 2.3.6
Description: A smarter and more powerful image generation plugin for AstrBot using Stable Diffusion.
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Callable, Tuple, Coroutine

import astrbot.api.message_components as Comp
from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, filter, MessageEventResult
from astrbot.api.star import Context, Star, register

from .core.client import SDAPIClient
from .core.generation import GenerationManager
from .utils.tag_manager import TagManager
from .utils.llm_helper import LLMHelper
from .static import messages

@register("SDGen_wzken", "wzken", "A smarter and more powerful image generation plugin for AstrBot using Stable Diffusion.", "2.3.6")
class SDGeneratorWzken(Star):
    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.config = config
        
        logger.info("SDGen_wzken plugin loaded. Initializing services...")
        self._initialize_services()
        logger.info("SDGen_wzken plugin initialization complete.")

    def _initialize_services(self):
        """Initializes all core services and managers."""
        plugin_dir = Path(__file__).parent.resolve()
        self.api_client = SDAPIClient(self.config)
        self.generator = GenerationManager(self.config, self.api_client)
        tags_file = plugin_dir / "data" / "tags.json"
        tags_file.parent.mkdir(exist_ok=True)
        self.tag_manager = TagManager(str(tags_file))
        self.llm_helper = LLMHelper(self.context)

    # --- LLM Tool Definition ---
    @filter.llm_tool("create_sd_image")
    async def generate_image(self, event: AstrMessageEvent, prompt: str) -> MessageEventResult:
        """
        使用Stable Diffusion模型，根据用户的描述创作一幅全新的图像。
        当用户明确表示想要“画”、“绘制”、“创作”或“生成”图片时，应使用此工具。
        后续步骤会对用户的简单描述进行优化和丰富，因此即使描述很简单也应调用此工具。

        Args:
            prompt(string): 用户对所需图像的描述。这可以是一个简单的概念（如“一个女孩”），也可以是一个更详细的场景。如果用户提供了图片，可以结合图片内容来识别并丰富描述，以生成更准确的提示词。
        """
        if not await self._permission_check(event):
            yield event.plain_result("Sorry, I don't have permission to draw in this chat.")
            return

        await event.send(event.plain_result(messages.MSG_GENERATING))
        # The prompt from the LLM is assumed to be good, but we still run it through the tag manager for aliases.
        final_prompt, _ = self.tag_manager.replace(prompt)
        async for result in self._generate_and_send(event, final_prompt):
            yield result

    # --- Generic Setting Handlers (Refactored) ---

    async def _handle_toggle(self, event: AstrMessageEvent, config_keys: List[List[str]], name: str):
        """Generic helper to toggle a boolean setting."""
        # Determine current status from the first key
        key_path = config_keys[0]
        current_status = self.config.get(key_path[0])
        if len(key_path) > 1:
            current_status = self.config.get(key_path[0], {}).get(key_path[1], False)

        new_status = not current_status
        
        for path in config_keys:
            if len(path) > 1:
                if path[0] not in self.config: self.config[path[0]] = {}
                self.config[path[0]][path[1]] = new_status
            else:
                self.config[path[0]] = new_status
                
        feedback = f"{name}已开启。" if new_status else f"{name}已关闭。"
        yield event.plain_result(feedback)

    async def _handle_value_setting(
        self, event: AstrMessageEvent, config_paths: List[List[str]], name: str, 
        value_type: type, validation: Callable[[Any], bool] = None, 
        api_check: Coroutine = None, unit: str = ""
    ):
        """Generic helper to set a configuration value (numeric or string)."""
        value_str = event.message_str.strip()
        
        # Get current value from the first path for display
        key_path = config_paths[0]
        current_val_dict = self.config
        for key in key_path[:-1]:
            current_val_dict = current_val_dict.get(key, {})
        current_val = current_val_dict.get(key_path[-1], 'N/A')

        if not value_str:
            yield event.plain_result(f"当前{name}为：`{current_val}`{unit}。")
            return

        try:
            new_value = value_type(value_str)
            if validation and not validation(new_value):
                yield event.plain_result(f"❌ 无效的{name}值。")
                return
            
            if api_check:
                is_valid, error_msg = await api_check(new_value)
                if not is_valid:
                    yield event.plain_result(error_msg)
                    return

            for path in config_paths:
                target_dict = self.config
                for key in path[:-1]:
                    if key not in target_dict: target_dict[key] = {}
                    target_dict = target_dict[key]
                target_dict[path[-1]] = new_value
            
            yield event.plain_result(f"{name}已设置为：`{new_value}`{unit}。")

        except ValueError:
            yield event.plain_result(f"❌ 格式错误，请输入一个有效的 {value_type.__name__}。")
        except ConnectionError:
            yield event.plain_result(messages.MSG_WEBUI_UNAVAILABLE)
        except Exception as e:
            logger.error(f"Failed to set {name}: {e}")
            yield event.plain_result(f"设置{name}失败。")

    # --- Command Definitions ---
    @filter.command_group("sd")
    def sd_group(self):
        pass

    # --- Toggle Commands ---
    @sd_group.command("verbose")
    async def handle_verbose(self, event: AstrMessageEvent):
        """Toggles verbose mode."""
        async for result in self._handle_toggle(event, [["enable_show_positive_prompt"]], "详细输出模式"):
            yield result

    @sd_group.command("private")
    async def handle_private_toggle(self, event: AstrMessageEvent):
        """Toggles private message mode."""
        async for result in self._handle_toggle(event, [["enable_private_message"]], "私聊发送模式"):
            yield result

    @sd_group.command("forward_reply")
    async def handle_forward_reply_toggle(self, event: AstrMessageEvent):
        """Toggles forward reply mode."""
        async for result in self._handle_toggle(event, [["enable_forward_reply"]], "合并转发回复模式"):
            yield result

    @sd_group.command("upscale")
    async def handle_upscale_toggle(self, event: AstrMessageEvent):
        """Toggles image upscaling mode."""
        async for result in self._handle_toggle(event, [["enable_upscale"]], "图像增强（超分辨率放大）模式"):
            yield result

    @sd_group.command("llm_gen")
    async def handle_llm_gen_toggle(self, event: AstrMessageEvent):
        """Toggles LLM prompt generation feature."""
        async for result in self._handle_toggle(event, [["enable_llm_prompt_generation"]], "LLM生成提示词功能"):
            yield result

    @sd_group.command("hr_toggle")
    async def handle_hr_toggle(self, event: AstrMessageEvent):
        """Toggles High-resolution fix mode."""
        async for result in self._handle_toggle(event, [["default_params", "enable_hr"]], "高分辨率修复（Hires. fix）模式"):
            yield result

    @sd_group.command("restore_faces")
    async def handle_restore_faces(self, event: AstrMessageEvent):
        """Toggles face restoration."""
        keys = [["default_params", "restore_faces"], ["img2img_params", "restore_faces"]]
        async for result in self._handle_toggle(event, keys, "面部修复功能"):
            yield result

    @sd_group.command("tiling")
    async def handle_tiling(self, event: AstrMessageEvent):
        """Toggles seamless tiling."""
        keys = [["default_params", "tiling"], ["img2img_params", "tiling"]]
        async for result in self._handle_toggle(event, keys, "无缝平铺功能"):
            yield result

    @sd_group.command("i2i_include_init_images")
    async def handle_i2i_include_init_images(self, event: AstrMessageEvent):
        """Toggles including initial images in i2i output."""
        async for result in self._handle_toggle(event, [["img2img_params", "include_init_images"]], "图生图输出包含初始图片功能"):
            yield result

    # --- Value Setting Commands ---
    @sd_group.command("timeout")
    async def handle_session_timeout(self, event: AstrMessageEvent):
        """Sets the session timeout in seconds."""
        async for result in self._handle_value_setting(event, [["session_timeout"]], "会话超时时间", int, validation=lambda v: v > 0, unit=" 秒"):
            yield result

    @sd_group.command("txt2img_steps")
    async def handle_txt2img_steps(self, event: AstrMessageEvent):
        """Sets the steps for Text-to-Image."""
        async for result in self._handle_value_setting(event, [["default_params", "steps"]], "文生图生成步数", int, validation=lambda v: v > 0):
            yield result

    @sd_group.command("txt2img_batch")
    async def handle_txt2img_batch_size(self, event: AstrMessageEvent):
        """Sets the batch size for Text-to-Image."""
        async for result in self._handle_value_setting(event, [["default_params", "batch_size"]], "文生图批量生成数量", int, validation=lambda v: v > 0):
            yield result

    @sd_group.command("txt2img_iter")
    async def handle_txt2img_n_iter(self, event: AstrMessageEvent):
        """Sets the iteration count for Text-to-Image."""
        async for result in self._handle_value_setting(event, [["default_params", "n_iter"]], "文生图生成迭代次数", int, validation=lambda v: v > 0):
            yield result

    @sd_group.command("i2i_steps")
    async def handle_i2i_steps(self, event: AstrMessageEvent):
        """Sets the steps for Image-to-Image."""
        async for result in self._handle_value_setting(event, [["img2img_params", "steps"]], "图生图生成步数", int, validation=lambda v: v > 0):
            yield result

    @sd_group.command("i2i_batch")
    async def handle_i2i_batch_size(self, event: AstrMessageEvent):
        """Sets the batch size for Image-to-Image."""
        async for result in self._handle_value_setting(event, [["img2img_params", "batch_size"]], "图生图批量生成数量", int, validation=lambda v: v > 0):
            yield result

    @sd_group.command("i2i_iter")
    async def handle_i2i_n_iter(self, event: AstrMessageEvent):
        """Sets the iteration count for Image-to-Image."""
        async for result in self._handle_value_setting(event, [["img2img_params", "n_iter"]], "图生图生成迭代次数", int, validation=lambda v: v > 0):
            yield result

    @sd_group.command("i2i_denoise")
    async def handle_i2i_denoising_strength(self, event: AstrMessageEvent):
        """Sets the denoising strength for Image-to-Image."""
        async for result in self._handle_value_setting(event, [["img2img_params", "denoising_strength"]], "图生图重绘幅度", float, validation=lambda v: 0.0 <= v <= 1.0):
            yield result

    @sd_group.command("hr_scale")
    async def handle_hr_scale(self, event: AstrMessageEvent):
        """Sets the upscale factor for High-resolution fix."""
        async for result in self._handle_value_setting(event, [["default_params", "hr_scale"]], "高分辨率修复放大倍数", float, validation=lambda v: v > 0):
            yield result

    @sd_group.command("hr_steps")
    async def handle_hr_steps(self, event: AstrMessageEvent):
        """Sets the second pass steps for High-resolution fix."""
        async for result in self._handle_value_setting(event, [["default_params", "hr_second_pass_steps"]], "高分辨率修复二阶段步数", int, validation=lambda v: v >= 0):
            yield result

    @sd_group.command("seed")
    async def handle_seed(self, event: AstrMessageEvent):
        """Sets the generation seed."""
        keys = [["default_params", "seed"], ["img2img_params", "seed"]]
        async for result in self._handle_value_setting(event, keys, "生成种子", int):
            yield result

    @sd_group.command("i2i_image_cfg_scale")
    async def handle_i2i_image_cfg_scale(self, event: AstrMessageEvent):
        """Sets the Image CFG Scale for Image-to-Image."""
        async for result in self._handle_value_setting(event, [["img2img_params", "image_cfg_scale"]], "图生图图像CFG Scale", float, validation=lambda v: v >= 0):
            yield result

    # --- String/Complex Value Setting Commands ---
    @sd_group.command("txt2img_prefix")
    async def handle_txt2img_prefix(self, event: AstrMessageEvent):
        """Sets or queries the positive prompt prefix for Text-to-Image."""
        async for result in self._handle_value_setting(event, [["positive_prompt_global"]], "文生图正向提示词前缀", str):
            yield result

    @sd_group.command("llm_prefix")
    async def handle_llm_prefix(self, event: AstrMessageEvent):
        """Sets or queries the LLM prompt prefix."""
        async for result in self._handle_value_setting(event, [["llm_prompt_prefix"]], "LLM提示词前缀", str):
            yield result

    async def _check_sampler(self, name: str) -> Tuple[bool, str]:
        samplers = await self.api_client.get_samplers()
        if not any(s['name'] == name for s in samplers):
            return False, f"❌ 采样器 `{name}` 不存在。请使用 `/sd list_txt2img_samplers` 查看可用列表。"
        return True, ""

    @sd_group.command("set_txt2img_sampler")
    async def handle_set_txt2img_sampler(self, event: AstrMessageEvent):
        """Sets the sampler for Text-to-Image."""
        async for result in self._handle_value_setting(event, [["default_params", "sampler"]], "文生图采样器", str, api_check=self._check_sampler):
            yield result

    @sd_group.command("set_i2i_sampler")
    async def handle_set_i2i_sampler(self, event: AstrMessageEvent):
        """Sets the sampler for Image-to-Image."""
        async for result in self._handle_value_setting(event, [["img2img_params", "sampler"]], "图生图采样器", str, api_check=self._check_sampler):
            yield result

    async def _check_scheduler(self, name: str) -> Tuple[bool, str]:
        schedulers = await self.api_client.get_schedulers()
        if not any(s['name'] == name for s in schedulers):
            return False, f"❌ 调度器 `{name}` 不存在。请使用 `/sd list_txt2img_schedulers` 查看可用列表。"
        return True, ""

    @sd_group.command("set_txt2img_scheduler")
    async def handle_set_txt2img_scheduler(self, event: AstrMessageEvent):
        """Sets the scheduler for Text-to-Image."""
        async for result in self._handle_value_setting(event, [["default_params", "scheduler"]], "文生图调度器", str, api_check=self._check_scheduler):
            yield result

    @sd_group.command("set_i2i_scheduler")
    async def handle_set_i2i_scheduler(self, event: AstrMessageEvent):
        """Sets the scheduler for Image-to-Image."""
        async for result in self._handle_value_setting(event, [["img2img_params", "scheduler"]], "图生图调度器", str, api_check=self._check_scheduler):
            yield result

    async def _check_upscaler(self, name: str) -> Tuple[bool, str]:
        upscalers = await self.api_client.get_upscalers()
        if not any(u['name'] == name for u in upscalers):
            return False, f"❌ 放大器 `{name}` 不存在。请使用 `/sd list_upscalers` 查看可用列表。"
        return True, ""

    @sd_group.command("set_upscaler")
    async def handle_set_upscaler(self, event: AstrMessageEvent):
        """Sets the upscaling algorithm."""
        async for result in self._handle_value_setting(event, [["upscale_params", "upscaler"]], "上采样算法", str, api_check=self._check_upscaler):
            yield result

    @sd_group.command("hr_upscaler")
    async def handle_hr_upscaler(self, event: AstrMessageEvent):
        """Sets the upscaler for High-resolution fix."""
        async for result in self._handle_value_setting(event, [["default_params", "hr_upscaler"]], "高分辨率修复放大器", str, api_check=self._check_upscaler):
            yield result

    # --- Other Commands ---
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

    # --- List Commands ---
    async def _list_api_resource(self, event: AstrMessageEvent, resource_name: str, api_call: Coroutine, name_key: str = 'name', title_key: str = 'title'):
        """Generic helper to list resources from the API."""
        try:
            items = await api_call()
            if not items:
                yield event.plain_result(f"未找到任何可用的{resource_name}。")
                return
            
            if isinstance(items, dict) and 'loaded' in items: # Special case for embeddings
                item_names = "\n".join([f"- `{e}`" for e in items['loaded'].keys()])
            else:
                item_names = "\n".join([f"- `{item.get(title_key, item.get(name_key))}`" for item in items])
            
            yield event.plain_result(f"可用的{resource_name}：\n{item_names}")
        except ConnectionError:
            yield event.plain_result(messages.MSG_WEBUI_UNAVAILABLE)
        except Exception as e:
            logger.error(f"Failed to list {resource_name}: {e}")
            yield event.plain_result(f"获取{resource_name}列表失败。")

    @sd_group.command("list_models")
    async def handle_list_models(self, event: AstrMessageEvent):
        """Lists all available Stable Diffusion models."""
        async for result in self._list_api_resource(event, "Stable Diffusion 模型", self.api_client.get_sd_models):
            yield result

    @sd_group.command("list_loras")
    async def handle_list_loras(self, event: AstrMessageEvent):
        """Lists all available LoRA models."""
        async for result in self._list_api_resource(event, "LoRA 模型", self.api_client.get_loras):
            yield result

    @sd_group.command("list_embeddings")
    async def handle_list_embeddings(self, event: AstrMessageEvent):
        """Lists all available Embedding models."""
        async for result in self._list_api_resource(event, "Embedding 模型", self.api_client.get_embeddings):
            yield result

    @sd_group.command("list_txt2img_samplers")
    async def handle_list_txt2img_samplers(self, event: AstrMessageEvent):
        """Lists all available Text-to-Image samplers."""
        async for result in self._list_api_resource(event, "文生图采样器", self.api_client.get_samplers):
            yield result

    @sd_group.command("list_i2i_samplers")
    async def handle_list_i2i_samplers(self, event: AstrMessageEvent):
        """Lists all available Image-to-Image samplers."""
        async for result in self._list_api_resource(event, "图生图采样器", self.api_client.get_samplers):
            yield result

    @sd_group.command("list_upscalers")
    async def handle_list_upscalers(self, event: AstrMessageEvent):
        """Lists all available upscaling algorithms."""
        async for result in self._list_api_resource(event, "上采样算法", self.api_client.get_upscalers):
            yield result

    @sd_group.command("list_txt2img_schedulers")
    async def handle_list_txt2img_schedulers(self, event: AstrMessageEvent):
        """Lists all available Text-to-Image schedulers."""
        async for result in self._list_api_resource(event, "文生图调度器", self.api_client.get_schedulers):
            yield result

    @sd_group.command("list_i2i_schedulers")
    async def handle_list_i2i_schedulers(self, event: AstrMessageEvent):
        """Lists all available Image-to-Image schedulers."""
        async for result in self._list_api_resource(event, "图生图调度器", self.api_client.get_schedulers):
            yield result

    @sd_group.command("set_model")
    async def handle_set_model(self, event: AstrMessageEvent):
        """Sets the current Stable Diffusion base model."""
        model_name = event.message_str.strip()
        if not model_name:
            yield event.plain_result("❌ 请提供要设置的模型名称。")
            return
        try:
            await self.api_client.set_model(model_name)
            self.config["base_model"] = model_name
            yield event.plain_result(f"基础模型已设置为：`{model_name}`。")
        except ConnectionError:
            yield event.plain_result(messages.MSG_WEBUI_UNAVAILABLE)
        except Exception as e:
            logger.error(f"Failed to set model: {e}")
            yield event.plain_result(f"设置模型失败，请检查模型名称是否正确。")

    # --- Tag Management ---
    @sd_group.command("tag")
    async def handle_tag(self, event: AstrMessageEvent):
        """Manages local tags."""
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
            
        yield event.plain_result("使用方法：\n- `/sd tag list`\n- `/sd tag add <名字> <提示词>`\n- `/sd tag 关键词:内容`\n- `/sd tag del 关键词`\n- `/sd tag rename <旧名字> <新名字>`\n- `/sd tag search <关键词>`\n- `/sd tag import {...}`")

    @sd_group.command("tag rename")
    async def handle_tag_rename(self, event: AstrMessageEvent):
        """Renames a local tag."""
        text = event.message_str.strip()
        parts = text.split(maxsplit=2)
        if len(parts) == 3:
            old_name, new_name = parts[1], parts[2]
            if self.tag_manager.rename_tag(old_name, new_name):
                yield event.plain_result(f"标签 `{old_name}` 已成功重命名为 `{new_name}`。")
            else:
                yield event.plain_result(f"❌ 重命名失败：未找到名为 `{old_name}` 的标签。")
        else:
            yield event.plain_result("❌ 格式错误，请输入 `/sd tag rename <旧名字> <新名字>`。")

    @sd_group.command("tag search")
    async def handle_tag_search(self, event: AstrMessageEvent):
        """Fuzzy searches for local tags by name."""
        keyword = event.message_str.strip()
        if not keyword:
            yield event.plain_result("❌ 请提供要搜索的关键词。")
            return
        found_tags = self.tag_manager.fuzzy_search(keyword)
        if not found_tags:
            yield event.plain_result(f"未找到包含关键词 '{keyword}' 的本地标签。")
            return
        tag_list_str = "\n".join([f"- `{key}`: `{value}`" for key, value in found_tags.items()])
        yield event.plain_result(f"找到的本地标签：\n{tag_list_str}")

    # --- Core Generation Logic ---
    async def _permission_check(self, event: AstrMessageEvent) -> bool:
        """Checks if the command can be executed in the current context."""
        group_id = event.get_group_id()
        if group_id in self.config.get("blacklist_groups", []):
            logger.info(f"Command ignored in blacklisted group: {group_id}")
            return False
        return True

    async def _generate_and_send(self, event: AstrMessageEvent, final_prompt: str, image_b64: str = None, is_inspire: bool = False):
        """A unified helper to generate and send images based on user configuration."""
        try:
            group_id = event.get_group_id()
            user_id = event.get_sender_id()

            if is_inspire or image_b64 is None:
                generated_images = await self.generator.generate_txt2img(final_prompt, group_id)
            else:
                generated_images = await self.generator.generate_img2img(image_b64, final_prompt, group_id)

            if not generated_images:
                yield event.plain_result(messages.MSG_API_ERROR)
                return

            processed_images = await self.generator.process_and_upscale_images(generated_images)
            image_components = [Comp.Image.fromBase64(img) for img in processed_images]
            
            # Add prompt to the message chain if enabled
            if self.config.get("enable_show_positive_prompt", True):
                image_components.insert(0, Comp.Plain(f"{messages.MSG_PROMPT_DISPLAY}: {final_prompt}"))

            # --- New Sending Logic ---
            send_private = self.config.get("enable_private_message", False)
            use_forward_reply = self.config.get("enable_forward_reply", False)

            if send_private:
                # Send as a private message to the user
                yield event.send_result(image_components, user_id=user_id)
            elif use_forward_reply:
                # Send as a forwardable node in the original channel
                node = Comp.Node(
                    uin=event.message_obj.self_id,
                    name="SD 生成结果",
                    content=image_components
                )
                yield event.chain_result([node])
            else:
                # Default: send as a direct reply in the original channel
                yield event.chain_result(image_components)

        except Exception as e:
            logger.error(f"An error occurred during image generation/sending: {e}", exc_info=True)
            yield event.plain_result(messages.MSG_UNKNOWN_ERROR)

    @filter.command("原生画", alias={"native"})
    async def handle_native(self, event: AstrMessageEvent):
        """Handles direct txt2img generation."""
        if not await self._permission_check(event): return
        prompt_text = event.message_str.strip()
        if not prompt_text:
            yield event.plain_result(messages.MSG_NO_PROMPT_PROVIDED)
            return
        await event.send(event.plain_result(messages.MSG_GENERATING))
        final_prompt, _ = self.tag_manager.replace(prompt_text)
        async for result in self._generate_and_send(event, final_prompt):
            yield result

    @filter.command("i2i")
    async def handle_i2i(self, event: AstrMessageEvent):
        """Handles image-to-image generation."""
        if not await self._permission_check(event): return
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

    async def _extract_image_and_text(self, event: AstrMessageEvent) -> Tuple[str, str]:
        """Extracts base64 image and text from a message event."""
        image_b64, prompt_text = None, ""
        if event.message_obj and event.message_obj.message:
            for comp in event.message_obj.message:
                if isinstance(comp, Comp.Image) and hasattr(comp, 'url') and comp.url:
                    image_b64 = await self.api_client.download_image_as_base64(comp.url)
                elif isinstance(comp, Comp.Plain):
                    prompt_text += comp.text + " "
        return image_b64, prompt_text.strip()

    @filter.command("inspire")
    async def handle_inspire(self, event: AstrMessageEvent):
        """Handles vision-based txt2img generation."""
        if not await self._permission_check(event): return
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
            image_b64=image_b64, user_instruction=prompt_text,
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

    @sd_group.command("help")
    async def handle_help(self, event: AstrMessageEvent):
        """Displays all available commands and their descriptions as a forwardable message."""
        help_message = "SD绘图插件可用指令：\n\n"
        commands = [
            # Manually list commands for clarity and order
            "- `/sd check`: 检查Stable Diffusion WebUI服务是否可用。",
            "- `/sd conf`: 显示当前插件配置。",
            "- `/sd help`: 显示此帮助信息。",
            # Toggles
            "- `/sd verbose`: 切换详细输出模式。",
            "- `/sd private`: 切换私聊发送模式。",
            "- `/sd forward_reply`: 切换合并转发回复模式。",
            "- `/sd upscale`: 切换超分辨率放大模式。",
            "- `/sd llm_gen`: 切换LLM生成提示词功能。",
            "- `/sd hr_toggle`: 切换高分辨率修复模式。",
            "- `/sd restore_faces`: 切换面部修复功能。",
            "- `/sd tiling`: 切换无缝平铺功能。",
            # Txt2Img Params
            "- `/sd txt2img_steps [步数]`: 设置文生图步数。",
            "- `/sd txt2img_batch [数量]`: 设置文生图批次大小。",
            "- `/sd txt2img_iter [次数]`: 设置文生图迭代次数。",
            "- `/sd txt2img_prefix [前缀]`: 设置文生图正向提示词前缀。",
            # Img2Img Params
            "- `/sd i2i_steps [步数]`: 设置图生图步数。",
            "- `/sd i2i_batch [数量]`: 设置图生图批次大小。",
            "- `/sd i2i_iter [次数]`: 设置图生图迭代次数。",
            "- `/sd i2i_denoise [幅度]`: 设置图生图重绘幅度 (0-1)。",
            "- `/sd i2i_image_cfg_scale [值]`: 设置图生图图像CFG Scale。",
            "- `/sd i2i_include_init_images`: 切换图生图是否包含原图。",
            # Shared Params
            "- `/sd seed [种子]`: 设置生成种子 (-1为随机)。",
            "- `/sd timeout [秒数]`: 设置会话超时时间。",
            "- `/sd llm_prefix [前缀]`: 设置LLM提示词前缀。",
            # Hires. fix
            "- `/sd hr_scale [倍数]`: 设置高分修复放大倍数。",
            "- `/sd hr_steps [步数]`: 设置高分修复二阶段步数。",
            "- `/sd hr_upscaler [名称]`: 设置高分修复放大器。",
            # Model/Resource Management
            "- `/sd list_models`: 列出所有可用SD模型。",
            "- `/sd set_model <名称>`: 设置当前SD模型。",
            "- `/sd list_loras`: 列出所有可用LoRA模型。",
            "- `/sd list_embeddings`: 列出所有可用Embedding。",
            "- `/sd list_upscalers`: 列出所有可用放大器。",
            "- `/sd set_upscaler <名称>`: 设置默认放大器。",
            "- `/sd list_txt2img_samplers`: 列出文生图可用采样器。",
            "- `/sd set_txt2img_sampler <名称>`: 设置文生图采样器。",
            "- `/sd list_i2i_samplers`: 列出图生图可用采样器。",
            "- `/sd set_i2i_sampler <名称>`: 设置图生图采样器。",
            "- `/sd list_txt2img_schedulers`: 列出文生图可用调度器。",
            "- `/sd set_txt2img_scheduler <名称>`: 设置文生图调度器。",
            "- `/sd list_i2i_schedulers`: 列出图生图可用调度器。",
            "- `/sd set_i2i_scheduler <名称>`: 设置图生图调度器。",
            # Tag Management
            "- `/sd tag`: 管理本地关键词 (list, add, del, rename, search, import)。"
        ]
        help_message += "\n".join(sorted(commands))
        
        node = Comp.Node(
            uin=event.message_obj.self_id,
            name="SD 绘图插件帮助手册",
            content=[Comp.Plain(text=help_message)]
        )
        yield event.chain_result([node])

    async def terminate(self):
        """Called when the plugin is unloaded/disabled to clean up resources."""
        if self.api_client:
            await self.api_client.close()
        logger.info("SDGen_wzken plugin terminated and resources cleaned up.")
