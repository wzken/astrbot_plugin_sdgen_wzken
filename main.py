# astrbot_plugin_sdgen_wzken/main.py

"""
Plugin Name: SDGen_wzken
Author: wzken
Version: 3.3.0
Description: A smarter and more powerful image generation plugin for AstrBot using Stable Diffusion.
"""

import json
import base64
import io
from PIL import Image
from pathlib import Path
from typing import List, Dict, Any, Callable, Tuple, Coroutine, Optional

import astrbot.api.message_components as Comp
from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, filter, MessageEventResult
from astrbot.api.star import Context, Star, register
from astrbot.core.utils.session_waiter import session_waiter, SessionController

from .core.client import SDAPIClient
from .core.generation import GenerationManager
from .utils.tag_manager import TagManager
from .utils.llm_helper import LLMHelper
from .utils.sd_utils import SDUtils
from .static import messages

@register("SDGen_wzken", "wzken", "A smarter and more powerful image generation plugin for AstrBot using Stable Diffusion.", "3.3.0", "https://github.com/wzken/astrabot_plugin_sdgen_wzken")
class SDGeneratorWzken(Star):

    TOGGLES_MAP = [
        {"name": "详细输出模式", "keys": ["enable_show_positive_prompt"]},
        {"name": "私聊发送结果", "keys": ["enable_private_message"]},
        {"name": "合并转发消息", "keys": ["enable_forward_reply"]},
        {"name": "LLM生成提示词", "keys": ["enable_llm_prompt_generation"]},
        {"name": "面部修复", "keys": ["default_params.restore_faces", "img2img_params.restore_faces"]},
        {"name": "无缝平铺", "keys": ["default_params.tiling", "img2img_params.tiling"]},
        {"name": "图生图包含原图", "keys": ["img2img_params.include_init_images"]},
    ]

    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.config = config
        
        logger.info("SDGen_wzken plugin loaded. Initializing services...")
        self._initialize_services()
        logger.info("SDGen_wzken plugin initialization complete.")

    def _initialize_services(self):
        """Initializes all core services and managers."""
        # Ensure the data directory exists
        plugin_dir = Path(__file__).parent.resolve()
        tags_dir = plugin_dir / "data"
        tags_dir.mkdir(exist_ok=True)

        self.api_client = SDAPIClient(self.config)
        self.sd_utils = SDUtils(self.config, self.context)
        self.generator = GenerationManager(self.config, self.api_client, self.sd_utils)
        
        tags_file = tags_dir.joinpath("tags.json")
        self.tag_manager = TagManager(str(tags_file))
        
        self.llm_helper = LLMHelper(self.context)

    @filter.llm_tool("create_sd_image")
    async def generate_image_tool(self, event: AstrMessageEvent, prompt: str) -> MessageEventResult:
        '''使用Stable Diffusion模型，根据用户的描述创作一幅全新的图像。
        这个工具的核心任务是将用户的自然语言描述（无论简单或复杂）转换成一个专业、详细、适合AI绘画的英文提示词，并用它来生成图像。
        当用户明确表示想要“画”、“绘制”、“创作”或“生成”图片时，应优先使用此工具。

        Args:
            prompt (string): 用户对所需图像的核心描述。根据输入类型，按以下方式处理，不要包含质量标签：
                - **当只有文本时**: 将该描述作为核心生成依据（如“一个女孩”或“蔚蓝档案的小春在沙滩上”），自动补全合理细节并优化视觉表现。
                - **当同时有图片和文本时**: 将此文本视为用户的核心创作意图。必须分析图片内容，提取如背景、服装、角色细节、光照等视觉元素，并将这些信息与用户的文本意图结合，形成一个完整、详细的最终提示词。
        '''
        if not await self._permission_check(event):
            yield event.plain_result(messages.MSG_PERMISSION_DENIED)
            return

        # Send an immediate message to keep the event alive, mirroring the successful pattern from the /draw command.
        await event.send(event.plain_result(messages.MSG_GENERATING))
        
        replaced_prompt, replacements = self.tag_manager.replace(prompt)

        if replacements and self.config.get("enable_show_positive_prompt", False):
            replacement_str = "\n".join([f"- `{orig}` -> `{new}`" for orig, new in replacements])
            await event.send(event.plain_result(messages.MSG_TAG_REPLACEMENTS_APPLIED.format(replacement_str=replacement_str)))
        
        if self.config.get("enable_llm_prompt_generation", False):
            llm_prompt_prefix = self.config.get("llm_prompt_prefix", "")
            llm_prompt = f"{llm_prompt_prefix}{replaced_prompt}" if llm_prompt_prefix else replaced_prompt
            
            base_prompt = await self._get_llm_completion(
                event,
                prompt=llm_prompt,
                contexts=[{"role": "system", "content": messages.SYSTEM_PROMPT_SD}],
                error_message="LLM failed to generate prompt.",
                fallback_value=replaced_prompt
            )
        else:
            base_prompt = replaced_prompt
        
        async for result in self._generate_and_send(event, base_prompt):
            yield result

    # --- Refactored Setting Handlers ---
    async def _set_config_value(
        self, event: AstrMessageEvent, name: str, value: Any, 
        config_paths: List[List[str]], validation: Optional[Callable[[Any], bool]] = None, 
        api_check: Optional[Coroutine] = None, unit: str = ""
    ):
        """A generic helper to set a configuration value and save it."""
        key_path = config_paths[0]
        current_val_dict = self.config
        for key in key_path[:-1]:
            current_val_dict = current_val_dict.get(key, {})
        current_val = current_val_dict.get(key_path[-1], 'N/A')

        if value is None:
            yield event.plain_result(messages.MSG_CURRENT_VALUE.format(name=name, current_val=current_val, unit=unit))
            return

        try:
            if validation and not validation(value):
                yield event.plain_result(messages.MSG_INVALID_VALUE.format(name=name))
                return
            
            if api_check:
                is_valid, error_msg = await api_check(value)
                if not is_valid:
                    yield event.plain_result(error_msg)
                    return

            for path_list in config_paths:
                target_dict = self.config
                for i, key in enumerate(path_list):
                    if i == len(path_list) - 1:
                        target_dict[key] = value
                    else:
                        if key not in target_dict:
                            target_dict[key] = {}
                        target_dict = target_dict[key]
            
            self.config.save_config()
            yield event.plain_result(messages.MSG_VALUE_SET.format(name=name, value=value, unit=unit))

        except ConnectionError:
            yield event.plain_result(messages.MSG_WEBUI_UNAVAILABLE)
        except Exception as e:
            logger.error(f"Failed to set {name}: {e}")
            yield event.plain_result(messages.MSG_SET_FAILED.format(name=name))

    # --- Command Definitions ---
    @filter.command_group("sd")
    def sd_group(self):
        '''SD绘图插件指令组'''
        pass


    # --- Value Setting Commands (Refactored with param injection) ---
    @sd_group.command("timeout")
    async def handle_session_timeout(self, event: AstrMessageEvent, value: Optional[int] = None):
        '''设置或查询会话超时时间（秒）'''
        async for r in self._set_config_value(event, "会话超时", value, [["session_timeout"]], validation=self.sd_utils.is_positive_int, unit="秒"): yield r
    @sd_group.command("steps")
    async def handle_txt2img_steps(self, event: AstrMessageEvent, value: Optional[int] = None):
        '''设置或查询文生图步数'''
        async for r in self._set_config_value(event, "文生图步数", value, [["default_params", "steps"]], validation=self.sd_utils.is_positive_int): yield r
    @sd_group.command("batch")
    async def handle_txt2img_batch_size(self, event: AstrMessageEvent, value: Optional[int] = None):
        '''设置或查询文生图批量生成数'''
        async for r in self._set_config_value(event, "文生图批量数", value, [["default_params", "batch_size"]], validation=self.sd_utils.is_positive_int): yield r
    @sd_group.command("iter")
    async def handle_txt2img_n_iter(self, event: AstrMessageEvent, value: Optional[int] = None):
        '''设置或查询文生图迭代次数'''
        async for r in self._set_config_value(event, "文生图迭代数", value, [["default_params", "n_iter"]], validation=self.sd_utils.is_positive_int): yield r
    @sd_group.command("i2i steps")
    async def handle_i2i_steps(self, event: AstrMessageEvent, value: Optional[int] = None):
        '''设置或查询图生图步数'''
        async for r in self._set_config_value(event, "图生图步数", value, [["img2img_params", "steps"]], validation=self.sd_utils.is_positive_int): yield r
    @sd_group.command("i2i batch")
    async def handle_i2i_batch_size(self, event: AstrMessageEvent, value: Optional[int] = None):
        '''设置或查询图生图批量生成数'''
        async for r in self._set_config_value(event, "图生图批量数", value, [["img2img_params", "batch_size"]], validation=self.sd_utils.is_positive_int): yield r
    @sd_group.command("i2i iter")
    async def handle_i2i_n_iter(self, event: AstrMessageEvent, value: Optional[int] = None):
        '''设置或查询图生图迭代次数'''
        async for r in self._set_config_value(event, "图生图迭代数", value, [["img2img_params", "n_iter"]], validation=self.sd_utils.is_positive_int): yield r
    @sd_group.command("i2i denoise")
    async def handle_i2i_denoising_strength(self, event: AstrMessageEvent, value: Optional[float] = None):
        '''设置或查询图生图重绘幅度'''
        async for r in self._set_config_value(event, "图生图重绘幅度", value, [["img2img_params", "denoising_strength"]], validation=self.sd_utils.is_valid_denoising_strength): yield r
    @sd_group.command("hr_scale")
    async def handle_hr_scale(self, event: AstrMessageEvent, value: Optional[float] = None):
        '''设置或查询高分辨率修复的放大倍数'''
        async for r in self._set_config_value(event, "高分修复放大倍数", value, [["default_params", "hr_scale"]], validation=self.sd_utils.is_positive_int): yield r
    @sd_group.command("hr_steps")
    async def handle_hr_steps(self, event: AstrMessageEvent, value: Optional[int] = None):
        '''设置或查询高分辨率修复的第二阶段步数'''
        async for r in self._set_config_value(event, "高分修复二阶段步数", value, [["default_params", "hr_second_pass_steps"]], validation=self.sd_utils.is_non_negative_int): yield r
    @sd_group.command("hr_denoise")
    async def handle_hr_denoising_strength(self, event: AstrMessageEvent, value: Optional[float] = None):
        '''设置或查询高分辨率修复的重绘幅度'''
        async for r in self._set_config_value(event, "高分修复重绘幅度", value, [["default_params", "hr_denoising_strength"]], validation=self.sd_utils.is_valid_hr_denoising_strength): yield r
    @sd_group.command("seed")
    async def handle_seed(self, event: AstrMessageEvent, value: Optional[int] = None):
        '''设置或查询生成种子，-1为随机'''
        keys = [["default_params", "seed"], ["img2img_params", "seed"]]
        async for r in self._set_config_value(event, "生成种子", value, keys, validation=self.sd_utils.is_valid_seed): yield r
    @sd_group.command("i2i image_cfg_scale")
    async def handle_i2i_image_cfg_scale(self, event: AstrMessageEvent, value: Optional[float] = None):
        '''设置或查询图生图的图像CFG'''
        async for r in self._set_config_value(event, "图生图图像CFG", value, [["img2img_params", "image_cfg_scale"]], validation=self.sd_utils.is_valid_image_cfg_scale): yield r

    @sd_group.command("res")
    async def handle_txt2img_res(self, event: AstrMessageEvent, value: Optional[str] = None):
        '''设置或查询文生图分辨率，格式：宽x高'''
        if value is None:
            w = self.config.get("default_params", {}).get("width", "N/A")
            h = self.config.get("default_params", {}).get("height", "N/A")
            yield event.plain_result(messages.MSG_CURRENT_RESOLUTION.format(name="文生图", width=w, height=h, auto_note=""))
            return
        
        try:
            width_str, height_str = value.lower().split('x')
            width, height = int(width_str), int(height_str)
            if not self.sd_utils.validate_resolution(width, height):
                yield event.plain_result(messages.MSG_INVALID_RESOLUTION)
                return
            
            if "default_params" not in self.config: self.config["default_params"] = {}
            self.config["default_params"]["width"] = width
            self.config["default_params"]["height"] = height
            self.config.save_config()
            yield event.plain_result(messages.MSG_RESOLUTION_SET.format(name="文生图", width=width, height=height))
        except (ValueError, IndexError):
            yield event.plain_result(messages.MSG_RESOLUTION_FORMAT_ERROR)

    @sd_group.command("i2i res")
    async def handle_i2i_res(self, event: AstrMessageEvent, value: Optional[str] = None):
        '''设置或查询图生图分辨率，格式：宽x高'''
        if value is None:
            w = self.config.get("img2img_params", {}).get("width", "自动")
            h = self.config.get("img2img_params", {}).get("height", "自动")
            yield event.plain_result(messages.MSG_CURRENT_RESOLUTION.format(name="图生图", width=w, height=h, auto_note=" (自动模式下此设置可能被覆盖)"))
            return
        
        try:
            width_str, height_str = value.lower().split('x')
            width, height = int(width_str), int(height_str)
            if not self.sd_utils.validate_resolution(width, height):
                yield event.plain_result(messages.MSG_INVALID_RESOLUTION)
                return
            
            if "img2img_params" not in self.config: self.config["img2img_params"] = {}
            self.config["img2img_params"]["width"] = width
            self.config["img2img_params"]["height"] = height
            self.config.save_config()
            yield event.plain_result(messages.MSG_RESOLUTION_SET.format(name="图生图", width=width, height=height))
        except (ValueError, IndexError):
            yield event.plain_result(messages.MSG_RESOLUTION_FORMAT_ERROR)

    # --- String/Complex Value Setting Commands ---
    @sd_group.command("prefix")
    async def handle_txt2img_prefix(self, event: AstrMessageEvent, *, value: Optional[str] = None):
        '''设置或查询文生图的全局正向提示词前缀'''
        async for r in self._set_config_value(event, "文生图正向提示词前缀", value, [["positive_prompt_global"]]): yield r
    @sd_group.command("i2i prefix")
    async def handle_i2i_prefix(self, event: AstrMessageEvent, *, value: Optional[str] = None):
        '''设置或查询图生图的全局正向提示词前缀'''
        async for r in self._set_config_value(event, "图生图正向提示词前缀", value, [["positive_prompt_i2i"]]): yield r
    @sd_group.command("llm prefix")
    async def handle_llm_prefix(self, event: AstrMessageEvent, *, value: Optional[str] = None):
        '''设置或查询LLM生成提示词时的前缀'''
        async for r in self._set_config_value(event, "LLM提示词前缀", value, [["llm_prompt_prefix"]]): yield r

    # --- Interactive/Session Commands ---
    async def _interactive_selector(
        self, event: AstrMessageEvent, resource_type: str, 
        fetch_coro: Coroutine, config_path: List[str], name_key: str = 'name',
        on_select: Optional[Callable[[Any], Coroutine[Any, Any, Tuple[bool, str]]]] = None
    ):
        """A generic interactive selector for resources like models, samplers, etc."""
        try:
            items = await fetch_coro()
            if not items:
                await event.send(event.plain_result(messages.MSG_FETCH_LIST_FAILED.format(resource_type=resource_type)))
                return

            item_names = [item.get(name_key, "N/A") if isinstance(item, dict) else item for item in items]
            
            current_val_dict = self.config
            for key in config_path[:-1]:
                current_val_dict = current_val_dict.get(key, {})
            current_val = current_val_dict.get(config_path[-1], 'N/A')

            prompt_lines = [messages.MSG_CURRENT_RESOURCE_PROMPT.format(resource_type=resource_type, current_val=current_val), messages.MSG_SELECT_NEW_RESOURCE_PROMPT.format(resource_type=resource_type)]
            prompt_lines.extend([f"{i+1}. {name}" for i, name in enumerate(item_names)])
            prompt_lines.append(messages.MSG_EXIT_PROMPT)
            
            await event.send(event.plain_result("\n".join(prompt_lines)))

            @session_waiter(timeout=self.config.get("session_timeout", 60))
            async def selector_waiter(controller: SessionController, inner_event: AstrMessageEvent):
                choice = inner_event.message_str.strip()

                if choice.lower() in ["退出", "exit", "quit", "cancel"]:
                    await inner_event.send(inner_event.plain_result(messages.MSG_OPERATION_CANCELLED))
                    controller.stop()
                    return

                try:
                    choice_idx = int(choice) - 1
                    if not (0 <= choice_idx < len(item_names)):
                        raise ValueError("Index out of range.")
                    
                    chosen_item = item_names[choice_idx]

                    if on_select:
                        success, message = await on_select(chosen_item)
                        if not success:
                            await inner_event.send(inner_event.plain_result(message))
                            controller.stop()
                            return
                    
                    target_dict = self.config
                    for key in config_path[:-1]:
                        if key not in target_dict: target_dict[key] = {}
                        target_dict = target_dict[key]
                    target_dict[config_path[-1]] = chosen_item
                    self.config.save_config()

                    await inner_event.send(inner_event.plain_result(messages.MSG_RESOURCE_SET.format(resource_type=resource_type, chosen_item=chosen_item)))
                    controller.stop()

                except ValueError:
                    await inner_event.send(inner_event.plain_result(messages.MSG_INVALID_CHOICE))
                    controller.keep(timeout=60, reset_timeout=True)
                except Exception as e:
                    logger.error(f"Error setting {resource_type} in session: {e}")
                    await inner_event.send(inner_event.plain_result(messages.MSG_ERROR_SETTING_RESOURCE.format(resource_type=resource_type, e=e)))
                    controller.stop()

            await selector_waiter(event)

        except ConnectionError:
            await event.send(event.plain_result(messages.MSG_WEBUI_UNAVAILABLE))
        except TimeoutError:
            await event.send(event.plain_result(messages.MSG_TIMEOUT_ERROR))
        except Exception as e:
            logger.error(f"Interactive selector for {resource_type} error: {e}", exc_info=True)
            await event.send(event.plain_result(messages.MSG_ERROR_OCCURRED.format(e=e)))
        finally:
            event.stop_event()

    @sd_group.command("model")
    async def handle_model_selection(self, event: AstrMessageEvent):
        '''交互式选择并设置基础模型'''
        async def set_model_in_api(model_title: str) -> Tuple[bool, str]:
            """Callback to set the model via API after user selection."""
            await event.send(event.plain_result(messages.MSG_MODEL_SWITCHING.format(model_title=model_title)))
            return await self.api_client.set_sd_model(model_title)

        await self._interactive_selector(
            event, 
            "模型", 
            self.api_client.get_sd_models, 
            ["base_model"], 
            name_key='title',
            on_select=set_model_in_api
        )

    @sd_group.command("sampler")
    async def handle_sampler_selection(self, event: AstrMessageEvent):
        '''交互式选择并设置采样器'''
        await self._select_sampler_or_scheduler(event, "采样器", self.api_client.get_samplers, "default_params")

    @sd_group.command("scheduler")
    async def handle_scheduler_selection(self, event: AstrMessageEvent):
        '''交互式选择并设置调度器'''
        await self._select_sampler_or_scheduler(event, "调度器", self.api_client.get_schedulers, "default_params")

    @sd_group.command("upscaler")
    async def handle_upscaler_menu(self, event: AstrMessageEvent):
        '''管理图像放大设置'''
        
        def get_menu_text():
            mode = self.config.get("upscaling_mode", "post")
            enabled = self.config.get("enable_upscale", False)
            
            mode_text = "后期处理放大" if mode == "post" else "高分辨率修复"
            enabled_text = "✅ 开" if enabled else "❌ 关"
            
            return messages.MSG_UPSCALER_MENU.format(mode_text=mode_text, enabled_text=enabled_text)

        await event.send(event.plain_result(get_menu_text()))

        @session_waiter(timeout=self.config.get("session_timeout", 60))
        async def menu_waiter(controller: SessionController, inner_event: AstrMessageEvent):
            choice = inner_event.message_str.strip()

            if choice == "1": # 切换模式
                current_mode = self.config.get("upscaling_mode", "post")
                new_mode = "hires" if current_mode == "post" else "post"
                self.config["upscaling_mode"] = new_mode
                
                is_upscale_enabled = self.config.get("enable_upscale", False)
                if "default_params" in self.config:
                    self.config["default_params"]["enable_hr"] = is_upscale_enabled and new_mode == "hires"
                self.config.save_config()

                await inner_event.send(inner_event.plain_result(messages.MSG_MODE_SWITCHED.format(menu_text=get_menu_text())))
                controller.keep(timeout=60, reset_timeout=True)

            elif choice == "2": # 选择算法
                controller.stop()
                mode = self.config.get("upscaling_mode", "post")
                if mode == "post":
                    await self._interactive_selector(inner_event, "后期处理放大器", self.api_client.get_upscalers, ["upscale_params", "upscaler"])
                else: # hires
                    await self._interactive_selector(inner_event, "高分辨率修复放大器", self.api_client.get_latent_upscalers, ["default_params", "hr_upscaler"])

            elif choice == "3": # 开/关
                new_status = not self.config.get("enable_upscale", False)
                self.config["enable_upscale"] = new_status
                if "default_params" in self.config:
                    self.config["default_params"]["enable_hr"] = new_status if self.config.get("upscaling_mode") == "hires" else False
                self.config.save_config()
                
                await inner_event.send(inner_event.plain_result(messages.MSG_TOGGLE_STATUS_SET.format(menu_text=get_menu_text())))
                controller.keep(timeout=60, reset_timeout=True)

            elif choice.lower() in ["退出", "exit", "quit", "cancel"]:
                await inner_event.send(inner_event.plain_result(messages.MSG_OPERATION_CANCELLED))
                controller.stop()
            
            else:
                await inner_event.send(inner_event.plain_result(messages.MSG_INVALID_CHOICE))
                controller.keep(timeout=60, reset_timeout=True)

        try:
            await menu_waiter(event)
        except TimeoutError:
            await event.send(event.plain_result(messages.MSG_TIMEOUT_ERROR))
        except Exception as e:
            logger.error(f"Upscaler menu error: {e}", exc_info=True)
            await event.send(event.plain_result(messages.MSG_ERROR_OCCURRED.format(e=e)))
        finally:
            event.stop_event()

    @sd_group.command("设置")
    async def handle_settings(self, event: AstrMessageEvent):
        '''交互式开启或关闭各项功能'''
        
        def get_toggle_status_text():
            lines = [messages.MSG_SETTINGS_SELECT_TOGGLE]
            for i, toggle in enumerate(self.TOGGLES_MAP):
                # For display, use the first key path to get current status
                key_path_str = toggle["keys"][0]
                keys = key_path_str.split('.')
                current_status = self.config
                for key in keys:
                    current_status = current_status.get(key, {})
                
                status_str = "✅ 开" if current_status else "❌ 关"
                lines.append(f"{i+1}. {toggle['name']} (当前: {status_str})")
            lines.append(messages.MSG_SETTINGS_EXIT_PROMPT)
            return "\n".join(lines)

        await event.send(event.plain_result(get_toggle_status_text()))

        @session_waiter(timeout=self.config.get("session_timeout", 60))
        async def settings_waiter(controller: SessionController, inner_event: AstrMessageEvent):
            choice = inner_event.message_str.strip()

            if choice.lower() in ["退出", "exit", "quit", "cancel"]:
                await inner_event.send(inner_event.plain_result(messages.MSG_EXIT_SETTINGS))
                controller.stop()
                return

            try:
                choice_idx = int(choice) - 1
                if not (0 <= choice_idx < len(self.TOGGLES_MAP)):
                    raise ValueError("Index out of range.")
                
                toggle_to_change = self.TOGGLES_MAP[choice_idx]
                
                # Get current status using the first key path
                first_key_path_str = toggle_to_change["keys"][0]
                first_keys = first_key_path_str.split('.')
                current_status = self.config
                for key in first_keys:
                    current_status = current_status.get(key, {})
                
                new_status = not current_status

                for key_path_str in toggle_to_change["keys"]:
                    keys = key_path_str.split('.')
                    target_dict = self.config
                    for i, key in enumerate(keys):
                        if i == len(keys) - 1:
                            target_dict[key] = new_status
                        else:
                            if key not in target_dict:
                                target_dict[key] = {}
                            target_dict = target_dict[key]
                self.config.save_config()
                
                status_feedback = "开启" if new_status else "关闭"
                await inner_event.send(inner_event.plain_result(messages.MSG_TOGGLE_FEEDBACK.format(toggle_name=toggle_to_change['name'], status_feedback=status_feedback)))
                
                await inner_event.send(inner_event.plain_result(get_toggle_status_text()))
                controller.keep(timeout=60, reset_timeout=True)

            except ValueError:
                await inner_event.send(inner_event.plain_result(messages.MSG_INVALID_CHOICE))
                controller.keep(timeout=60, reset_timeout=True)
            except Exception as e:
                logger.error(f"Error in settings session: {e}")
                await inner_event.send(inner_event.plain_result(messages.MSG_ERROR_OCCURRED.format(e=e)))
                controller.stop()

        try:
            await settings_waiter(event)
        except TimeoutError:
            await event.send(event.plain_result(messages.MSG_TIMEOUT_ERROR))
        except Exception as e:
            logger.error(f"handle_settings error: {e}", exc_info=True)
            await event.send(event.plain_result(messages.MSG_ERROR_OCCURRED.format(e=e)))
        finally:
            event.stop_event()

    # --- Other Commands ---
    @sd_group.command("check")
    async def handle_check(self, event: AstrMessageEvent):
        '''检查Stable Diffusion WebUI服务是否可用'''
        if await self.api_client.check_availability(): yield event.plain_result(messages.MSG_WEBUI_AVAILABLE)
        else: yield event.plain_result(messages.MSG_WEBUI_UNAVAILABLE)

    @sd_group.command("conf")
    async def handle_conf(self, event: AstrMessageEvent):
        '''显示当前所有配置参数'''
        try:
            t2i_params = self.sd_utils.get_generation_params_str()
            i2i_params = self.sd_utils.get_img2img_params_str()
            upscale_params = self.sd_utils.get_upscale_params_str()
            
            conf_str = (
                f"{messages.MSG_TXT2IMG_PARAMS_TITLE}\n{t2i_params}\n\n"
                f"{messages.MSG_IMG2IMG_PARAMS_TITLE}\n{i2i_params}\n\n"
                f"{messages.MSG_UPSCALE_PARAMS_TITLE}\n{upscale_params}"
            )
            yield event.plain_result(conf_str)
        except Exception as e:
            logger.error(f"Failed to display configuration: {e}")
            yield event.plain_result(messages.MSG_GET_CONFIG_FAILED)

    # --- List Commands ---
    @sd_group.command("list_loras")
    async def handle_list_loras(self, event: AstrMessageEvent):
        '''列出所有可用的LoRA模型'''
        async for r in self._list_api_resource(event, "LoRA模型", self.api_client.get_loras): yield r
    @sd_group.command("list_embeddings")
    async def handle_list_embeddings(self, event: AstrMessageEvent):
        '''列出所有可用的Embedding模型'''
        async for r in self._list_api_resource(event, "Embedding", self.api_client.get_embeddings): yield r

    async def _list_api_resource(self, event: AstrMessageEvent, resource_name: str, api_call: Coroutine, name_key: str = 'name', title_key: str = 'title'):
        try:
            items = await api_call()
            if not items:
                yield event.plain_result(messages.MSG_RESOURCE_NOT_FOUND.format(resource_name=resource_name))
                return
            
            if isinstance(items, dict) and 'loaded' in items:
                item_names = "\n".join([f"- `{e}`" for e in items['loaded'].keys()])
            else:
                item_names = "\n".join([f"- `{item.get(title_key, item.get(name_key))}`" for item in items])
            
            yield event.plain_result(messages.MSG_AVAILABLE_RESOURCE.format(resource_name=resource_name, item_names=item_names))
        except ConnectionError:
            yield event.plain_result(messages.MSG_WEBUI_UNAVAILABLE)
        except Exception as e:
            logger.error(f"Failed to list {resource_name}: {e}")
            yield event.plain_result(messages.MSG_FETCH_LIST_FAILED.format(resource_name=resource_name))

    # --- Tag Management ---
    @sd_group.command("tag")
    async def handle_tag(self, event: AstrMessageEvent, *, text: str):
        '''管理本地关键词替换。用法: /sd tag <list|add|del|import|关键词:内容>'''
        if not text:
            yield event.plain_result(messages.MSG_TAG_HELP_USAGE)
            return
        if text == "list":
            all_tags = self.tag_manager.get_all()
            if not all_tags:
                yield event.plain_result(messages.MSG_NO_LOCAL_TAGS)
                return
            tag_list_str = "\n".join([f"- `{k}`: `{v}`" for k, v in all_tags.items()])
            yield event.plain_result(messages.MSG_TAGS_SAVED.format(tag_list_str=tag_list_str))
            return
        if text.startswith("del "):
            key_to_del = text[4:].strip()
            if self.tag_manager.del_tag(key_to_del): yield event.plain_result(messages.MSG_TAG_DELETED.format(key=key_to_del))
            else: yield event.plain_result(messages.MSG_TAG_NOT_FOUND.format(key=key_to_del))
            return
        if text.startswith("add "):
            parts = text[4:].strip().split(maxsplit=1)
            if len(parts) == 2:
                self.tag_manager.set_tag(parts[0], parts[1])
                yield event.plain_result(messages.MSG_TAG_SET.format(key=parts[0]))
            else: yield event.plain_result(messages.MSG_TAG_ADD_FORMAT_ERROR)
            return
        if text.startswith("import "):
            json_str = text[7:].strip()
            try:
                new_tags = json.loads(json_str)
                if not isinstance(new_tags, dict): raise json.JSONDecodeError(messages.ERROR_NOT_JSON_OBJECT, json_str, 0)
                self.tag_manager.import_tags(new_tags)
                yield event.plain_result(messages.MSG_TAGS_IMPORTED)
            except json.JSONDecodeError: yield event.plain_result(messages.MSG_IMPORT_FAILED_INVALID_JSON)
            return
        if ":" in text:
            key, value = text.split(":", 1)
            key, value = key.strip(), value.strip()
            if key and value:
                self.tag_manager.set_tag(key, value)
                yield event.plain_result(messages.MSG_TAG_SET.format(key=key))
            else: yield event.plain_result(messages.MSG_TAG_SET_FORMAT_ERROR)
            return
        yield event.plain_result(messages.MSG_TAG_HELP_USAGE)

    @sd_group.command("tag rename")
    async def handle_tag_rename(self, event: AstrMessageEvent, old_name: str, new_name: str):
        '''重命名一个本地关键词'''
        if self.tag_manager.rename_tag(old_name, new_name):
            yield event.plain_result(messages.MSG_TAG_RENAMED.format(old_name=old_name, new_name=new_name))
        else:
            yield event.plain_result(messages.MSG_TAG_RENAME_FAILED_NOT_FOUND.format(old_name=old_name))

    @sd_group.command("tag search")
    async def handle_tag_search(self, event: AstrMessageEvent, *, keyword: str):
        '''模糊搜索本地关键词'''
        if not keyword:
            yield event.plain_result(messages.MSG_TAG_SEARCH_NO_KEYWORD)
            return
        found_tags = self.tag_manager.fuzzy_search(keyword)
        if not found_tags:
            yield event.plain_result(messages.MSG_TAG_NOT_FOUND_KEYWORD.format(keyword=keyword))
            return
        tag_list_str = "\n".join([f"- `{k}`: `{v}`" for k, v in found_tags.items()])
        yield event.plain_result(messages.MSG_TAGS_FOUND.format(tag_list_str=tag_list_str))

    # --- Core Generation Logic ---
    def _extract_prompt_from_message(self, message: str, aliases: List[str]) -> str:
        """Removes a command alias from the start of a message string."""
        message_lower = message.lower()
        # Sort aliases by length descending to match longer ones first
        sorted_aliases = sorted(aliases, key=len, reverse=True)
        for alias in sorted_aliases:
            # Check for command with and without slash
            for command in [f"/{alias}", alias]:
                if message_lower.startswith(command.lower()):
                    # Ensure there's a space after the command or it's the whole message
                    if len(message) == len(command) or message[len(command)].isspace():
                        return message[len(command):].strip()
        return message

    async def _permission_check(self, event: AstrMessageEvent) -> bool:
        group_id = event.get_group_id()
        if group_id in self.config.get("blacklist_groups", []):
            logger.info(f"Command ignored in blacklisted group: {group_id}")
            return False
        return True

    async def _generate_and_send(self, event: AstrMessageEvent, base_prompt: str, image_info: dict = None, is_inspire: bool = False, is_native: bool = False):
        try:
            group_id = event.get_group_id()
            is_i2i = image_info is not None

            full_positive_prompt, full_negative_prompt = self.sd_utils.get_full_prompts(
                base_prompt, group_id, is_i2i, is_native
            )

            if is_inspire:
                generated_images = await self.generator.generate_txt2img(full_positive_prompt, full_negative_prompt)
            elif is_i2i:
                generated_images = await self.generator.generate_img2img(image_info, full_positive_prompt, full_negative_prompt)
            else:
                generated_images = await self.generator.generate_txt2img(full_positive_prompt, full_negative_prompt)

            if not generated_images:
                await event.send(event.plain_result(messages.MSG_API_ERROR))
                return
 
            processed_images = await self.generator.process_and_upscale_images(generated_images)
            
            await self.sd_utils.send_image_results(event, processed_images, full_positive_prompt)

        except Exception as e:
            logger.error(f"An error occurred during image generation/sending: {e}", exc_info=True)
            yield event.plain_result(messages.MSG_UNKNOWN_ERROR)

    async def _get_llm_completion(self, event: AstrMessageEvent, prompt: str, contexts: List[Dict[str, str]], error_message: str, fallback_value: str) -> str:
        """Helper to get LLM completion with error handling and fallback."""
        try:
            llm_response = await self.context.get_using_provider().text_chat(
                prompt=prompt,
                contexts=contexts
            )
            if llm_response.role != "assistant" or not llm_response.completion_text:
                logger.warning(f"LLM failed to return a valid completion: {error_message}")
                return fallback_value
            completion_text = llm_response.completion_text.strip()
            if completion_text.startswith("```") and completion_text.endswith("```"):
                completion_text = completion_text.strip("`").strip()
            return completion_text
        except Exception as e:
            logger.error(f"An error occurred during LLM call: {e}", exc_info=True)
            return fallback_value

    async def _select_sampler_or_scheduler(self, event: AstrMessageEvent, resource_type: str, get_api_call: Callable[[Any], Coroutine], config_key_prefix: str):
        """Helper to select sampler or scheduler type (txt2img/img2img) and then the specific resource."""
        await event.send(event.plain_result(messages.MSG_INTERACTIVE_SELECT_TYPE.format(resource_type=resource_type)))

        @session_waiter(timeout=self.config.get("session_timeout", 30))
        async def type_waiter(controller: SessionController, type_event: AstrMessageEvent):
            choice = type_event.message_str.strip()
            if choice == '1':
                controller.stop()
                await self._interactive_selector(type_event, f"文生图{resource_type}", get_api_call(), [config_key_prefix, resource_type.lower().replace(' ', '_') + '_name'])
            elif choice == '2':
                controller.stop()
                await self._interactive_selector(type_event, f"图生图{resource_type}", get_api_call(), ["img2img_params", resource_type.lower().replace(' ', '_') + '_name'])
            elif choice.lower() in ["退出", "exit", "quit", "cancel"]:
                await type_event.send(type_event.plain_result(messages.MSG_OPERATION_CANCELLED))
                controller.stop()
            else:
                await type_event.send(type_event.plain_result(messages.MSG_INVALID_TYPE_CHOICE))
                controller.keep(timeout=30, reset_timeout=True)
        
        try:
            await type_waiter(event)
        except TimeoutError:
            await event.send(event.plain_result(messages.MSG_TIMEOUT_ERROR))
        except Exception as e:
            logger.error(f"Error in _select_sampler_or_scheduler for {resource_type}: {e}", exc_info=True)
            await event.send(event.plain_result(messages.MSG_ERROR_OCCURRED.format(e=e)))
        finally:
            event.stop_event()


    @filter.command("原生画", alias={"native"})
    async def handle_native(self, event: AstrMessageEvent):
        '''使用本地替换后的提示词直接进行文生图，不经过LLM'''
        if not await self._permission_check(event): return

        aliases = ["原生画", "native"]
        prompt_text = self._extract_prompt_from_message(event.message_str.strip(), aliases)

        if not prompt_text:
            yield event.plain_result(messages.MSG_NO_PROMPT_PROVIDED)
            return
            
        await event.send(event.plain_result(messages.MSG_GENERATING))
        
        replaced_prompt, _ = self.tag_manager.replace(prompt_text)
        async for result in self._generate_and_send(event, replaced_prompt, is_native=True):
            yield result

    @filter.command("i2i")
    async def handle_i2i(self, event: AstrMessageEvent, *, prompt_text: str):
        '''根据输入图片和文本提示词智能生成新图像'''
        if not await self._permission_check(event): return
        
        # 1. Extract image
        try:
            image_info = await self._extract_image_from_event(event)
        except ConnectionError:
            yield event.plain_result(messages.MSG_IMG_DOWNLOAD_FAILED)
            return
        if not image_info:
            yield event.plain_result(messages.MSG_NO_IMAGE_PROVIDED)
            return

        await event.send(event.plain_result(messages.MSG_IMG2IMG_GENERATING))

        # 2. Interrogate image for initial tags
        try:
            payload = {
                "image": image_info["b64"],
                "model": self.config.get("interrogator_model", "wd-eva02-large-tagger-v3"),
                "threshold": self.config.get("interrogator_threshold", 0.35)
            }
            interrogate_result = await self.api_client.interrogate(payload)
            
            if not (interrogate_result and "caption" in interrogate_result):
                logger.warning("Image interrogation failed to return a valid caption.")
                yield event.plain_result(messages.MSG_API_ERROR) # Generic error
                return

            sorted_tags = sorted(interrogate_result['caption'].items(), key=lambda item: item[1], reverse=True)
            initial_tags = ", ".join([tag for tag, _ in sorted_tags])
            
            if self.config.get("enable_show_positive_prompt", False):
                 logger.info(f"Interrogated tags: {initial_tags}")

        except ConnectionError:
            yield event.plain_result(messages.MSG_WEBUI_UNAVAILABLE)
            return
        except Exception as e:
            logger.error(f"An error occurred during i2i interrogation: {e}", exc_info=True)
            yield event.plain_result(messages.MSG_UNKNOWN_ERROR)
            return

        # 3. Use LLM to refine tags
        final_prompt = await self._get_llm_completion(
            event,
            prompt=f"User's modification request: '{prompt_text}'",
            contexts=[
                {"role": "system", "content": messages.SYSTEM_PROMPT_I2I_REFINEMENT},
                {"role": "user", "content": f"Original image tags: '{initial_tags}'"}
            ],
            error_message="LLM failed to return a valid refinement for i2i prompt.",
            fallback_value=initial_tags
        )

        # 4. Generate the final image
        final_prompt_replaced, _ = self.tag_manager.replace(final_prompt)
        
        async for result in self._generate_and_send(event, final_prompt_replaced, image_info=image_info, is_native=False):
            yield result

    async def _extract_image_from_event(self, event: AstrMessageEvent) -> Optional[Dict[str, Any]]:
        """Extracts the first base64 image and its dimensions from a message event."""
        if event.message_obj and event.message_obj.message:
            for comp in event.message_obj.message:
                if isinstance(comp, Comp.Image) and hasattr(comp, 'url') and comp.url:
                    image_bytes = await self.api_client.download_image_as_bytes(comp.url)
                    if image_bytes:
                        image_b64 = base64.b64encode(image_bytes).decode("utf-8")
                        with Image.open(io.BytesIO(image_bytes)) as img:
                            width, height = img.size
                        return {"b64": image_b64, "width": width, "height": height}
        return None

    @filter.command("inspire")
    async def handle_inspire(self, event: AstrMessageEvent, *, prompt_text: str):
        '''从图片中汲取灵感并结合文本提示词进行创作'''
        try:
            if not await self._permission_check(event): return
            try:
                image_info = await self._extract_image_from_event(event)
            except ConnectionError:
                yield event.plain_result(messages.MSG_IMG_DOWNLOAD_FAILED)
                return
            if not image_info:
                yield event.plain_result(messages.MSG_NO_IMAGE_PROVIDED)
                return
            
            await event.send(event.plain_result(messages.MSG_GENERATING))
            
            prompt_text, _ = self.tag_manager.replace(prompt_text)
            
            # This command now requires a vision-capable model.
            # We will use the low-level `text_chat` to include the image.
            
            curr_cid = await self.context.conversation_manager.get_curr_conversation_id(event.unified_msg_origin)
            conversation = None
            context = []
            if curr_cid:
                conversation = await self.context.conversation_manager.get_conversation(event.unified_msg_origin, curr_cid)
                if conversation:
                    context = json.loads(conversation.history)

            # The user's text instruction is the new prompt
            full_prompt = f"Based on the provided image, {prompt_text}"

            yield event.request_llm(
                prompt=full_prompt,
                session_id=curr_cid,
                contexts=context,
                system_prompt=messages.SYSTEM_PROMPT_SD,
                image_urls=[image_info["b64"]], # Pass image as base64
                conversation=conversation,
                func_tool_manager=self.context.get_llm_tool_manager()
            )
        except Exception as e:
            logger.error(f"An error occurred during inspire command: {e}", exc_info=True)
            yield event.plain_result(messages.MSG_UNKNOWN_ERROR)
        finally:
            event.stop_event()

    @filter.command("反推", alias={"interrogate"})
    async def handle_interrogate(self, event: AstrMessageEvent):
        '''使用Tagger API反推图片的提示词'''
        if not await self._permission_check(event): return
        try:
            image_info = await self._extract_image_from_event(event)
        except ConnectionError:
            yield event.plain_result(messages.MSG_IMG_DOWNLOAD_FAILED)
            return
        if not image_info:
            yield event.plain_result(messages.MSG_NO_IMAGE_PROVIDED)
            return

        await event.send(event.plain_result(messages.MSG_INTERROGATING_IMAGE))

        try:
            payload = {
                "image": image_info["b64"],
                "model": self.config.get("interrogator_model", "wd-eva02-large-tagger-v3"),
                "threshold": self.config.get("interrogator_threshold", 0.35)
            }
            result = await self.api_client.interrogate(payload)
            
            if result and "caption" in result:
                # Sort all tags by confidence
                sorted_tags = sorted(result['caption'].items(), key=lambda item: item[1], reverse=True)
                
                # Get all tag names from the sorted list, as the API already filtered by threshold
                all_tag_names = [tag for tag, _ in sorted_tags]
                tags_str = ", ".join(all_tag_names)

                # Format all confidences for the second line
                confidences_str = str([round(confidence, 2) for _, confidence in sorted_tags])

                yield event.plain_result(messages.MSG_INTERROGATION_RESULT.format(tags_str=tags_str, confidences_str=confidences_str))
            else:
                yield event.plain_result(messages.MSG_INTERROGATION_FAILED_NO_RESULT)
        except ConnectionError:
            yield event.plain_result(messages.MSG_WEBUI_UNAVAILABLE)
        except Exception as e:
            logger.error(f"An error occurred during interrogation: {e}", exc_info=True)
            yield event.plain_result(messages.MSG_INTERROGATION_UNKNOWN_ERROR.format(e=e))

    @filter.command("反推 set")
    async def handle_interrogator_threshold_set(self, event: AstrMessageEvent, value: Optional[float] = None):
        '''设置或查询反推时使用的置信度阈值'''
        async for r in self._set_config_value(event, "反推阈值", value, [["interrogator_threshold"]], validation=lambda v: 0.0 <= v <= 1.0): yield r

    @filter.command("反推 model")
    async def handle_interrogator_model_selection(self, event: AstrMessageEvent):
        '''交互式选择并设置反推模型'''
        await self._interactive_selector(
            event, 
            "反推模型", 
            self.api_client.get_interrogators, 
            ["interrogator_model"]
        )

    @sd_group.command("help")
    async def handle_help(self, event: AstrMessageEvent):
        '''显示所有可用指令的帮助信息'''
        core = [
            messages.HELP_CORE_NATIVE_DRAW,
            messages.HELP_CORE_I2I,
            messages.HELP_CORE_INSPIRE,
            messages.HELP_CORE_INTERROGATE,
            messages.HELP_CORE_CHECK,
            messages.HELP_CORE_CONF,
            messages.HELP_CORE_HELP,
        ]

        management = [
            messages.HELP_MANAGEMENT_SETTINGS,
            messages.HELP_MANAGEMENT_MODEL,
            messages.HELP_MANAGEMENT_SAMPLER,
            messages.HELP_MANAGEMENT_SCHEDULER,
            messages.HELP_MANAGEMENT_UPSCALER,
            messages.HELP_MANAGEMENT_INTERROGATE_MODEL,
            messages.HELP_MANAGEMENT_INTERROGATE_SET,
            messages.HELP_MANAGEMENT_LIST_LORAS,
            messages.HELP_MANAGEMENT_LIST_EMBEDDINGS,
            messages.HELP_MANAGEMENT_TAG,
        ]
        
        params = [
            messages.HELP_PARAMS_RES,
            messages.HELP_PARAMS_I2I_RES,
            messages.HELP_PARAMS_STEPS,
            messages.HELP_PARAMS_I2I_STEPS,
            messages.HELP_PARAMS_BATCH,
            messages.HELP_PARAMS_I2I_BATCH,
            messages.HELP_PARAMS_ITER,
            messages.HELP_PARAMS_I2I_ITER,
            messages.HELP_PARAMS_PREFIX,
            messages.HELP_PARAMS_I2I_PREFIX,
            messages.HELP_PARAMS_LLM_PREFIX,
            messages.HELP_PARAMS_I2I_DENOISE,
            messages.HELP_PARAMS_I2I_IMAGE_CFG_SCALE,
            messages.HELP_PARAMS_SEED,
            messages.HELP_PARAMS_TIMEOUT,
            messages.HELP_PARAMS_HR_SCALE,
            messages.HELP_PARAMS_HR_STEPS,
            messages.HELP_PARAMS_HR_DENOISE,
        ]

        all_commands = sorted(core + management + params)
        help_message = messages.HELP_TITLE + "\n".join(all_commands)
        
        node = Comp.Node(uin=event.message_obj.self_id, name=messages.HELP_NODE_NAME, content=[Comp.Plain(text=help_message)])
        yield event.chain_result([node])

    async def terminate(self):
        if self.api_client:
            await self.api_client.close()
        logger.info("SDGen_wzken plugin terminated.")
