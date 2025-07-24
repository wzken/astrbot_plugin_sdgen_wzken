# astrbot_plugin_sdgen_wzken/main.py

"""
Plugin Name: SDGen_wzken
Author: wzken
Version: 3.1.0
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
from astrbot.api.event import AstrMessageEvent, filter as astr_filter, MessageEventResult
from astrbot.api.star import Context, Star, register
from astrbot.core.utils.session_waiter import session_waiter, SessionController

from .core.client import SDAPIClient
from .core.generation import GenerationManager
from .utils.tag_manager import TagManager
from .utils.llm_helper import LLMHelper
from .utils.sd_utils import SDUtils
from .static import messages

@register("SDGen_wzken", "wzken", "A smarter and more powerful image generation plugin for AstrBot using Stable Diffusion.", "3.1.0")
class SDGeneratorWzken(Star):

    TOGGLES_MAP = [
        {"name": "详细输出模式", "keys": [["enable_show_positive_prompt"]]},
        {"name": "私聊发送结果", "keys": [["enable_private_message"]]},
        {"name": "合并转发消息", "keys": [["enable_forward_reply"]]},
        {"name": "图像增强 (超分)", "keys": [["enable_upscale"]]},
        {"name": "LLM生成提示词", "keys": [["enable_llm_prompt_generation"]]},
        {"name": "高分辨率修复", "keys": [["default_params", "enable_hr"]]},
        {"name": "面部修复", "keys": [["default_params", "restore_faces"], ["img2img_params", "restore_faces"]]},
        {"name": "无缝平铺", "keys": [["default_params", "tiling"], ["img2img_params", "tiling"]]},
        {"name": "图生图包含原图", "keys": [["img2img_params", "include_init_images"]]},
    ]

    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.config = config
        
        logger.info("SDGen_wzken plugin loaded. Initializing services...")
        self._initialize_services()
        logger.info("SDGen_wzken plugin initialization complete.")

    def _initialize_services(self):
        """Initializes all core services and managers."""
        plugin_dir = Path(__file__).parent.resolve()
        tags_dir = plugin_dir / "data"
        tags_dir.mkdir(exist_ok=True)

        self.api_client = SDAPIClient(self.config)
        self.sd_utils = SDUtils(self.config, self.context)
        self.generator = GenerationManager(self.config, self.api_client, self.sd_utils)
        
        tags_file = tags_dir / "tags.json"
        self.tag_manager = TagManager(str(tags_file))
        
        self.llm_helper = LLMHelper(self.context)

    # --- LLM Tool Definition ---
    @astr_filter.llm_tool("create_sd_image")
    async def generate_image(self, event: AstrMessageEvent, prompt) -> MessageEventResult:
        '''使用Stable Diffusion模型，根据用户的描述创作一幅全新的图像。
        这个工具的核心任务是将用户的自然语言描述（无论简单或复杂）转换成一个专业、详细、适合AI绘画的英文提示词，并用它来生成图像。
        当用户明确表示想要“画”、“绘制”、“创作”或“生成”图片时，应优先使用此工具。

        Args:
            prompt (string): 用户对所需图像的核心描述。根据输入类型，按以下方式处理：
                - **当只有文本时**: 将该描述作为核心生成依据（如“一个女孩”或“蔚蓝档案的小春在沙滩上”），自动补全合理细节并优化视觉表现。”。
                - **当同时有图片和文本时**: 将此文本视为用户的核心创作意图。必须分析图片内容，提取如背景、服装、角色细节、光照等视觉元素，并将这些信息与用户的文本意图结合，形成一个完整、详细的最终提示词。
        '''
        if not await self._permission_check(event):
            yield event.plain_result("Sorry, I don't have permission to draw in this chat.")
            return

        await event.send(event.plain_result(messages.MSG_GENERATING))
        
        replaced_prompt, replacements = self.tag_manager.replace(prompt)

        if replacements and self.config.get("enable_show_positive_prompt", False):
            replacement_str = "\n".join([f"- `{orig}` -> `{new}`" for orig, new in replacements])
            await event.send(event.plain_result(f"已应用本地关键词替换：\n{replacement_str}"))
        
        final_prompt = await self.llm_helper.generate_text_prompt(
            base_prompt=replaced_prompt,
            guidelines=self.config.get("prompt_guidelines", ""),
            prefix=self.config.get("llm_prompt_prefix", "")
        )
        
        async for result in self._generate_and_send(event, final_prompt):
            yield result

    # --- Generic Setting Handlers (Refactored) ---
    async def _handle_value_setting(
        self, event: AstrMessageEvent, config_paths: List[List[str]], name: str, 
        value_type: type, validation: Callable[[Any], bool] = None, 
        api_check: Coroutine = None, unit: str = ""
    ):
        value_str = event.message_str.strip()
        
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
    @astr_filter.command_group("sd")
    def sd_group(self):
        '''SD绘图插件指令组'''
        pass

    # --- Value Setting Commands ---
    @sd_group.command("timeout")
    async def handle_session_timeout(self, event: AstrMessageEvent):
        '''设置或查询会话超时时间（秒）'''
        async for result in self._handle_value_setting(event, [["session_timeout"]], "会话超时", int, validation=lambda v: v > 0, unit="秒"): yield result
    @sd_group.command("steps")
    async def handle_txt2img_steps(self, event: AstrMessageEvent):
        '''设置或查询文生图步数'''
        async for result in self._handle_value_setting(event, [["default_params", "steps"]], "文生图步数", int, validation=lambda v: v > 0): yield result
    @sd_group.command("batch")
    async def handle_txt2img_batch_size(self, event: AstrMessageEvent):
        '''设置或查询文生图批量生成数'''
        async for result in self._handle_value_setting(event, [["default_params", "batch_size"]], "文生图批量数", int, validation=lambda v: v > 0): yield result
    @sd_group.command("iter")
    async def handle_txt2img_n_iter(self, event: AstrMessageEvent):
        '''设置或查询文生图迭代次数'''
        async for result in self._handle_value_setting(event, [["default_params", "n_iter"]], "文生图迭代数", int, validation=lambda v: v > 0): yield result
    @sd_group.command("i2i steps")
    async def handle_i2i_steps(self, event: AstrMessageEvent):
        '''设置或查询图生图步数'''
        async for result in self._handle_value_setting(event, [["img2img_params", "steps"]], "图生图步数", int, validation=lambda v: v > 0): yield result
    @sd_group.command("i2i batch")
    async def handle_i2i_batch_size(self, event: AstrMessageEvent):
        '''设置或查询图生图批量生成数'''
        async for result in self._handle_value_setting(event, [["img2img_params", "batch_size"]], "图生图批量数", int, validation=lambda v: v > 0): yield result
    @sd_group.command("i2i iter")
    async def handle_i2i_n_iter(self, event: AstrMessageEvent):
        '''设置或查询图生图迭代次数'''
        async for result in self._handle_value_setting(event, [["img2img_params", "n_iter"]], "图生图迭代数", int, validation=lambda v: v > 0): yield result
    @sd_group.command("i2i denoise")
    async def handle_i2i_denoising_strength(self, event: AstrMessageEvent):
        '''设置或查询图生图重绘幅度'''
        async for result in self._handle_value_setting(event, [["img2img_params", "denoising_strength"]], "图生图重绘幅度", float, validation=lambda v: 0.0 <= v <= 1.0): yield result
    @sd_group.command("hr_scale")
    async def handle_hr_scale(self, event: AstrMessageEvent):
        '''设置或查询高分辨率修复的放大倍数'''
        async for result in self._handle_value_setting(event, [["default_params", "hr_scale"]], "高分修复放大倍数", float, validation=lambda v: v > 0): yield result
    @sd_group.command("hr_steps")
    async def handle_hr_steps(self, event: AstrMessageEvent):
        '''设置或查询高分辨率修复的第二阶段步数'''
        async for result in self._handle_value_setting(event, [["default_params", "hr_second_pass_steps"]], "高分修复二阶段步数", int, validation=lambda v: v >= 0): yield result
    @sd_group.command("seed")
    async def handle_seed(self, event: AstrMessageEvent):
        '''设置或查询生成种子，-1为随机'''
        keys = [["default_params", "seed"], ["img2img_params", "seed"]]
        async for result in self._handle_value_setting(event, keys, "生成种子", int): yield result
    @sd_group.command("i2i image_cfg_scale")
    async def handle_i2i_image_cfg_scale(self, event: AstrMessageEvent):
        '''设置或查询图生图的图像CFG'''
        async for result in self._handle_value_setting(event, [["img2img_params", "image_cfg_scale"]], "图生图图像CFG", float, validation=lambda v: v >= 0): yield result

    @sd_group.command("res")
    async def handle_txt2img_res(self, event: AstrMessageEvent):
        '''设置或查询文生图分辨率，格式：宽x高'''
        value_str = event.message_str.strip()
        if not value_str:
            w = self.config.get("default_params", {}).get("width", "N/A")
            h = self.config.get("default_params", {}).get("height", "N/A")
            yield event.plain_result(f"当前文生图分辨率为: `{w}x{h}`。")
            return
        
        try:
            width_str, height_str = value_str.lower().split('x')
            width, height = int(width_str), int(height_str)
            if not (width > 0 and height > 0 and width % 64 == 0 and height % 64 == 0):
                yield event.plain_result("❌ 无效的分辨率。宽和高都必须是大于0且为64的倍数。")
                return
            
            if "default_params" not in self.config: self.config["default_params"] = {}
            self.config["default_params"]["width"] = width
            self.config["default_params"]["height"] = height
            yield event.plain_result(f"文生图分辨率已设置为: `{width}x{height}`。")
        except (ValueError, IndexError):
            yield event.plain_result("❌ 格式错误。请输入 `宽度x高度`，例如 `512x768`。")

    @sd_group.command("i2i res")
    async def handle_i2i_res(self, event: AstrMessageEvent):
        '''设置或查询图生图分辨率，格式：宽x高'''
        value_str = event.message_str.strip()
        if not value_str:
            w = self.config.get("img2img_params", {}).get("width", "自动")
            h = self.config.get("img2img_params", {}).get("height", "自动")
            yield event.plain_result(f"当前图生图分辨率为: `{w}x{h}` (自动模式下此设置可能被覆盖)。")
            return
        
        try:
            width_str, height_str = value_str.lower().split('x')
            width, height = int(width_str), int(height_str)
            if not (width > 0 and height > 0 and width % 64 == 0 and height % 64 == 0):
                yield event.plain_result("❌ 无效的分辨率。宽和高都必须是大于0且为64的倍数。")
                return
            
            if "img2img_params" not in self.config: self.config["img2img_params"] = {}
            self.config["img2img_params"]["width"] = width
            self.config["img2img_params"]["height"] = height
            yield event.plain_result(f"图生图分辨率已设置为: `{width}x{height}`。")
        except (ValueError, IndexError):
            yield event.plain_result("❌ 格式错误。请输入 `宽度x高度`，例如 `512x768`。")

    # --- String/Complex Value Setting Commands ---
    @sd_group.command("prefix")
    async def handle_txt2img_prefix(self, event: AstrMessageEvent):
        '''设置或查询文生图的全局正向提示词前缀'''
        async for result in self._handle_value_setting(event, [["positive_prompt_global"]], "文生图正向提示词前缀", str): yield result
    @sd_group.command("i2i prefix")
    async def handle_i2i_prefix(self, event: AstrMessageEvent):
        '''设置或查询图生图的全局正向提示词前缀'''
        async for result in self._handle_value_setting(event, [["positive_prompt_i2i"]], "图生图正向提示词前缀", str): yield result
    @sd_group.command("llm prefix")
    async def handle_llm_prefix(self, event: AstrMessageEvent):
        '''设置或查询LLM生成提示词时的前缀'''
        async for result in self._handle_value_setting(event, [["llm_prompt_prefix"]], "LLM提示词前缀", str): yield result

    # --- Interactive/Session Commands ---
    async def _interactive_selector(
        self, event: AstrMessageEvent, resource_type: str, 
        fetch_coro: Coroutine, config_paths: List[List[str]], name_key: str = 'name'
    ):
        """A generic interactive selector for resources like models, samplers, etc."""
        try:
            items = await fetch_coro()
            if not items:
                await event.send(event.plain_result(f"无法获取{resource_type}列表。"))
                return

            item_names = [item[name_key] for item in items]
            
            current_values = []
            for path in config_paths:
                current_val_dict = self.config
                for key in path[:-1]:
                    current_val_dict = current_val_dict.get(key, {})
                current_values.append(current_val_dict.get(path[-1], 'N/A'))

            current_value_str = " / ".join([f"`{v}`" for v in current_values])
            prompt_lines = [f"当前{resource_type}: {current_value_str}\n", f"请选择新的{resource_type} (回复数字):"]
            prompt_lines.extend([f"{i+1}. {name}" for i, name in enumerate(item_names)])
            prompt_lines.append("\n(输入 \"退出\" 来取消)")
            
            await event.send(event.plain_result("\n".join(prompt_lines)))

            @session_waiter(timeout=60)
            async def selector_waiter(controller: SessionController, inner_event: AstrMessageEvent):
                choice = inner_event.message_str.strip()

                if choice.lower() in ["退出", "exit", "quit", "cancel"]:
                    await inner_event.send(inner_event.plain_result("操作已取消。"))
                    controller.stop()
                    return

                try:
                    choice_idx = int(choice) - 1
                    if not (0 <= choice_idx < len(item_names)):
                        raise ValueError("Index out of range.")
                    
                    chosen_item = item_names[choice_idx]
                    
                    for path in config_paths:
                        target_dict = self.config
                        for key in path[:-1]:
                            if key not in target_dict: target_dict[key] = {}
                            target_dict = target_dict[key]
                        target_dict[path[-1]] = chosen_item

                    await inner_event.send(inner_event.plain_result(f"✅ {resource_type}已设置为: `{chosen_item}`"))
                    controller.stop()

                except ValueError:
                    await inner_event.send(inner_event.plain_result("❌ 无效的选择，请输入列表中的数字或'退出'。"))
                    controller.keep(timeout=60, reset_timeout=True)
                except Exception as e:
                    logger.error(f"Error setting {resource_type} in session: {e}")
                    await inner_event.send(inner_event.plain_result(f"设置{resource_type}时发生错误: {e}"))
                    controller.stop()

            await selector_waiter(event)

        except ConnectionError:
            await event.send(event.plain_result(messages.MSG_WEBUI_UNAVAILABLE))
        except TimeoutError:
            await event.send(event.plain_result("⌛ 操作超时，已取消。"))
        except Exception as e:
            logger.error(f"Interactive selector for {resource_type} error: {e}", exc_info=True)
            await event.send(event.plain_result(f"发生错误: {e}"))

    @sd_group.command("model")
    async def handle_model_selection(self, event: AstrMessageEvent):
        '''交互式选择并设置基础模型'''
        await self._interactive_selector(event, "模型", self.api_client.get_sd_models, [["base_model"]], name_key='title')
        event.stop_event()

    @sd_group.command("sampler")
    async def handle_sampler_selection(self, event: AstrMessageEvent):
        '''交互式选择并设置采样器'''
        await event.send(event.plain_result("请选择要设置的采样器类型:\n1. 文生图 (txt2img)\n2. 图生图 (img2img)\n\n(回复数字, 或'退出')"))

        @session_waiter(timeout=30)
        async def type_waiter(controller: SessionController, type_event: AstrMessageEvent):
            choice = type_event.message_str.strip()
            if choice == '1':
                controller.stop()
                await self._interactive_selector(type_event, "文生图采样器", self.api_client.get_samplers, [["default_params", "sampler"]])
            elif choice == '2':
                controller.stop()
                await self._interactive_selector(type_event, "图生图采样器", self.api_client.get_samplers, [["img2img_params", "sampler"]])
            elif choice.lower() in ["退出", "exit", "quit", "cancel"]:
                await type_event.send(type_event.plain_result("操作已取消。"))
                controller.stop()
            else:
                await type_event.send(type_event.plain_result("无效选择，请输入 1 或 2。"))
                controller.keep(timeout=30, reset_timeout=True)
        
        try:
            await type_waiter(event)
        except TimeoutError:
            await event.send(event.plain_result("⌛ 操作超时，已取消。"))
        except Exception as e:
            logger.error(f"handle_sampler_selection error: {e}", exc_info=True)
            await event.send(event.plain_result(f"发生错误: {e}"))
        finally:
            event.stop_event()

    @sd_group.command("scheduler")
    async def handle_scheduler_selection(self, event: AstrMessageEvent):
        '''交互式选择并设置调度器'''
        await event.send(event.plain_result("请选择要设置的调度器类型:\n1. 文生图 (txt2img)\n2. 图生图 (img2img)\n\n(回复数字, 或'退出')"))

        @session_waiter(timeout=30)
        async def type_waiter(controller: SessionController, type_event: AstrMessageEvent):
            choice = type_event.message_str.strip()
            if choice == '1':
                controller.stop()
                await self._interactive_selector(type_event, "文生图调度器", self.api_client.get_schedulers, [["default_params", "scheduler"]])
            elif choice == '2':
                controller.stop()
                await self._interactive_selector(type_event, "图生图调度器", self.api_client.get_schedulers, [["img2img_params", "scheduler"]])
            elif choice.lower() in ["退出", "exit", "quit", "cancel"]:
                await type_event.send(type_event.plain_result("操作已取消。"))
                controller.stop()
            else:
                await type_event.send(type_event.plain_result("无效选择，请输入 1 或 2。"))
                controller.keep(timeout=30, reset_timeout=True)
        
        try:
            await type_waiter(event)
        except TimeoutError:
            await event.send(event.plain_result("⌛ 操作超时，已取消。"))
        except Exception as e:
            logger.error(f"handle_scheduler_selection error: {e}", exc_info=True)
            await event.send(event.plain_result(f"发生错误: {e}"))
        finally:
            event.stop_event()

    @sd_group.command("upscaler")
    async def handle_upscaler_selection(self, event: AstrMessageEvent):
        '''交互式选择并设置上采样（放大）算法'''
        config_paths = [["upscale_params", "upscaler"], ["default_params", "hr_upscaler"]]
        await self._interactive_selector(event, "上采样算法", self.api_client.get_upscalers, config_paths)
        event.stop_event()

    @sd_group.command("设置")
    async def handle_settings(self, event: AstrMessageEvent):
        '''交互式开启或关闭各项功能'''
        
        def get_toggle_status_text():
            lines = ["请选择要切换的设置 (回复数字):"]
            for i, toggle in enumerate(self.TOGGLES_MAP):
                key_path = toggle["keys"][0]
                current_status = self.config.get(key_path[0], {}).get(key_path[1], False) if len(key_path) > 1 else self.config.get(key_path[0], False)
                status_str = "✅ 开" if current_status else "❌ 关"
                lines.append(f"{i+1}. {toggle['name']} (当前: {status_str})")
            lines.append("\n(输入 \"退出\" 来取消)")
            return "\n".join(lines)

        await event.send(event.plain_result(get_toggle_status_text()))

        @session_waiter(timeout=60)
        async def settings_waiter(controller: SessionController, inner_event: AstrMessageEvent):
            choice = inner_event.message_str.strip()

            if choice.lower() in ["退出", "exit", "quit", "cancel"]:
                await inner_event.send(inner_event.plain_result("已退出设置。"))
                controller.stop()
                return

            try:
                choice_idx = int(choice) - 1
                if not (0 <= choice_idx < len(self.TOGGLES_MAP)):
                    raise ValueError("Index out of range.")
                
                toggle_to_change = self.TOGGLES_MAP[choice_idx]
                
                key_path = toggle_to_change["keys"][0]
                current_status = self.config.get(key_path[0], {}).get(key_path[1], False) if len(key_path) > 1 else self.config.get(key_path[0], False)
                new_status = not current_status

                for path in toggle_to_change["keys"]:
                    target_dict = self.config
                    for key in path[:-1]:
                        if key not in target_dict: target_dict[key] = {}
                        target_dict = target_dict[key]
                    target_dict[path[-1]] = new_status
                
                status_feedback = "开启" if new_status else "关闭"
                await inner_event.send(inner_event.plain_result(f"✅ {toggle_to_change['name']} 已{status_feedback}。"))
                
                await inner_event.send(inner_event.plain_result(get_toggle_status_text()))
                controller.keep(timeout=60, reset_timeout=True)

            except ValueError:
                await inner_event.send(inner_event.plain_result("❌ 无效的选择，请输入列表中的数字或'退出'。"))
                controller.keep(timeout=60, reset_timeout=True)
            except Exception as e:
                logger.error(f"Error in settings session: {e}")
                await inner_event.send(inner_event.plain_result(f"处理设置时发生错误: {e}"))
                controller.stop()

        try:
            await settings_waiter(event)
        except TimeoutError:
            await event.send(event.plain_result("⌛ 操作超时，已取消。"))
        except Exception as e:
            logger.error(f"handle_settings error: {e}", exc_info=True)
            await event.send(event.plain_result(f"发生错误: {e}"))
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
            conf_str = f"---文生图参数---\n{t2i_params}\n\n---图生图参数---\n{i2i_params}"
            yield event.plain_result(conf_str)
        except Exception as e:
            logger.error(f"Failed to display configuration: {e}")
            yield event.plain_result("获取配置失败。")

    # --- List Commands ---
    @sd_group.command("list_loras")
    async def handle_list_loras(self, event: AstrMessageEvent):
        '''列出所有可用的LoRA模型'''
        async for result in self._list_api_resource(event, "LoRA模型", self.api_client.get_loras): yield result
    @sd_group.command("list_embeddings")
    async def handle_list_embeddings(self, event: AstrMessageEvent):
        '''列出所有可用的Embedding模型'''
        async for result in self._list_api_resource(event, "Embedding", self.api_client.get_embeddings): yield result

    async def _list_api_resource(self, event: AstrMessageEvent, resource_name: str, api_call: Coroutine, name_key: str = 'name', title_key: str = 'title'):
        try:
            items = await api_call()
            if not items:
                yield event.plain_result(f"未找到任何可用的{resource_name}。")
                return
            
            if isinstance(items, dict) and 'loaded' in items:
                item_names = "\n".join([f"- `{e}`" for e in items['loaded'].keys()])
            else:
                item_names = "\n".join([f"- `{item.get(title_key, item.get(name_key))}`" for item in items])
            
            yield event.plain_result(f"可用的{resource_name}：\n{item_names}")
        except ConnectionError:
            yield event.plain_result(messages.MSG_WEBUI_UNAVAILABLE)
        except Exception as e:
            logger.error(f"Failed to list {resource_name}: {e}")
            yield event.plain_result(f"获取{resource_name}列表失败。")

    # --- Tag Management ---
    @sd_group.command("tag")
    async def handle_tag(self, event: AstrMessageEvent):
        '''管理本地关键词替换。用法: /sd tag <list|add|del|import|关键词:内容>'''
        text = event.message_str.strip()
        if text == "list":
            all_tags = self.tag_manager.get_all()
            if not all_tags:
                yield event.plain_result("当前没有本地标签。")
                return
            tag_list_str = "\n".join([f"- `{k}`: `{v}`" for k, v in all_tags.items()])
            yield event.plain_result(f"已保存的本地标签：\n{tag_list_str}")
            return
        if text.startswith("del "):
            key_to_del = text[4:].strip()
            if self.tag_manager.del_tag(key_to_del): yield event.plain_result(messages.MSG_TAG_DELETED.format(key=key_to_del))
            else: yield event.plain_result(f"❌ 未找到标签 '{key_to_del}'。")
            return
        if text.startswith("add "):
            parts = text[4:].strip().split(maxsplit=1)
            if len(parts) == 2:
                self.tag_manager.set_tag(parts[0], parts[1])
                yield event.plain_result(messages.MSG_TAG_SET.format(key=parts[0]))
            else: yield event.plain_result("❌ 格式错误，应为 `/sd tag add <名> <提示词>`。")
            return
        if text.startswith("import "):
            json_str = text[7:].strip()
            try:
                new_tags = json.loads(json_str)
                if not isinstance(new_tags, dict): raise json.JSONDecodeError("Not a JSON object.", json_str, 0)
                self.tag_manager.import_tags(new_tags)
                yield event.plain_result(messages.MSG_TAGS_IMPORTED)
            except json.JSONDecodeError: yield event.plain_result("❌ 导入失败：非有效JSON。")
            return
        if ":" in text:
            key, value = text.split(":", 1)
            key, value = key.strip(), value.strip()
            if key and value:
                self.tag_manager.set_tag(key, value)
                yield event.plain_result(messages.MSG_TAG_SET.format(key=key))
            else: yield event.plain_result("❌ 格式错误，应为 `/sd tag 关键词:内容`。")
            return
        yield event.plain_result("用法: `/sd tag <list|add|del|rename|search|import|关键词:内容>`")

    @sd_group.command("tag rename")
    async def handle_tag_rename(self, event: AstrMessageEvent):
        '''重命名一个本地关键词'''
        parts = event.message_str.strip().split(maxsplit=1)
        if len(parts) == 2:
            old_name, new_name = parts
            if self.tag_manager.rename_tag(old_name, new_name): yield event.plain_result(f"标签 `{old_name}` 已重命名为 `{new_name}`。")
            else: yield event.plain_result(f"❌ 重命名失败: 未找到标签 `{old_name}`。")
        else: yield event.plain_result("❌ 格式错误，应为 `/sd tag rename <旧名> <新名>`。")

    @sd_group.command("tag search")
    async def handle_tag_search(self, event: AstrMessageEvent):
        '''模糊搜索本地关键词'''
        keyword = event.message_str.strip()
        if not keyword:
            yield event.plain_result("❌ 请提供搜索关键词。")
            return
        found_tags = self.tag_manager.fuzzy_search(keyword)
        if not found_tags:
            yield event.plain_result(f"未找到含 '{keyword}' 的标签。")
            return
        tag_list_str = "\n".join([f"- `{k}`: `{v}`" for k, v in found_tags.items()])
        yield event.plain_result(f"找到的标签：\n{tag_list_str}")

    # --- Core Generation Logic ---
    async def _permission_check(self, event: AstrMessageEvent) -> bool:
        group_id = event.get_group_id()
        if group_id in self.config.get("blacklist_groups", []):
            logger.info(f"Command ignored in blacklisted group: {group_id}")
            return False
        return True

    async def _generate_and_send(self, event: AstrMessageEvent, final_prompt: str, image_info: dict = None, is_inspire: bool = False):
        try:
            group_id = event.get_group_id()
            user_id = event.get_sender_id()
            is_i2i = image_info is not None

            if is_i2i:
                positive_prefix = self.config.get("positive_prompt_i2i", "")
            else:
                positive_prefix = self.config.get("positive_prompt_global", "")
            
            whitelist_groups = self.config.get("whitelist_groups", [])
            if group_id in whitelist_groups:
                positive_prefix = self.config.get("positive_prompt_whitelist", "masterpiece, best quality")
            
            negative_prefix = self.config.get("negative_prompt_global", "(worst quality, low quality:1.4)")
            full_positive_prompt = ", ".join(filter(None, [positive_prefix, final_prompt]))
            full_negative_prompt = negative_prefix

            if is_inspire:
                generated_images = await self.generator.generate_txt2img(full_positive_prompt, full_negative_prompt)
            elif is_i2i:
                generated_images = await self.generator.generate_img2img(image_info, full_positive_prompt, full_negative_prompt)
            else:
                generated_images = await self.generator.generate_txt2img(full_positive_prompt, full_negative_prompt)

            if not generated_images:
                yield event.plain_result(messages.MSG_API_ERROR)
                return

            processed_images = await self.generator.process_and_upscale_images(generated_images)
            image_components = [Comp.Image.fromBase64(img) for img in processed_images]
            
            if self.config.get("enable_show_positive_prompt", True):
                image_components.insert(0, Comp.Plain(f"{messages.MSG_PROMPT_DISPLAY}: {full_positive_prompt}"))

            send_private = self.config.get("enable_private_message", False)
            use_forward_reply = self.config.get("enable_forward_reply", False)

            if send_private:
                yield event.send_result(image_components, user_id=user_id)
            elif use_forward_reply:
                node = Comp.Node(uin=event.message_obj.self_id, name="SD 生成结果", content=image_components)
                yield event.chain_result([node])
            else:
                yield event.chain_result(image_components)

        except Exception as e:
            logger.error(f"An error occurred during image generation/sending: {e}", exc_info=True)
            yield event.plain_result(messages.MSG_UNKNOWN_ERROR)

    @astr_filter.command("原生画", alias={"native"})
    async def handle_native(self, event: AstrMessageEvent):
        '''使用本地替换后的提示词直接进行文生图，不经过LLM'''
        if not await self._permission_check(event): return
        prompt_text = event.message_str.strip()
        if not prompt_text:
            yield event.plain_result(messages.MSG_NO_PROMPT_PROVIDED)
            return
        await event.send(event.plain_result(messages.MSG_GENERATING))
        final_prompt = prompt_text
        async for result in self._generate_and_send(event, final_prompt):
            yield result

    @astr_filter.command("i2i")
    async def handle_i2i(self, event: AstrMessageEvent):
        '''根据输入图片和文本提示词生成新图像'''
        if not await self._permission_check(event): return
        try:
            image_info = await self._extract_image_and_text(event)
        except ConnectionError:
            yield event.plain_result(messages.MSG_IMG_DOWNLOAD_FAILED)
            return
        if not image_info or not image_info.get("b64"):
            yield event.plain_result(messages.MSG_NO_IMAGE_PROVIDED)
            return
        
        await event.send(event.plain_result(messages.MSG_IMG2IMG_GENERATING))
        
        prompt_text = image_info.get("prompt", "")
        replaced_prompt, _ = self.tag_manager.replace(prompt_text)
        
        if self.config.get("enable_llm_prompt_generation", True):
            final_prompt = await self.llm_helper.generate_text_prompt(
                base_prompt=replaced_prompt,
                guidelines=self.config.get("prompt_guidelines", ""),
                prefix=self.config.get("llm_prompt_prefix", "")
            )
        else:
            final_prompt = replaced_prompt
            
        async for result in self._generate_and_send(event, final_prompt, image_info=image_info):
            yield result

    async def _extract_image_and_text(self, event: AstrMessageEvent) -> Optional[Dict[str, Any]]:
        """Extracts base64 image, its dimensions, and text from a message event."""
        image_b64, prompt_text = None, ""
        width, height = 0, 0

        if event.message_obj and event.message_obj.message:
            for comp in event.message_obj.message:
                if isinstance(comp, Comp.Image) and hasattr(comp, 'url') and comp.url:
                    image_bytes = await self.api_client.download_image_as_bytes(comp.url)
                    if image_bytes:
                        image_b64 = base64.b64encode(image_bytes).decode("utf-8")
                        with Image.open(io.BytesIO(image_bytes)) as img:
                            width, height = img.size
                elif isinstance(comp, Comp.Plain):
                    prompt_text += comp.text + " "
        
        if image_b64:
            return {"b64": image_b64, "width": width, "height": height, "prompt": prompt_text.strip()}
        return None

    @astr_filter.command("inspire")
    async def handle_inspire(self, event: AstrMessageEvent):
        '''从图片中汲取灵感并结合文本提示词进行创作'''
        if not await self._permission_check(event): return
        try:
            image_info = await self._extract_image_and_text(event)
        except ConnectionError:
            yield event.plain_result(messages.MSG_IMG_DOWNLOAD_FAILED)
            return
        if not image_info or not image_info.get("b64"):
            yield event.plain_result(messages.MSG_NO_IMAGE_PROVIDED)
            return
        
        await event.send(event.plain_result(messages.MSG_GENERATING))
        prompt_text, _ = self.tag_manager.replace(image_info.get("prompt", ""))
        
        final_prompt = await self.llm_helper.generate_prompt_from_image(
            image_b64=image_info["b64"], user_instruction=prompt_text,
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
        '''显示所有可用指令的帮助信息'''
        core = [
            "- `/原生画 <提示词>`: 直接绘图，不经优化。",
            "- `/i2i <提示词>`: 图生图。",
            "- `/inspire <提示词>`: 从图片中汲取灵感创作。",
            "- `/sd check`: 检查服务是否可用。",
            "- `/sd conf`: 显示当前配置。",
            "- `/sd help`: 显示此帮助。",
        ]

        management = [
            "- `/sd 设置`: 交互式开启/关闭各项功能。",
            "- `/sd model`: 交互式选择模型。",
            "- `/sd sampler`: 交互式选择采样器。",
            "- `/sd scheduler`: 交互式选择调度器。",
            "- `/sd upscaler`: 交互式选择放大器。",
            "- `/sd list_loras`: 列出LoRA模型。",
            "- `/sd list_embeddings`: 列出Embedding。",
            "- `/sd tag`: 管理本地关键词。",
        ]
        
        params = [
            "- `/sd res <宽x高>`: 设置文生图分辨率。",
            "- `/sd i2i res <宽x高>`: 设置图生图分辨率。",
            "- `/sd steps [步数]`: 设置文生图步数。",
            "- `/sd i2i steps [步数]`: 设置图生图步数。",
            "- `/sd batch [数量]`: 设置文生图批量数。",
            "- `/sd i2i batch [数量]`: 设置图生图批量数。",
            "- `/sd iter [次数]`: 设置文生图迭代数。",
            "- `/sd i2i iter [次数]`: 设置图生图迭代数。",
            "- `/sd prefix [前缀]`: 设置文生图正向提示词前缀。",
            "- `/sd i2i prefix [前缀]`: 设置图生图正向提示词前缀。",
            "- `/sd llm prefix [前缀]`: 设置LLM提示词前缀。",
            "- `/sd i2i denoise [幅度]`: 设置图生图重绘幅度 (0-1)。",
            "- `/sd i2i image_cfg_scale [值]`: 设置图生图图像CFG。",
            "- `/sd seed [种子]`: 设置生成种子 (-1为随机)。",
            "- `/sd timeout [秒数]`: 设置会话超时。",
            "- `/sd hr_scale [倍数]`: 设置高分修复放大倍数。",
            "- `/sd hr_steps [步数]`: 设置高分修复二阶段步数。",
        ]

        all_commands = sorted(core + management + params)
        help_message = "SD绘图插件可用指令：\n\n" + "\n".join(all_commands)
        
        node = Comp.Node(uin=event.message_obj.self_id, name="SD 绘图插件帮助手册", content=[Comp.Plain(text=help_message)])
        yield event.chain_result([node])

    async def terminate(self):
        if self.api_client:
            await self.api_client.close()
        logger.info("SDGen_wzken plugin terminated.")
