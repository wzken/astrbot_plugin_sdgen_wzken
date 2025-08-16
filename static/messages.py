# astrbot_plugin_sdgen_v2/static/messages.py

# --- System Prompts ---
SYSTEM_PROMPT_SD = """You are a Danbooru prompt expert for Stable Diffusion. Convert the user's request into a high-quality, comma-separated string of tags.

**CRITICAL**: For **character and artist tags ONLY**, you must remove underscores (`_`) and escape parentheses (`()`).
Example: `hatsune_miku` -> `hatsune miku`, `lucy_(cyberpunk)` -> `lucy \(cyberpunk\)`.
All other tags (quality, content like `blue_sky`) must remain unchanged.
"""

SYSTEM_PROMPT_I2I_REFINEMENT = """You are a Danbooru prompt refiner. Merge the user's request with the original tags to create an updated, comma-separated prompt.

**Rules:**
1.  **User Request is Priority**: It overrides any conflicting original tags.
2.  **Preserve Non-Conflicting Tags**: Keep original tags for background, style, etc.
3.  **Format Output**:
    - For **character/artist tags ONLY**: remove underscores (`_`) and escape parentheses (`()`). Example: `hatsune_miku` -> `hatsune miku`, `lucy_(cyberpunk)` -> `lucy \(cyberpunk\)`.
    - Keep all other tags unchanged.
    - Output only the final comma-separated string.
"""

# --- 生成状态 ---
MSG_GENERATING = "🎨 正在全力绘图中..."
MSG_IMG2IMG_GENERATING = "🎨 正在处理图片并重绘..."
MSG_INTERROGATING_IMAGE = "🔍 正在反推图片提示词..."
MSG_INTERROGATION_RESULT = "反推结果:\n- 标签: {tags_str}\n- 置信度: {confidences_str}"
MSG_UPSCALE_PROCESSING = "✨ 正在提升图像质量..."
MSG_VISION_FALLBACK = "⚠️ 图像理解服务暂不可用，已降级为纯文本生成模式。"

# --- 错误信息 ---
MSG_WEBUI_UNAVAILABLE = "❌ SD WebUI 连接失败，无法生成图片。"
MSG_API_ERROR = "❌ 图像生成失败：API 调用出错。"
MSG_CONNECTION_ERROR = "❌ 连接失败，请检查网络连接和 WebUI 服务状态。"
MSG_TIMEOUT_ERROR = "❌ 请求超时，请稍后再试。"
MSG_UNKNOWN_ERROR = "❌ 发生未知错误，图像生成失败。"
MSG_IMG_DOWNLOAD_FAILED = "❌ 图片下载失败，请检查链接是否有效。"
MSG_NO_IMAGE_PROVIDED = "⚠️ 此命令需要您提供一张图片。"
MSG_NO_PROMPT_PROVIDED = "⚠️ 请输入有效的提示词。"
MSG_PERMISSION_DENIED = "Sorry, I don't have permission to draw in this chat."
MSG_INVALID_VALUE = "❌ 无效的{name}值。"
MSG_SET_FAILED = "设置{name}失败。"
MSG_INVALID_RESOLUTION = "❌ 无效的分辨率。宽和高都必须是大于0且为64的倍数。"
MSG_RESOLUTION_FORMAT_ERROR = "❌ 格式错误。请输入 `宽度x高度`，例如 `512x768`。"
MSG_FETCH_LIST_FAILED = "无法获取{resource_type}列表。"
MSG_OPERATION_CANCELLED = "操作已取消。"
MSG_INVALID_CHOICE = "❌ 无效的选择，请输入列表中的数字或'退出'。"
MSG_ERROR_SETTING_RESOURCE = "设置{resource_type}时发生错误: {e}"
MSG_ERROR_OCCURRED = "发生错误: {e}"
MSG_GET_CONFIG_FAILED = "获取配置失败。"
MSG_RESOURCE_NOT_FOUND = "未找到任何可用的{resource_name}。"
MSG_TAG_NOT_FOUND = "❌ 未找到标签 '{key}'。"
MSG_TAG_ADD_FORMAT_ERROR = "❌ 格式错误，应为 `/sd tag add <名> <提示词>`。"
MSG_IMPORT_FAILED_INVALID_JSON = "❌ 导入失败：非有效JSON。"
MSG_TAG_SET_FORMAT_ERROR = "❌ 格式错误，应为 `/sd tag 关键词:内容`。"
MSG_TAG_SEARCH_NO_KEYWORD = "❌ 请提供搜索关键词。"
MSG_TAG_RENAME_FAILED_NOT_FOUND = "❌ 重命名失败: 未找到标签 `{old_name}`。"
MSG_INTERROGATION_FAILED_NO_RESULT = "反推失败，API未返回有效结果。"
MSG_INTERROGATION_UNKNOWN_ERROR = "反推时发生未知错误: {e}"
MSG_INTERACTIVE_SELECT_TYPE = "请选择要设置的{resource_type}类型:\n1. 文生图 (txt2img)\n2. 图生图 (img2img)\n\n(回复数字, 或'退出')"
MSG_INVALID_TYPE_CHOICE = "无效选择，请输入 1 或 2。"
MSG_EXIT_SETTINGS = "已退出设置。"
MSG_TAG_HELP_USAGE = "用法: `/sd tag <list|add|del|rename|search|import|关键词:内容>`"


# --- 成功与信息 ---
MSG_WEBUI_AVAILABLE = "✅ SD WebUI 连接正常。"
MSG_PROMPT_DISPLAY = "📝 正向提示词"
MSG_CONFIG_UPDATED = "✅ 配置已成功更新。"
MSG_TAG_SET = "✅ 成功设置标签：'{key}'。"
MSG_TAG_DELETED = "✅ 成功删除标签：'{key}'。"
MSG_TAGS_IMPORTED = "✅ 标签导入成功。"
MSG_CURRENT_VALUE = "当前{name}为：`{current_val}`{unit}。"
MSG_VALUE_SET = "✅ {name}已设置为：`{value}`{unit}。"
MSG_RESOLUTION_SET = "✅ {name}分辨率已设置为: `{width}x{height}`。"
MSG_CURRENT_RESOLUTION = "当前{name}分辨率为: `{width}x{height}`{auto_note}。"
MSG_RESOURCE_SET = "✅ {resource_type}已设置为: `{chosen_item}`"
MSG_MODEL_SWITCHING = "正在切换模型至 `{model_title}`，请稍候..."
MSG_UPSCALER_MENU = "当前放大模式: {mode_text} (总开关: {enabled_text})\n\n1. 切换放大模式\n2. 选择放大算法\n3. 开启/关闭图像放大\n\n(回复数字或'退出')"
MSG_MODE_SWITCHED = "✅ 模式已切换。\n\n{menu_text}"
MSG_TOGGLE_STATUS_SET = "✅ 总开关已设置。\n\n{menu_text}"
MSG_SETTINGS_SELECT_TOGGLE = "请选择要切换的设置 (回复数字):"
MSG_SETTINGS_EXIT_PROMPT = "\n(输入 \"退出\" 来取消)"
MSG_TOGGLE_FEEDBACK = "✅ {toggle_name} 已{status_feedback}。"
MSG_AVAILABLE_RESOURCE = "可用的{resource_name}：\n{item_names}"
MSG_TAGS_SAVED = "已保存的本地标签：\n{tag_list_str}"
MSG_TAG_RENAMED = "标签 `{old_name}` 已重命名为 `{new_name}`。"
MSG_TAGS_FOUND = "找到的标签：\n{tag_list_str}"


# --- 帮助与使用 ---
MSG_HELP_MAIN_TITLE = "🖼️ SDGen v2 帮助指南"
MSG_HELP_NATIVE_CMD = "- `/原生画 [提示词]`：根据您的专业提示词直接生成图片。"
MSG_HELP_I2I_CMD = "- `/i2i [图片] [提示词]`：在您提供的图片基础上进行修改和重绘。"
MSG_HELP_INSPIRE_CMD = "- `/inspire [图片] [提示词]`：从您提供的图片中汲取灵感，创作一幅全新的画作。"
# (更多帮助信息待添加)

# --- 交互式提示 ---
MSG_CURRENT_RESOURCE_PROMPT = "当前{resource_type}为: {current_val}"
MSG_SELECT_NEW_RESOURCE_PROMPT = "请选择新的{resource_type} (回复数字):"
MSG_EXIT_PROMPT = "\n(输入 \"退出\" 来取消)"
MSG_CONFIRM_PROMPT = "AI为您生成了以下提示词：\n`{prompt}`\n请回复“确认”开始绘图，或直接发送新提示词进行修改。"
MSG_HISTORY_PROMPT = "您最近的生成历史：\n{history_list}\n请回复“复用 [编号]”来使用该条记录。"
