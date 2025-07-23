# astrbot_plugin_sdgen_v2/static/messages.py

# --- 生成状态 ---
MSG_GENERATING = "🎨 正在全力绘图中..."
MSG_IMG2IMG_GENERATING = "🎨 正在处理图片并重绘..."
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

# --- 成功与信息 ---
MSG_WEBUI_AVAILABLE = "✅ SD WebUI 连接正常。"
MSG_PROMPT_DISPLAY = "📝 正向提示词"
MSG_CONFIG_UPDATED = "✅ 配置已成功更新。"
MSG_TAG_SET = "✅ 成功设置标签：'{key}'。"
MSG_TAG_DELETED = "✅ 成功删除标签：'{key}'。"
MSG_TAGS_IMPORTED = "✅ 标签导入成功。"

# --- 帮助与使用 ---
MSG_HELP_MAIN_TITLE = "🖼️ SDGen v2 帮助指南"
MSG_HELP_NATIVE_CMD = "- `/原生画 [提示词]`：根据您的专业提示词直接生成图片。"
MSG_HELP_I2I_CMD = "- `/i2i [图片] [提示词]`：在您提供的图片基础上进行修改和重绘。"
MSG_HELP_INSPIRE_CMD = "- `/inspire [图片] [提示词]`：从您提供的图片中汲取灵感，创作一幅全新的画作。"
# (更多帮助信息待添加)

# --- 交互式提示 ---
MSG_CONFIRM_PROMPT = "AI为您生成了以下提示词：\n`{prompt}`\n请回复“确认”开始绘图，或直接发送新提示词进行修改。"
MSG_HISTORY_PROMPT = "您最近的生成历史：\n{history_list}\n请回复“复用 [编号]”来使用该条记录。"
