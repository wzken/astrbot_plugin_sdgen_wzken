# AstrBot SDGen Plugin by wzken

这是一个为 [AstrBot](https://github.com/Soulter/AstrBot) 设计的强大且智能的 Stable Diffusion 图像生成插件。它通过与 Stable Diffusion WebUI (如 a1111/Forge/ComfyUI) 的 API 对接，提供了丰富的图像生成和参数调整功能。

## ✨ 功能特性

- **多种生成模式**:
  - **文生图 (`/inspire`)**: 结合 LLM（视觉模型）理解图片内容，并根据您的文本描述生成新的、富有创意的图像。
  - **图生图 (`/i2i`)**: 以您提供的图片为基础，结合文本提示词进行二次创作。
  - **原生画 (`/原生画`)**: 直接使用您提供的提示词进行绘图，不经过 LLM 处理，适合专业用户。
- **丰富的参数控制**:
  - **高分辨率修复 (Hires. fix)**: 一键开启高分辨率修复，并可调整放大倍数、放大器和步数。
  - **通用参数**: 控制种子 (`seed`)、面部修复 (`restore_faces`) 和无缝平铺 (`tiling`)。
  - **精细化调整**: 分别设置文生图和图生图的分辨率、步数、批次大小、迭代次数、CFG Scale 等。
- **模型与资源管理**:
  - 动态列出并切换 Stable Diffusion 基础模型、LoRA 模型、采样器、调度器和放大器。
- **智能提示词工程**:
  - **LLM 辅助**: 自动使用大型语言模型优化和丰富您的基本提示词。
  - **本地关键词**: 支持添加、删除、查询和导入本地关键词替换规则，简化常用提示词。
- **灵活的发送方式**:
  - **私聊发送**: 可选将生成的图片直接私聊发送给用户，避免在群聊中刷屏。 (`/sd private`)
  - **转发回复**: 在群聊中，可选择以“合并转发”的形式进行回复，保持聊天窗口的整洁。 (`/sd forward_reply`)
- **用户友好的交互**:
  - **帮助信息**: 通过 `/sd help` 以合并转发的形式提供清晰的指令列表。
  - **状态检查**: 随时通过 `/sd check` 检查与 WebUI 服务的连接状态。
  - **配置显示**: 使用 `/sd conf` 查看当前所有生效的配置。

## 🚀 快速开始

1.  **安装**:
    *   将此插件文件夹放入 AstrBot 的 `plugins` 目录下。
    *   在 AstrBot 的管理面板中启用此插件。
2.  **配置**:
    *   插件首次加载时，会在 `data/config/` 目录下生成 `SDGen_wzken_config.json` 文件。
    *   您需要在 AstrBot 的管理面板中找到该插件的配置页面，至少填写您的 **Stable Diffusion WebUI API 地址** (`webui_url`)。
3.  **使用**:
    *   发送 `/sd help` 查看所有可用指令。
    *   开始使用 `/i2i`, `/inspire`, `/原生画` 等指令进行创作！

## 📝 指令概览

以下是主要指令的概览，完整列表请使用 `/sd help` 查看。

### 核心绘图指令

- `/i2i <图片> [提示词]`: 图生图。
- `/inspire <图片> [提示词]`: 智能文生图（基于图片理解）。
- `/原生画 <提示词>`: 直接使用提示词进行文生图。

### 主要设置指令

- `/sd set_model <模型名称>`: 设置基础模型。
- `/sd hr_toggle`: 切换高分辨率修复模式。
- `/sd private`: 切换私聊发送模式。
- `/sd forward_reply`: 切换合并转发回复模式。
- `/sd seed <种子值>`: 设置种子，-1为随机。
- `/sd txt2img_steps <步数>`: 设置文生图步数。
- `/sd i2i_denoise <幅度>`: 设置图生图重绘幅度。
- `/sd tag add <名称> <内容>`: 添加本地关键词。

## 🛠️ 配置说明

插件的所有配置项都可以在 AstrBot 管理面板中进行可视化编辑。以下是一些关键配置项的说明：

- `webui_url`: **(必需)** 您的 Stable Diffusion WebUI API 地址，例如 `http://127.0.0.1:7860`。
- `enable_llm_prompt_generation`: 是否为 `/i2i` 和 `/inspire` 命令启用 LLM 辅助生成提示词。
- `enable_upscale`: 是否在生成图片后自动进行放大处理。
- `default_params`: 文生图的默认参数。
- `img2img_params`: 图生图的默认参数。

## 🤝 贡献

欢迎通过提交 Pull Request 或 Issue 来为这个项目做出贡献。

## 📄 开源许可

本项目基于 [MIT License](LICENSE) 开源。
