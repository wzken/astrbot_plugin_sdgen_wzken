# astrbot_plugin_sdgen_v2/core/config.py

from pydantic import BaseModel, Field, HttpUrl
from typing import List, Dict, Optional

class Txt2ImgParams(BaseModel):
    width: int = Field(default=512, ge=64, le=1920)
    height: int = Field(default=512, ge=64, le=1920)
    steps: int = Field(default=20, ge=10, le=50)
    sampler: str = "Euler a"
    scheduler: str = "Karras"
    cfg_scale: float = Field(default=7.0, ge=1.0, le=20.0)
    batch_size: int = Field(default=1, ge=1, le=10)
    n_iter: int = Field(default=1, ge=1, le=5)

class Img2ImgParams(BaseModel):
    denoising_strength: float = Field(default=0.75, ge=0.0, le=1.0)
    steps: int = Field(default=20, ge=10, le=50)
    sampler: str = "Euler a"
    scheduler: str = "Karras"
    cfg_scale: float = Field(default=7.0, ge=1.0, le=20.0)
    batch_size: int = Field(default=1, ge=1, le=10)
    n_iter: int = Field(default=1, ge=1, le=5)

class UpscaleParams(BaseModel):
    upscaler: str = "Latent"
    upscale_factor: float = Field(default=2.0, ge=1.0, le=8.0)

class PluginConfig(BaseModel):
    webui_url: HttpUrl = "http://127.0.0.1:7860"
    session_timeout: int = Field(default=120, ge=10, le=300)
    max_concurrent_tasks: int = Field(default=10, ge=1)
    
    enable_llm_prompt_generation: bool = True
    enable_upscale: bool = False
    enable_show_positive_prompt: bool = True
    enable_forward_message: bool = False

    positive_prompt_global: str = ""
    negative_prompt_global: str = "(worst quality, low quality:1.4), deformed, bad anatomy"
    
    whitelist_groups: List[str] = []
    blacklist_groups: List[str] = []
    
    # Whitelist-exclusive prompts
    positive_prompt_whitelist: str = "masterpiece, best quality"
    negative_prompt_whitelist: str = ""

    base_model: Optional[str] = None
    llm_prompt_prefix: str = "Your default LLM prompt prefix here..."
    prompt_guidelines: str = ""

    default_params: Txt2ImgParams = Field(default_factory=Txt2ImgParams)
    img2img_params: Img2ImgParams = Field(default_factory=Img2ImgParams)
    upscale_params: UpscaleParams = Field(default_factory=UpscaleParams)

class ConfigManager:
    def __init__(self, raw_config: Dict):
        self.config = PluginConfig.parse_obj(raw_config)

    def get(self, key: str, default=None):
        return getattr(self.config, key, default)

    def set(self, key: str, value):
        if hasattr(self.config, key):
            setattr(self.config, key, value)
            # Here you would also save the config to a file
            # For now, we just update the in-memory model
        else:
            raise AttributeError(f"Config key '{key}' not found.")

    def save_to_file(self, path: str):
        # To be implemented: save self.config.dict() to a JSON file
        pass
