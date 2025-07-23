# astrbot_plugin_sdgen_wzken/main.py

"""
Plugin Name: SDGen_wzken
Author: wzken
Version: 2.0.0
Description: A smarter and more powerful image generation plugin for AstrBot using Stable Diffusion.
"""

from astrbot.api.star import register, Star, Context
from astrbot.api.all import AstrBotConfig, logger
from pathlib import Path

from .core.config import ConfigManager
from .core.client import SDAPIClient
from .core.generation import GenerationManager
from .utils.tag_manager import TagManager
from .utils.llm_helper import LLMHelper
from .commands.i2i import register_i2i_commands
from .commands.inspire import register_inspire_commands
from .commands.settings import register_settings_commands
from .commands.native import register_native_commands

@register("SDGen_wzken", "wzken", "SDGen_wzken", "2.0.0")
class SDGeneratorWzken(Star):
    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        
        logger.info("SDGen_wzken plugin loaded. Initializing services...")
        self._initialize_services(config)
        self._register_commands()
        logger.info("SDGen_wzken plugin initialization complete.")

    def _initialize_services(self, raw_config: AstrBotConfig):
        """Initializes all core services and managers using dependency injection."""
        plugin_dir = Path(__file__).parent.resolve()
        
        # 1. Config Manager
        self.config_manager = ConfigManager(raw_config)
        
        # 2. API Client
        self.api_client = SDAPIClient(self.config_manager.config)
        
        # 3. Generation Manager
        self.generator = GenerationManager(self.config_manager.config, self.api_client)
        
        # 4. Tag Manager
        tags_file = plugin_dir / "data" / "tags.json"
        tags_file.parent.mkdir(exist_ok=True) # Ensure data directory exists
        self.tag_manager = TagManager(str(tags_file))

        # 5. LLM Helper
        self.llm_helper = LLMHelper(self.context)

    def _register_commands(self):
        """Registers all command handlers."""
        register_native_commands(self)
        register_i2i_commands(self)
        register_inspire_commands(self)
        register_settings_commands(self)

    async def terminate(self):
        """Called when the plugin is unloaded/disabled to clean up resources."""
        if self.api_client:
            await self.api_client.close()
        logger.info("SDGen_wzken plugin terminated and resources cleaned up.")
