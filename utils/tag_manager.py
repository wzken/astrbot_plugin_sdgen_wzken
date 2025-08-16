# astrbot_plugin_sdgen_wzken/utils/tag_manager.py

import json
import asyncio
import aiofiles
import re
from typing import Dict, List, Tuple
from astrbot.api.all import logger

class TagManager:
    def __init__(self, tags_file_path: str):
        self.path = tags_file_path
        self.lock = asyncio.Lock()
        self.tags: Dict[str, str] = {}
        # Schedule the initial load, but don't block the constructor
        asyncio.create_task(self._load_tags())

    async def _load_tags(self):
        async with self.lock:
            try:
                async with aiofiles.open(self.path, 'r', encoding='utf-8') as f:
                    content = await f.read()
                    if content:
                        self.tags = json.loads(content)
            except FileNotFoundError:
                logger.info(f"Tags file not found at {self.path}, a new one will be created.")
                self.tags = {}
            except json.JSONDecodeError:
                logger.error(f"Failed to decode JSON from {self.path}. Starting with empty tags.")
                self.tags = {}
            except Exception as e:
                logger.error(f"An unexpected error occurred while loading tags: {e}")
                self.tags = {}

    async def _save_tags(self):
        async with self.lock:
            try:
                async with aiofiles.open(self.path, 'w', encoding='utf-8') as f:
                    await f.write(json.dumps(self.tags, indent=2, ensure_ascii=False))
            except Exception as e:
                logger.error(f"Failed to save tags to {self.path}: {e}")

    def get_all(self) -> Dict[str, str]:
        return self.tags

    def set_tag(self, key: str, value: str):
        self.tags[key] = value
        asyncio.create_task(self._save_tags())

    def del_tag(self, key: str) -> bool:
        if key in self.tags:
            del self.tags[key]
            asyncio.create_task(self._save_tags())
            return True
        return False

    def rename_tag(self, old_key: str, new_key: str) -> bool:
        if old_key in self.tags and new_key not in self.tags:
            self.tags[new_key] = self.tags.pop(old_key)
            asyncio.create_task(self._save_tags())
            return True
        return False

    def import_tags(self, new_tags: Dict[str, str]):
        self.tags.update(new_tags)
        asyncio.create_task(self._save_tags())

    def replace(self, text: str) -> Tuple[str, List[Tuple[str, str]]]:
        replacements = []
        # Sort keys by length, descending, to match longer keys first
        sorted_keys = sorted(self.tags.keys(), key=len, reverse=True)
        
        # Create a regex pattern that finds any of the keys
        # Use word boundaries to avoid matching substrings inside other words
        pattern = re.compile(r'\b(' + '|'.join(map(re.escape, sorted_keys)) + r')\b')

        def repl(match):
            original_word = match.group(0)
            replacement_word = self.tags[original_word]
            replacements.append((original_word, replacement_word))
            return replacement_word

        processed_text = pattern.sub(repl, text)
        return processed_text, replacements

    def fuzzy_search(self, keyword: str) -> Dict[str, str]:
        """Performs a case-insensitive fuzzy search for a keyword in both keys and values."""
        keyword_lower = keyword.lower()
        return {
            k: v for k, v in self.tags.items()
            if keyword_lower in k.lower() or keyword_lower in v.lower()
        }
