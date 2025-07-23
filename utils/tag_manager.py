# astrbot_plugin_sdgen_v2/utils/tag_manager.py

import json
import os
import threading
from typing import Dict, List, Tuple

class TagManager:
    def __init__(self, tags_file_path: str):
        self.path = tags_file_path
        self.lock = threading.Lock()
        self.tags = self._load()

    def _load(self) -> Dict[str, str]:
        """Loads tags from the JSON file."""
        if os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (IOError, json.JSONDecodeError):
                # logger.error(f"Failed to load tags from {self.path}: {e}")
                return {}
        return {}

    def _save(self):
        """Saves the current tags to the JSON file."""
        try:
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(self.tags, f, ensure_ascii=False, indent=2)
        except IOError:
            # logger.error(f"Failed to save tags to {self.path}: {e}")
            pass

    def replace(self, text: str) -> Tuple[str, List[str]]:
        """
        Replaces keywords in the text with their corresponding tag values.
        Returns the modified text and a list of keys that were changed.
        """
        changed_keys = []
        # Sort by key length descending to replace longer matches first (e.g., "blue eyes" before "blue")
        sorted_tags = sorted(self.tags.items(), key=lambda x: len(x[0]), reverse=True)
        
        for key, value in sorted_tags:
            if key in text:
                text = text.replace(key, value)
                if key not in changed_keys:
                    changed_keys.append(key)
        return text, changed_keys

    def set_tag(self, key: str, value: str):
        """Adds or updates a tag."""
        with self.lock:
            self.tags[key] = value
            self._save()

    def del_tag(self, key: str) -> bool:
        """Deletes a tag. Returns True if successful, False otherwise."""
        with self.lock:
            if key in self.tags:
                del self.tags[key]
                self._save()
                return True
            return False

    def get_all(self) -> Dict[str, str]:
        """Returns a copy of all tags."""
        return self.tags.copy()

    def import_tags(self, new_tags: Dict[str, str], overwrite: bool = True):
        """
        Imports a dictionary of tags.
        If overwrite is True, existing keys will be updated.
        """
        with self.lock:
            if overwrite:
                self.tags.update(new_tags)
            else:
                for key, value in new_tags.items():
                    self.tags.setdefault(key, value)
            self._save()
