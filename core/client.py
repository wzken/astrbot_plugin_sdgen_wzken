# astrbot_plugin_sdgen_v2/core/client.py

import asyncio
import base64
import aiohttp
from typing import Dict, List, Any, Optional
from astrbot.api.all import logger

class SDAPIClient:
    def __init__(self, config: dict):
        self.config = config
        self._session: Optional[aiohttp.ClientSession] = None
        self._base_url = self.config.get("webui_url", "http://127.0.0.1:7860").rstrip('/')

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create an aiohttp session."""
        if self._session is None or self._session.closed:
            timeout_seconds = self.config.get("session_timeout", 120)
            timeout = aiohttp.ClientTimeout(total=timeout_seconds)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session

    async def close(self):
        """Close the aiohttp session."""
        if self._session and not self._session.closed:
            await self._session.close()

    async def _request(self, method: str, endpoint: str, **kwargs) -> Any:
        """General request handler."""
        session = await self._get_session()
        url = f"{self._base_url}{endpoint}"
        try:
            async with session.request(method, url, **kwargs) as response:
                response.raise_for_status()
                return await response.json()
        except aiohttp.ClientError as e:
            logger.error(f"API request to {url} failed: {e}")
            raise ConnectionError(f"Failed to connect to Stable Diffusion API at {url}.") from e

    async def post(self, endpoint: str, payload: Dict) -> Dict:
        return await self._request("post", endpoint, json=payload)

    async def get(self, endpoint: str) -> Any:
        return await self._request("get", endpoint)

    async def get_sd_models(self) -> List[Dict]:
        return await self.get("/sdapi/v1/sd-models")

    async def get_samplers(self) -> List[Dict]:
        return await self.get("/sdapi/v1/samplers")

    async def get_loras(self) -> List[Dict]:
        return await self.get("/sdapi/v1/loras")

    async def get_upscalers(self) -> List[Dict]:
        return await self.get("/sdapi/v1/upscalers")

    async def get_embeddings(self) -> Dict:
        return await self.get("/sdapi/v1/embeddings")

    async def get_schedulers(self) -> List[Dict]:
        return await self.get("/sdapi/v1/schedulers")

    async def txt2img(self, payload: Dict) -> Dict:
        return await self.post("/sdapi/v1/txt2img", payload)

    async def img2img(self, payload: Dict) -> Dict:
        return await self.post("/sdapi/v1/img2img", payload)

    async def extra_single_image(self, payload: Dict) -> Dict:
        """For upscaling or other extra features."""
        return await self.post("/sdapi/v1/extra-single-image", payload)

    async def set_model(self, model_name: str) -> None:
        """Set the active Stable Diffusion model."""
        payload = {"sd_model_checkpoint": model_name}
        await self.post("/sdapi/v1/options", payload)

    async def check_availability(self) -> bool:
        """Check if the WebUI API is available."""
        try:
            await self.get("/sdapi/v1/progress")
            return True
        except ConnectionError:
            return False
            
    async def download_image_as_base64(self, url: str) -> str:
        """Downloads an image from a URL and returns it as a base64 string."""
        session = await self._get_session()
        try:
            async with session.get(url) as response:
                response.raise_for_status()
                image_bytes = await response.read()
                return base64.b64encode(image_bytes).decode("utf-8")
        except aiohttp.ClientError as e:
            raise ConnectionError(f"Failed to download image from {url}.") from e
