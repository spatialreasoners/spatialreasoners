import io
from functools import lru_cache
from pathlib import Path

from PIL import ImageFont


class FontCache:
    def __init__(self, font_path: Path):
        # Load the font file into memory to avoid file system reads
        with open(font_path, "rb") as f:
            self.font_data = f.read()

    @lru_cache(maxsize=None)
    def get_font(self, size: int) -> ImageFont.FreeTypeFont:
        """
        Retrieve a FreeTypeFont instance for a specific size.
        Cached to avoid creating duplicate instances.
        """
        return ImageFont.truetype(io.BytesIO(self.font_data), size)