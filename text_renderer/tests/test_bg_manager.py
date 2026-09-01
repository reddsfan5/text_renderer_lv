from pathlib import Path

from PIL import Image

from text_renderer.bg_manager import BgManager


def test_transparency_check_skips_jpeg_decode(tmp_path: Path):
    invalid_jpeg = tmp_path / "background.jpg"
    invalid_jpeg.write_bytes(b"not an image")

    manager = BgManager.__new__(BgManager)
    assert manager._is_transparent_image(invalid_jpeg) is False


def test_transparency_check_detects_png_alpha(tmp_path: Path):
    opaque = tmp_path / "opaque.png"
    transparent = tmp_path / "transparent.png"
    Image.new("RGBA", (2, 2), (255, 255, 255, 255)).save(opaque)
    Image.new("RGBA", (2, 2), (255, 255, 255, 0)).save(transparent)

    manager = BgManager.__new__(BgManager)
    assert manager._is_transparent_image(opaque) is False
    assert manager._is_transparent_image(transparent) is True
