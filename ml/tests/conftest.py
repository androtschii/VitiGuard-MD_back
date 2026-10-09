from pathlib import Path

from PIL import Image, ImageDraw


def make_image(path: Path, seed: int, size: tuple[int, int] = (64, 48)) -> None:
    """Картинка с узором, зависящим от seed: у разных seed разный dHash."""
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", size, (20, 120, 20))
    draw = ImageDraw.Draw(image)
    width, height = size
    for index in range(6):
        x = (seed * 37 + index * 11) % width
        y = (seed * 17 + index * 23) % height
        draw.ellipse(
            (x, y, x + width // 4, y + height // 4), fill=(200, 180 - index * 20, 40)
        )
    image.save(path)
