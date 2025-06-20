import numpy as np
from PIL import Image

from text_renderer.render_by_gt_img import WordRender


def handwritting_on_bg(
        word_render: WordRender,
        text,
        text_color: tuple[int, int, int, int] = (0, 0, 0, 255),
        save_dir: str = ''
) -> Image:
    """

    """

    text_img, text = word_render(text)
    w, h = text_img.size

    xmin, ymin, xmax, ymax = 0, 0, w, h

    box_n = np.asarray([[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]])

    bbox = box_n.tolist()
    font_base = 'handwritting'

    return text_img, bbox, font_base
