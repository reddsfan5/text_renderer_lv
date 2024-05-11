import time

from text_renderer.config import get_cfg
from text_renderer.render_book_spine import BookSpineRender

if __name__ == '__main__':
    config_path = r'D:\lxd_code\text_renderer_lv\example_data\config_for_book_spine_info_gen.py'
    generator_cfg = get_cfg(config_path)[0]
    s = time.time()
    for i in range(100):
        render = BookSpineRender(generator_cfg.render_cfg)
        # data = render()
    print(time.time()-s)
