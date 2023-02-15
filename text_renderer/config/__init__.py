import importlib
from importlib.util import spec_from_file_location
import os
import random
import typing
from abc import abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Union
import colorsys
import numpy as np
from PIL.Image import Image as PILImage

from text_renderer.effect import Effects
from text_renderer.layout import Layout
from text_renderer.layout.same_line import SameLineLayout

if typing.TYPE_CHECKING:
    from text_renderer.corpus import Corpus


@dataclass
class PerspectiveTransformCfg:
    """
    Base class for PerspectiveTransform
    """

    x: float = 10
    y: float = 5
    z: float = 1.5
    scale: int = 1
    fovy: int = 50

    @abstractmethod
    def get_xyz(self) -> Tuple[int, int, int]:
        pass


@dataclass
class FixedPerspectiveTransformCfg(PerspectiveTransformCfg):
    def get_xyz(self) -> Tuple[float, float, float]:
        return 15, 15, 1.2


@dataclass
class UniformPerspectiveTransformCfg(PerspectiveTransformCfg):
    """
    x,y,z are uniform distributed
    """

    def get_xyz(self) -> Tuple[float, float, float]:
        x = np.random.uniform(-self.x, self.x)
        y = np.random.uniform(-self.y, self.y)
        z = np.random.uniform(-self.z, self.z)
        return x, y, z


@dataclass
class NormPerspectiveTransformCfg(PerspectiveTransformCfg):
    """
    x,y,z are normally distributed
    """

    def cliped_rand_norm(self, mu=0, sigma3: float = 1):
        """
        :param mu: mean
        :param sigma3: 99% (mu-3*sigma, mu+3*sigma)
        :return:
            float
        """
        # 标准差
        sigma = sigma3 / 3
        dst = sigma * np.random.randn() + mu
        dst = np.clip(dst, 0 - sigma3, sigma3)
        return dst

    def get_xyz(self) -> Tuple[float, float, float]:
        x = self.cliped_rand_norm(0, self.x)
        y = self.cliped_rand_norm(0, self.y)
        z = self.cliped_rand_norm(0, self.z)
        return x, y, z


class TextColorCfg:
    """
    Base class for TextColorCfg
    """

    @abstractmethod
    def get_color(self, bg_img: PILImage) -> Tuple[int, int, int, int]:
        pass


@dataclass
class FixedTextColorCfg(TextColorCfg):
    # For generate effect/layout example
    def get_color(self, bg_img: PILImage) -> Tuple[int, int, int, int]:
        alpha = 255
        text_color = (0, 0, 0, alpha)
        # text_color = (255, 50, 0, alpha)

        return text_color


@dataclass
class SimpleTextColorCfg(TextColorCfg):
    """
    Randomly use mean value of background image
    """

    alpha: Tuple[int, int] = (240, 255)

    def get_color(self, bg_img: PILImage) -> Tuple[int, int, int, int]:
        np_img = np.array(bg_img)
        mean = np.mean(np_img, axis=(0, 1))[:3]
        # 注意这里是RGB还是BGR。
        '''
        在HSV空间，1/2 是补色。
                values = [
            # rgb, hsv
            ((0.0, 0.0, 0.0), (  0  , 0.0, 0.0)), # black
            ((0.0, 0.0, 1.0), (4./6., 1.0, 1.0)), # blue
            ((0.0, 1.0, 0.0), (2./6., 1.0, 1.0)), # green
            ((0.0, 1.0, 1.0), (3./6., 1.0, 1.0)), # cyan
            ((1.0, 0.0, 0.0), (  0  , 1.0, 1.0)), # red
            ((1.0, 0.0, 1.0), (5./6., 1.0, 1.0)), # purple
            ((1.0, 1.0, 0.0), (1./6., 1.0, 1.0)), # yellow
            ((1.0, 1.0, 1.0), (  0  , 0.0, 1.0)), # white
            ((0.5, 0.5, 0.5), (  0  , 0.0, 0.5)), # grey
        ]
        '''
        color_h, color_s, color_v = colorsys.rgb_to_hsv(*(mean / 255).tolist())
        anti_h = color_h - random.uniform(.25, .5) if color_h > .5 else color_h + random.uniform(.25, .5)
        # anti_s = color_s - random.uniform(1 / 4, .5) if color_s > .5 else color_s + random.uniform(1 / 4, .5)
        anti_s = random.uniform(.1,1.0)
        anti_v = color_v - random.uniform(.3, .5) if color_v > .5 else color_v + random.uniform(.3, .5)

        anti_r,anti_g,anti_b = (np.array(colorsys.hsv_to_rgb(anti_h,anti_s,anti_v))*255).astype(np.uint8).tolist()

        alpha = np.random.randint(*self.alpha)
        safe_value = 70
        # r = np.random.randint(0,int(mean[0])-safe_value) if mean[0]>127 else np.random.randint(int(mean[0])+safe_value,255)
        # g = np.random.randint(0,int(mean[1])-safe_value) if mean[1]>127 else np.random.randint(int(mean[1])+safe_value,255)
        # b = np.random.randint(0,int(mean[2])-safe_value) if mean[2]>127 else np.random.randint(int(mean[2])+safe_value,255)

        # todo lvxiaodong safe color   https://docs.python.org/zh-cn/3/library/colorsys.html
        # channel = []
        # for i in range(3):
        #     left = min((int(mean[i]) - safe_value) % 255, int(mean[i] - safe_value - 30) % 255)
        #     right = max((int(mean[i]) - safe_value) % 255, int(mean[i] - safe_value - 30) % 255)
        #     # channel.append(np.random.randint(left, right))
        #
        #     # todo lvxiaodong tem
        #     channel.append(0)

        # text_color = (channel[0], channel[1], channel[2], alpha)
        text_color = (anti_r, anti_g, anti_b, alpha)

        return text_color

    # def get_color(self, bg_img: PILImage) -> Tuple[int, int, int, int]:
    #     np_img = np.array(bg_img)
    #     mean = np.mean(np_img)
    #
    #     alpha = np.random.randint(*self.alpha)
    #
    #
    #
    #     r = np.random.randint(0, int(mean * 0.7))
    #     g = np.random.randint(0, int(mean * 0.7))
    #     b = np.random.randint(0, int(mean * 0.7))
    #
    #
    #     text_color = (r, g, b, alpha)
    #
    #     return text_color


# noinspection PyUnresolvedReferences
@dataclass
class RenderCfg:
    """

    Parameters
    ----------
    corpus : Union[Corpus, List[Corpus]]

    corpus_effects : Union[Effects, List[Effects]]
        Effects apply on text mask image of each corpus.
        Effects used at this stage must return changed bbox of text if it modified it.
    bg_dir : Path
        Background image directory
    pre_load_bg_img : bool
        True: Load all image into memory
    layout : Layout
        Layout will applied if corpus is a List
    perspective_transform : PerspectiveTransformCfg
        Apply Perspective Transform
    layout_effects : Effects
        Effects apply on merged text mask image output by Layout.
    render_effects : Effects
        Effects apply on final image.
    height : int
        Resize(keep ratio) image to height, set -1 disables resize
    gray : bool
        Save image as gray image
    text_color_cfg : TextColorCfg
        If not None, will overwrite text_color_cfg in CorpusCfg
        useful to set same text color when use multi corpus
    return_bg_and_mask: bool
    """

    corpus: Union["Corpus", List["Corpus"]]
    corpus_effects: Union[Effects, List[Effects]] = None
    bg_dir: Path = None
    pre_load_bg_img: bool = True
    layout: Layout = SameLineLayout()
    perspective_transform: PerspectiveTransformCfg = None
    layout_effects: Effects = None
    render_effects: Effects = None
    height: int = 32
    gray: bool = True
    text_color_cfg: TextColorCfg = None
    return_bg_and_mask: bool = False


# noinspection PyUnresolvedReferences
@dataclass
class GeneratorCfg:
    """
    Parameters
    ----------
    num_image : int
        Number of images generated
    save_dir : Path
        The directory where the data is stored
    render_cfg : RenderCfg
        Configuration of Render
    """

    num_image: int
    save_dir: Path
    render_cfg: RenderCfg


def get_cfg(config_file: str) -> List[GeneratorCfg]:
    """

    Args:
        config_file: full path of a config file

    Returns:

    """
    print(os.path.join(os.getcwd(), config_file))
    module = import_module_from_file(config_file)
    cfgs = getattr(module, "configs", None)
    if cfgs is None:
        raise RuntimeError(f"Load configs failed: {config_file}")

    assert all(
        [isinstance(cfg, GeneratorCfg) for cfg in cfgs]
    ), "Please make sure all items in configs is GeneratorCfg"

    return cfgs


def import_module_from_file(full_path_to_module):
    """
    Import a module given the full path/filename of the .py file

    https://stackoverflow.com/questions/28836713/from-folder-name-import-variable-python-3-4-2
    """
    module = None
    try:

        # Get module name and path from full path
        module_dir, module_file = os.path.split(full_path_to_module)
        module_name, module_ext = os.path.splitext(module_file)

        # Get module "spec" from filename
        spec = spec_from_file_location(module_name, full_path_to_module)

        module = spec.loader.load_module()

    except Exception as ec:
        # Simple error printing
        # Insert "sophisticated" stuff here
        print(ec)

    finally:
        return module
