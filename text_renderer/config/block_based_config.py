
from typing import Type, Dict, Any, Literal

from lv_tools.corpus.text_preprocess import MixedTextSplitter, DelimiterSplitter, TextSplitter
from lv_tools.font.safe_font import FontChoice
from text_renderer.config import SafeTextColorCfg, TextColorCfg, GrayStyleCfg

# 配置模板：映射文本类型 -> 分割器类 + 参数
SPLITTER_CONFIG: Dict[str, Dict[str, Any]] = {
    "0主标题": {
        "splitter_cls": MixedTextSplitter,
        "init_params": {}
    },
    "5作者": {
        "splitter_cls": DelimiterSplitter,
        "init_params": {"delimiter": ","}
    },
    "7索书号": {
        "splitter_cls": DelimiterSplitter,
        "init_params": {"delimiter": "/"}
    },
    "2分辑号": {
        "splitter_cls": MixedTextSplitter,
        "init_params": {}
    },
}


def create_splitter(text_type: str, text: str) -> TextSplitter:
    """根据类型动态实例化分割器"""
    config = SPLITTER_CONFIG.get(text_type)
    if not config:
        raise ValueError(f"Unsupported text type: {text_type}")

    # 获取分割器类和参数
    splitter_cls: Type[TextSplitter] = config["splitter_cls"]
    init_params: Dict[str, Any] = config["init_params"]

    # 动态实例化（确保所有参数包含text）
    return splitter_cls(text=text, **init_params)

FONT_COLOR_CONFIG: Dict[Literal['0主标题', '1副标题', '2分辑号', '3版本', '4丛书项', '5作者', '6出版社', '7索书号',
                 '8杂项'], TextColorCfg] = {
    "0主标题": SafeTextColorCfg(),
    "1副标题": SafeTextColorCfg(),
    "2分辑号": SafeTextColorCfg(),
    "3版本": SafeTextColorCfg(),
    "4丛书项": SafeTextColorCfg(),
    "5作者": SafeTextColorCfg(),
    "6出版社": SafeTextColorCfg(),
    "7索书号": GrayStyleCfg(),
    "8杂项": SafeTextColorCfg()


}









if __name__ == '__main__':
    # 使用示例
    splitter = create_splitter('7索书号',"世界     hello world 못으/로.만드/123ภาษา？ไทยa/bc")
    print(splitter.get_max_divisions())
    ret = splitter.split_text(4)
    print(ret)
