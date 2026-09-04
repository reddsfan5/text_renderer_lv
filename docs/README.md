# 文档索引

本目录是 `text_renderer_lv` 的使用与设计文档入口。  
生成数据请优先看命令文档；行为变更请看更新说明。

## 快速入口

| 文档 | 说明 |
| --- | --- |
| [commands.md](commands.md) | 正式 CLI：`single-line` / `callnumber-multiline` / `callnumber-detection` |
| [configuration.md](configuration.md) | YAML 多语言配置（语料、字体、颜色、排版） |
| [call_number_rec_multiline_lmdb.md](call_number_rec_multiline_lmdb.md) | 多行索书号识别：行/列书写、默认 patterns、高宽比护栏 |
| [UPDATES.md](UPDATES.md) | 迭代更新说明（按时间倒序，追踪设计取舍） |

仓库根目录还有 [README.md](../README.md)（项目总览与目录结构）。

## 常用命令

```powershell
# 安装开发版
python -m pip install -e .

# 单行多语言识别数据
python -m text_renderer.cli single-line --config configs/multilingual.yaml --language korean

# 多行索书号识别（按行 / 按列）
python -m text_renderer.cli callnumber-multiline --write-direction row --num 10000
python -m text_renderer.cli callnumber-multiline --write-direction column --num 10000
```

完整参数与输出约定见 [commands.md](commands.md)。

## 文档地图

```text
docs/
├── README.md                          # 本索引
├── commands.md                        # 命令行入口
├── configuration.md                   # YAML 配置
├── call_number_rec_multiline_lmdb.md  # 索书号多行识别专题
├── UPDATES.md                         # 迭代记录
├── config.rst / dataset.rst           # Sphinx API 说明（上游风格）
├── corpus/ effect/ layout/            # 组件 API
└── note/render_pipeline.rst           # 渲染流水线笔记
```

## 维护约定

1. **新功能**：专题文档写清楚「怎么用 + 为什么这样设计」；在 [UPDATES.md](UPDATES.md) 顶部追加一条摘要。  
2. **改 CLI / 默认行为**：同步更新 [commands.md](commands.md) 与对应专题文档。  
3. **本文件只做索引**：不堆长教程；细节放到专题页。  
4. Sphinx 构建（可选）：

```bash
cd docs
make html
# 打开 _build/html/index.html
```

## 上游参考

本仓库基于 [oh-my-ocr/text_renderer](https://github.com/oh-my-ocr/text_renderer)。  
Corpus / Effect / Layout 等模块 API 仍可参考上游文档：  
https://oh-my-ocr.github.io/text_renderer/index.html
