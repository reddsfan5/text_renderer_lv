# YAML 配置说明

主入口 `main.py` 支持 YAML 配置和原有 Python 配置。推荐使用
`example_data/multilingual.yaml`：它把公共生成参数放在顶层，把相互耦合的
语料、字符集、字体和文字排版放进同一个语言档案。

配置优先级为：命令行参数 > YAML 参数 > 程序默认值。

## 多语言配置结构

```yaml
version: 2

generator:
  active_languages: [korean]
  num_images: 100
  output_dir: "F:/output/{timestamp}/{language}"

background:
  dir: "D:/assets/backgrounds"
  preload: true

render:
  write_mode: oneline
  text_color_mode: gray

languages:
  korean:
    corpus:
      path: "F:/corpus/korean.txt"
      charset_path: "D:/charset/korean.txt"
      min_length: 2
      max_length: 12
      long_text: split
    font:
      dir: "D:/fonts/korean"
      size: [30, 36]
    text:
      orientation: vertical
      rotate_characters: true
      clip_length: 12
      char_spacing: 0
      join_str: ""
```

`languages` 下可以继续增加 `chinese`、`tibetan`、`japanese` 等任意名称的
语言档案。每个档案必须包含 `corpus` 和 `font`，从结构上避免错配语料和
字体。`text` 紧邻二者，保存与该文字系统相关的横竖排和字符旋转参数。

示例配置已根据 `effect_layout_example.py` 整理出以下可直接选择的档案：

| 档案名 | 内容 |
| --- | --- |
| `chinese` | 中文书名语料及简体字体 |
| `korean` | 韩文补充语料及韩文字体 |
| `korean_catalog` | 280 万韩文书名、作者、出版社语料及韩文字体 |
| `japanese` | 日文书名语料及日文字体 |
| `japanese_callnumber` | 日文索书号语料及日文字体 |
| `french` | 法文书目信息及法文字体 |
| `spanish` | 西班牙文语料及拉丁字体 |
| `russian` | 俄文规范化语料及俄文字体 |
| `german` | 德文 Open Library 语料及德文字体 |
| `tibetan` | 藏文语料及藏文字体 |

同一语言有多套字体时，可在语言档案内声明字体变体。修改 `active` 即可切换，
不需要重新查找或复制路径：

```yaml
languages:
  chinese:
    corpus:
      path: "F:/corpus/chinese.txt"
    font:
      active: standard
      variants:
        standard:
          dir: "D:/fonts/chinese-standard"
          size: [30, 36]
        single:
          dir: "D:/fonts/chinese-single"
          size: [20, 21]
```

`multilingual.yaml` 中已收录旧 Python 示例定义的全部字体目录，包括中文
标准、mini、艺术、单字体、常规字体，以及拉丁手写、多语言拉丁、韩文、
日文、俄文和藏文字体。

顶层 `background`、`render` 是公共默认值。某种语言确实需要不同配置时，
可在该语言档案内增加同名节点，只覆盖需要变化的字段：

```yaml
languages:
  tibetan:
    corpus:
      path: "F:/corpus/tibetan.txt"
    font:
      dir: "D:/fonts/tibetan"
      size: [30, 36]
    text:
      orientation: horizontal
      rotate_characters: false
    render:
      text_color_mode: color
    background:
      dir: "D:/assets/tibetan_backgrounds"
```

语言档案还可用 `num_images` 和 `output_dir` 覆盖公共生成数量和输出目录。
输出路径支持 `{timestamp}` 和 `{language}` 占位符。

## 运行方式

使用 YAML 中 `active_languages` 指定的语言：

```powershell
python main.py --config example_data/multilingual.yaml
```

临时只生成藏文：

```powershell
python main.py --config example_data/multilingual.yaml --language tibetan
```

一次生成中文和韩文；`--language` 可重复使用：

```powershell
python main.py `
  --config example_data/multilingual.yaml `
  --language chinese `
  --language korean
```

覆盖少量核心参数：

```powershell
python main.py `
  --config example_data/multilingual.yaml `
  --language korean `
  --num-images 20 `
  --output-dir F:/dataset/OCR/smoke_test `
  --num-processes 0
```

选择多个语言并指定 `--output-dir` 时，程序会自动在该目录下按语言名称建立
子目录，防止数据相互覆盖。

原来的 Python 配置仍可使用：

```powershell
python main.py --config example_data/effect_layout_example.py --dataset lmdb
```

旧版单语言 YAML（`version: 1`）也继续兼容，例如
`example_data/korean_vertical.yaml`，但它不支持 `--language`。

## 命令行核心参数

| 参数 | 作用 |
| --- | --- |
| `--config` | YAML 或旧版 Python 配置路径 |
| `--language` | 选择语言档案；可重复传入 |
| `--num-images` | 覆盖每个所选语言的生成数量 |
| `--output-dir` | 覆盖输出根目录 |
| `--dataset` | 覆盖存储类型：`lmdb` 或 `img` |
| `--num-processes` | 覆盖生成进程数；`0` 为单进程 |
| `--log-period` | 覆盖进度日志间隔百分比 |

下划线形式的旧参数 `--num_processes`、`--log_period` 继续兼容。

## 语料与渲染参数

`corpus.long_text` 支持：

- `split`：按最大长度切成多条；
- `truncate`：只保留开头；
- `drop`：丢弃超长文本。

`render.corpus_effects` 当前支持：

- `padding`
- `dropout_rand`
- `dropout_horizontal`
- `dropout_vertical`
- `emboss`
- `motion_blur`

复杂的多语料布局和自定义 Python 增强器仍可使用 Python 配置，避免把 YAML
变成另一套程序语言。
