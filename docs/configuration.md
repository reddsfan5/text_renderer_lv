# YAML 配置说明

主生成入口 `main.py` 同时支持新的 YAML 配置和原有 Python 配置。推荐把会改变数据分布、输出格式或资源占用的参数放进 YAML；命令行只保留任务启动时最常覆盖的参数。

配置优先级为：命令行参数 > YAML 参数 > 程序默认值。

## 运行方式

直接使用 YAML 中的全部配置：

```powershell
python main.py --config example_data/korean_vertical.yaml
```

临时生成 20 张并覆盖输出位置，无需修改 YAML：

```powershell
python main.py `
  --config example_data/korean_vertical.yaml `
  --num-images 20 `
  --output-dir F:/dataset/OCR/smoke_test `
  --num-processes 0
```

仍然可以使用原来的 Python 配置：

```powershell
python main.py --config example_data/effect_layout_example.py --dataset lmdb
```

## 命令行核心参数

| 参数 | 作用 |
| --- | --- |
| `--config` | YAML 或旧版 Python 配置路径 |
| `--num-images` | 覆盖生成数量 |
| `--output-dir` | 覆盖输出目录 |
| `--dataset` | 覆盖存储类型：`lmdb` 或 `img` |
| `--num-processes` | 覆盖生成进程数；`0` 为单进程 |
| `--log-period` | 覆盖进度日志间隔百分比 |

下划线形式的旧参数 `--num_processes`、`--log_period` 继续兼容。

## YAML 中暴露的重要参数

| 配置段 | 重要参数 |
| --- | --- |
| `runtime` | 进程数、进度日志间隔 |
| `storage` | LMDB/图片模式、JPEG 质量、LMDB 初始容量、扩容量和写缓存 |
| `generator` | 样本数量、输出目录；`{timestamp}` 会替换为运行时间 |
| `corpus` | 语料文件、字符集、标点策略、空格规范化、最短/最长长度、长文本处理、横竖排、字符翻转、字符间距 |
| `font` | 字体目录、随机字号范围 |
| `background` | 背景目录、是否预加载到内存 |
| `render` | 单行/多行、灰度/彩色文字、颜色范围、透明度、透视变换、文字增强 |

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

例如：

```yaml
render:
  corpus_effects:
    - type: padding
      p: 0.5
      w_ratio: [0.0, 0.05]
      h_ratio: [0.0, 0.2]
      center: false
    - type: dropout_rand
      p: 0.2
      dropout_p: [0.1, 0.2]
```

复杂的多语料布局、自定义 Python 增强器等高级扩展仍使用 Python 配置，避免 YAML 变成另一套程序语言。
