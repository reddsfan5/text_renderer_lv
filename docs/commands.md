# 数据生成命令

项目只提供一个正式入口：`text-renderer`。安装开发版本后，可以在任意目录调用：

```powershell
python -m pip install -e .
text-renderer --help
```

在仓库根目录开发时，也可以把 `text-renderer` 等价替换为
`python -m text_renderer.cli`。

## 单行识别数据

单行生成由 YAML 配置驱动：

```powershell
text-renderer single-line `
  --config configs/multilingual.yaml `
  --language korean `
  --num-images 10000 `
  --output-dir D:\dataset\korean
```

`--language` 可以重复传入。命令行中的数据量、输出目录、进程数和存储格式会覆盖
YAML 中相应设置。生成 LMDB 后默认抽取 50 条样本到 `samples/`；传入
`--sample-count 0` 可关闭。

查看全部参数：

```powershell
text-renderer single-line --help
```

## 多行索书号识别数据

```powershell
# 按行（默认布局表 _DEFAULT_PATTERNS_ROW）
text-renderer callnumber-multiline `
  --write-direction row `
  --out D:\dataset\callnumber_rec_lmdb `
  --num 10000 `
  --sample-count 50

# 按列（默认布局表 _DEFAULT_PATTERNS_COLUMN）
text-renderer callnumber-multiline `
  --write-direction column `
  --out D:\dataset\callnumber_rec_col_lmdb `
  --num 10000
```

输出是识别用 LMDB。默认同时生成 `samples/` 标注预览和 `ori_data/` 原图标签。
输出目录已存在时命令会终止；确认需要替换时添加 `--overwrite`。
布局与高宽比说明见
[多行索书号识别 LMDB 生成器](call_number_rec_multiline_lmdb.md)；
迭代记录见 [项目更新说明](UPDATES.md)。

## 索书号检测数据

```powershell
text-renderer callnumber-detection `
  --xlsx D:\data\callnumbers.xlsx `
  --out D:\dataset\callnumber_detection `
  --num 10000 `
  --width 4000 `
  --height 2000
```

输出为图片和对应的 LabelMe JSON。输出目录已存在时同样需要显式传入
`--overwrite`。

## 命令层次

- `text_renderer.cli`：只解析一级子命令并分发。
- `text_renderer.pipelines`：组织完整生成流程、参数校验和输出。
- `text_renderer.renderers`：提供可复用的领域渲染函数，不负责命令行入口。
- `scripts`：只放检查、语料处理、调试和实验工具，不放正式生成入口。
