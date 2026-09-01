# 多行索书号识别 LMDB 生成器

脚本：[`callnumber_rec_multiline_lmdb.py`](../scripts/generate/callnumber_rec_multiline_lmdb.py)

该脚本用于生成多行索书号识别切片，并直接写入 LMDB。每条记录包含图像、完整文本标签、四点文本框、分数、文件名和字体文件名，可用于 OCR 识别训练及检测/识别联合数据。

## 快速开始

在仓库根目录执行：

```powershell
python -m scripts.generate.callnumber_rec_multiline_lmdb --num 10000 --out .\callnumber_lmdb
```

默认使用 `learned` 策略和内置索书号分布模型，不需要 Excel、pandas 或任何索书号数据文件。

常用参数示例：

```powershell
# 固定随机种子，生成 50000 条
python -m scripts.generate.callnumber_rec_multiline_lmdb --num 50000 --seed 2026 --out .\callnumber_lmdb

# 使用语法策略，而不是内置分布策略
python -m scripts.generate.callnumber_rec_multiline_lmdb --generator-strategy grammar --num 10000

# 首行大写字母按内置统计频率采样；默认是 A-Z 均匀采样
python -m scripts.generate.callnumber_rec_multiline_lmdb --line1-letter-mode learned --num 10000
```

运行前需要确保渲染器已经配置可用字体。输出目录是 LMDB 目录，脚本完成后可直接交给现有 OCR 数据读取流程。

## 索书号生成策略

策略接口定义在 [`call_number_generator.py`](../text_renderer/corpus/call_number_generator.py)。两种策略都实现 `generate()` 和 `summary()`，每次调用 `generate()` 返回一个按书写顺序排列的非空行列表。

### `learned`：内置分布模型

`BuiltInDistributionCallNumberGenerator` 将参考数据归纳出的规律固化为代码常量，包括：

- 两行/三行索书号的比例；
- 各行可见字符长度分布；
- 首行的字母-数字-小数点/连接符模板；
- 第二行的数字和 `= - ( ) :` 等标点模板；
- 第三行四位年份的范围（2005--2024）及年代采样；
- 首行字母的统计频率（仅在 `--line1-letter-mode learned` 时使用）。

它只采样规则和随机字符，不保存或抽取原始行，因此可以无限生成。运行时完全不访问外部索书号文件。

### `grammar`：通用语法模型

- 第一行：1--2 个大写字母，后面补数字至指定长度；
- 第二行：纯数字；
- 第三行：纯数字，长度达到配置要求时可按概率加入 `v` 或 `.v` 前缀；
- 任意行长度为 `0` 时，该行不写入。

## `patterns` 配置

`--patterns` 接收 JSON 数组。每项包含 `lengths` 和可选的 `weight`：

```powershell
python -m scripts.generate.callnumber_rec_multiline_lmdb `
  --patterns '[{"lengths":[2,2,2],"weight":1.5},{"lengths":[2,1,0],"weight":2.5},{"lengths":[0,0,3],"weight":1}]' `
  --num 10000
```

规则：

- `lengths` 最多 3 个整数，对应第 1、2、3 行；
- 长度必须是非负整数，没有固定的 4 字符上限；
- 长度为 `0` 表示省略对应行，例如 `[2, 1, 0]` 只生成两行；
- `weight` 是该结构被采样的相对权重，至少有一个结构权重大于 0；
- 长度统计的是最终可见字符数，`.`、`-`、`=`、`(`、`)`、`v` 等字符都计入长度。

脚本内置的默认结构位于 `scripts.generate.callnumber_rec_multiline_lmdb` 的 `_DEFAULT_PATTERNS`，当前包含 `[2,2,2]`、`[3,3,3]`、`[2,1,1]`、`[4,4,0]`、`[5,5,0]`、`[6,6,0]`、`[7,7,0]`、`[2,1,0]`、`[2,0,2]` 和 `[3,0,3]`。

## 图像合成和增强

生成流程如下：

1. 按策略生成索书号各行；
2. 在书脊色背景上绘制带阴影、边框和轻微纸张色差的文本区域；
3. 对整张图加入渐变照明和细微颗粒，避免纯白、纯平背景；
4. 从文字紧框向四周裁剪扩展区域，扩展内容来自同一张背景图，而不是额外白边；
5. 按文字紧框的高宽比 `h / w` 决定是否逆时针旋转；
6. 缩放图像和四点框后写入 LMDB。

### 旋转规则

默认参数：

```text
rotation_aspect_ratio_threshold_l = 0.7
rotation_aspect_ratio_threshold_r = 1.5
rotation_probability = 0.5
```

- `h / w <= 0.7`：文字很扁，不旋转；
- `0.7 < h / w < 1.5`：按 `rotation_probability` 随机逆时针旋转 90°；
- `h / w >= 1.5`：文字很高，必定逆时针旋转 90°。

可通过以下参数调整：

```powershell
python -m scripts.generate.callnumber_rec_multiline_lmdb `
  --rotation-aspect-ratio-threshold-l 0.8 `
  --rotation-aspect-ratio-threshold-r 1.8 `
  --rotation-prob 0.35
```

旋转后的四点框顺序为：**左下 → 左上 → 右上 → 右下**。图像和框始终同步旋转、缩放。

### 尺寸和噪声

- `--scale-min` / `--scale-max`：渲染字号/画布的随机缩放范围，默认 `1.6`--`2.2`；
- `--output-scale`：写入 LMDB 前的最终宽高比例，默认 `0.8`；
- `--side-ratio`：文字紧框四周的背景扩展比例，默认 `1.0`；
- `--noise`：开启额外噪声和模糊增强。

示例：

```powershell
python -m scripts.generate.callnumber_rec_multiline_lmdb `
  --output-scale 0.5 `
  --side-ratio 1.5 `
  --noise `
  --num 10000
```

## 标签和 LMDB 字段

标签由实际绘制的各行文本拼接而成，行之间使用一个空格。例如三行文本 `I12`、`34=5`、`2024` 的标签是：

```text
I12 34=5 2024
```

第三行中的 `v` 或 `.v`（如果策略生成）会原样出现在标签中，不会被丢弃。

每条记录写入的信息包括：

| 字段 | 含义 |
| --- | --- |
| `image` | BGR 编码的识别切片 |
| `transcription` / `label` | 完整索书号文本 |
| `points` / `bbox` | 文本四点框，已同步旋转和缩放 |
| `scores` | 当前生成器写入 `1` |
| `file` | 形如 `image-000000001_0_90.jpg` 的逻辑文件名 |
| `font` | 实际使用的字体文件名，不含目录 |

字体每 20 条样本切换一个批次；字体对象按路径和字号缓存。`font` 字段用于追踪每条文本使用的字体。

## 验证和排查

查看完整参数：

```powershell
python -m scripts.generate.callnumber_rec_multiline_lmdb --help
```

验证代码语法：

```powershell
python -m py_compile .\text_renderer\corpus\call_number_generator.py .\scripts\generate\callnumber_rec_multiline_lmdb.py
```

如果生成结果中出现行数或长度不符合预期，优先检查：

1. `--patterns` 是否为合法 JSON；
2. 每项是否包含 `lengths`，且长度是否写成整数；
3. 是否误把 `0` 行当作空字符串；
4. 是否使用了正确的 `--generator-strategy`；
5. `--output-scale` 是否过小导致图像细节损失。
