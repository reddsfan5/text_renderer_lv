# 项目更新说明

按时间倒序记录会影响数据生成行为或使用方式的迭代，便于追踪设计取舍。  
功能说明以专题文档为准；本文件侧重「为什么改、改了什么、怎么用」。

---

## 2026-09-04 — column 模式文字逆时针旋转 90°

### 动机

按列书写的拉丁索书号字形需侧躺；原先逐字正立往下叠，未旋转。

### 行为变化

- `draw_fitted_multicolumn_text`：每条 line 先横排绘制，再整体**逆时针**转 90° 后作为一列粘贴

### 涉及文件

- `text_renderer/renderers/callnumber.py`
- `docs/call_number_rec_multiline_lmdb.md`

---

## 2026-09-04 — 索书号行尾禁止非封闭符号

### 动机

索书号行不能以 `.`、`-`、`=`、`(` 等未封闭符号结尾；`learned` 策略里存在模板 `LDDD.`，且小数点插入可能落在末位。

### 行为变化

- 模板 `LDDD.` 改为 `LD.DD`
- 小数点仅插入数字段中部（至少前后各留一位）
- 生成后统一 `forbid_open_line_ending` 兜底

### 涉及文件

- `text_renderer/corpus/call_number_generator.py`
- `text_renderer/tests/test_callnumber_open_ending.py`

---

## 2026-09-04 — 索书号多行：分方向布局 + 高宽比护栏

专题文档（完整）：[call_number_rec_multiline_lmdb.md](call_number_rec_multiline_lmdb.md)

### 动机

识别切片高宽比过大会损害可用性。原先用「事后按 h/w 强制旋转」纠偏，心智模型重，且与「按行 / 按列书写」产品语义纠缠。

改为：

- **主路径**：分方向默认 `_DEFAULT_PATTERNS_ROW` / `_DEFAULT_PATTERNS_COLUMN`（改结构 = 改表）
- **辅路径**：`--max-aspect-ratio` 护自定义 patterns + 渲染后紧框复检
- **按列**：`draw_fitted_multicolumn_text` 横排后逆时针转 90° 成列（见同日后续条目）

### 设计取舍

| 方案 | 结论 |
| --- | --- |
| 分方向两套默认表 | **采用**（日常负担最低） |
| 统一大表 + 启动自动过滤 | 不用作默认（row/column 合格集合不对称；表内会出现采不到的项） |
| 仅渲染后丢弃 | 只作兜底，不作默认策略 |

用户心智：`--write-direction` +（可选）`--max-aspect-ratio`；日常改分布只改对应默认表。

### 行为变化

- 新增 `--write-direction {row,column}`（默认 `row`）
- 新增 `--max-aspect-ratio`（默认 `1.2`）
- `--patterns` 默认 `None` → 未传则注入分方向默认表
- `--rotation-prob` 默认改为 `0`；旋转后仍须达标才允许
- **删除** `--rotation-aspect-ratio-threshold-l/r` 及高瘦必转

`lengths` 估算：

- `row`：`h/w ≈ n_rows / max_line_len`
- `column`：`h/w ≈ max_col_len / n_cols`

例：`[1,1,1]` 在 row 拒、column 过；`[3,3,0]` 在 row 过、column 拒（阈值 1.2）。

### 涉及文件

- `text_renderer/pipelines/callnumber_multiline_lmdb.py`
- `text_renderer/renderers/callnumber.py`
- `docs/call_number_rec_multiline_lmdb.md`、`docs/commands.md`
- `text_renderer/tests/test_callnumber_multiline_aspect.py`

### 使用

```powershell
python -m text_renderer.cli callnumber-multiline --write-direction row --num 10000
python -m text_renderer.cli callnumber-multiline --write-direction column --num 10000

# 自定义：row 下超限项会被滤掉
python -m text_renderer.cli callnumber-multiline `
  --write-direction row --max-aspect-ratio 1.2 `
  --patterns '[{"lengths":[3,3,0],"weight":2},{"lengths":[1,1,1],"weight":1}]'
```

### 非目标

不改 generator 语法职责；不做自动布局搜索；detection 流水线不同步本改造。

---

## 2026-09-03 — 文字颜色可配置 + LMDB full 可视化

### 行为变化

- YAML `text_color_strategy`：`fixed` / `adaptive`（别名 `safe`）/ `gray` 真正生效（不再被 `text_color_mode: gray` 绕过）
- `lmdbs.lmdbs.visualize` 支持 `full|samples|ori-data|show`；生成结束后可走 `visualize_full` 导出原图与标注预览

### 涉及文件

- `text_renderer/render.py`、`text_renderer/config/yaml_config.py`
- `configs/multilingual.yaml`、`configs/korean_vertical.yaml`
- `lmdbs/lmdbs/visualize.py`、`text_renderer/cli.py`
