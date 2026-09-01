# 脚本目录

辅助命令按职责分组，并统一从仓库根目录以模块方式运行：

```powershell
python -m scripts.generate.callnumber_rec_multiline_lmdb --help
python -m scripts.corpus.build_korean_coverage_corpus --help
python -m scripts.inspect.lmdb_char_statistics --help
```

- `generate/`：专用数据生成入口。
- `corpus/`：语料构建和补全。
- `inspect/`：数据统计与可视化。
- `experiments/`：调试和人工实验代码。
