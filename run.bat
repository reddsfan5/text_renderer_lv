cd D:\lxd_code\text_renderer_lv
call conda activate base

python main.py --config .\example_data/effect_layout_example.py --dataset lmdb --num_processes 1 --log_period 2
@pause
