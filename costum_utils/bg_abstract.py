import cv2
import os
from os import path as osp
def bg_abs(ori_bg_dir,dst_bg_dir):
    bg_img_paths = [osp.join(ori_bg_dir,basename) for basename in os.listdir(ori_bg_dir) if basename.endswith('jpg')]
    for bp in bg_img_paths:
        print(bp)
        bg_arr = cv2.imread(bp)
        h,w = bg_arr.shape[:2]

        dst_bg_arr = bg_arr[0:800,0:160,:] if h>w else bg_arr[0:160,0:800,:]

        cv2.imwrite(osp.join(dst_bg_dir,osp.basename(bp)),dst_bg_arr)
if __name__ == '__main__':
    ori = r'E:\lxd\text_renderer_lv\example_data\bg_ori'
    dst = r'E:\lxd\text_renderer_lv\example_data\bg'
    bg_abs(ori,dst)


