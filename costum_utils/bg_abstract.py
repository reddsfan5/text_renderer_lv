import cv2
import os
from os import path as osp


def bg_abs(ori_bg_dir, dst_bg_dir):
    bg_img_paths = [osp.join(ori_bg_dir, basename) for basename in os.listdir(ori_bg_dir) if basename.endswith('jpg')]
    for bp in bg_img_paths:
        print(bp)
        bg_arr = cv2.imread(bp)
        h, w = bg_arr.shape[:2]
        bg_arr = cv2.resize(bg_arr,(1000,int(1000*w/h)))
        dst_bg_arr = bg_arr[0:800, 0:160, :] if h > w else bg_arr[0:160, 0:800, :]
        if dst_bg_arr.shape[0] > dst_bg_arr.shape[1]:
            dst_bg_arr = cv2.rotate(dst_bg_arr, cv2.ROTATE_90_CLOCKWISE)
        dst_bg_arr = cv2.GaussianBlur(dst_bg_arr,(11,11),9)
        cv2.imwrite(osp.join(dst_bg_dir, osp.basename(bp)), dst_bg_arr)
        dst_bg_arr = dst_bg_arr[..., ::-1]
        cv2.imwrite(osp.join(dst_bg_dir, 'counterpart_' + osp.basename(bp)), dst_bg_arr)

if __name__ == '__main__':
    ori = r'E:\lxd_dataset\Textures\PVC'
    dst = r'E:\lxd\OCR_project\OCR_SOURCE\bg'
    bg_abs(ori, dst)
