import random

import cv2
# points, coord_dst, size_dst = compute_perspective_coord_map(points_, check=check)
# matrix = cv2.getPerspectiveTransform(np.float32(points), np.float32(coord_dst))
# piece = cv2.warpPerspective(image, matrix, size_dst)  # (h, w))

import cv2
import numpy as np
import matplotlib.pyplot as plt
# 1.ori_rec  2.dst_pers

def tp2fp(x1,y1,x2,y2):
    return [[x1,y1],[x2,y1],[x2,y2],[x1,y2]]


pers_couples  = [
    [[[86, 54], [330, 54], [330, 204], [86, 204]],[[35,64],[310,56],[386,205],[113,210]]],
    [[[119, 176], [396, 176], [396, 275], [119, 275]],[[109,173],[433,171],[505,275],[185,277]]],
    [[[153, 114], [609, 114], [609, 214], [153, 214]],[[45,105],[636,131],[682,223],[98,197]]],
    [[[56, 119], [318, 119], [318, 285], [56, 285]],[[40,161],[312,52],[336,236],[62,347]]],
    [[[11, 99], [460, 99], [460, 282], [11, 282]],[[35,138],[454,72],[446,240],[18,308]]],
    [[[60, 95], [493, 95], [493, 282], [60, 282]],[[46,135],[475,62],[487,246],[57,318]]],
    [[[66, 92], [670, 92], [670, 310], [66, 310]],[[76,97],[610,146],[647,327],[115,270]]],
    [[[127, 36], [643, 36], [643, 148], [127, 148]],[[140,91],[727,43],[693,143],[105,190]]],
    [[[288, 87], [656, 87], [656, 218], [288, 218]],[[380,87],[883,80],[793,211],[292,215]]],
    [[[124, 20], [804, 20], [804, 204], [124, 204]],[[239,103],[866,36],[812,180],[188,240]]],
    [[[37,46],[366,45],[366,149],[37,149]],[[3,79],[360,2],[387,113],[32,190]]],
    [[[178,221],[744,221],[744,377],[178,377]],[[200,256],[777,196],[700,334],[117,392]]]
]
def sam_couples(pers_couples,rand_bin=2):
    ori_couple = random.choice(pers_couples)
    couple_arr = np.array(ori_couple)
    if rand_bin == 0:
        return ori_couple
    rand_arr = np.random.randint(-rand_bin,rand_bin,(2,4,2))
    couple_arr_mod = couple_arr + rand_arr
    couple_list = couple_arr_mod.tolist()
    random.shuffle(couple_list)
    return couple_list




if __name__ == '__main__':
    print(tp2fp(86,54,330,204))
    img = cv2.imread(r'C:\Users\AlvaSystmsTraing001\Desktop\front_view.png')
    h,w = img.shape[:2]
    sam_couple = sam_couples(pers_couples,2)

    ranc,pers  = sam_couple

    matrix = cv2.getPerspectiveTransform(np.array(ranc,np.float32), np.array(pers,np.float32))
    piece = cv2.warpPerspective(img, matrix, (h,w))  # (h, w))
    plt.imshow(piece)
    plt.show()
    print(sam_couples(pers_couples))
