import cv2
import numpy as np
import matplotlib.pyplot as plt
img1_path = r'E:\lxd\text_renderer\example_data\effect_layout_image\color_image\images/000000190.jpg'
img2_path = r'E:\lxd\text_renderer\example_data\effect_layout_image\color_image\images/000000183.jpg'
img1 = cv2.imread(img1_path)
img2 = cv2.imread(img2_path)

img=cv2.hconcat([img1,img2])
plt.imshow(img)
plt.show()