import cv2
import numpy as np
from lib.image_generator import *

print("loading image generator...")
input_width = 448
input_height = 448
item_path = "./items"
background_path = "./backgrounds"
generator = ImageGenerator(item_path, background_path)

# generate random sample
x, t = generator.generate_samples(
    n_samples=64,
    n_items=3,
    crop_width=input_width,
    crop_height=input_height,
    min_item_scale=1,
    max_item_scale=3,
    rand_angle=15,
    minimum_crop=0.8,
    delta_hue=0.01,
    delta_sat_scale=0.5,
    delta_val_scale=0.5
)

for i, image in enumerate(x):
    print(t[i])
    cv2.imshow("w", np.transpose(image, (1, 2, 0)))
    cv2.waitKey(300)
