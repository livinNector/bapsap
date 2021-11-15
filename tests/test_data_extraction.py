from bapsap.tubestate import TubeState
from bapsap.image.data_extraction import  extract_ball_color_codes

import os

test_imgs_dir = os.path.join(os.path.dirname(__file__),"test_images")
print(test_imgs_dir)
for image in os.listdir(test_imgs_dir):
    img_path = os.path.join(test_imgs_dir,image)
    print(img_path)
    assert os.path.exists(img_path)
    color_code,tube_coords = extract_ball_color_codes(img_path,4,True,plot=True)
    assert color_code,tube_coords
