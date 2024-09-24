# Import libraries
import numpy as np, cv2, os
from PIL import Image; from datetime import datetime
from glob import glob; from tqdm import tqdm
from matplotlib import pyplot as plt
from st_server_libs.segmodel import Seg_model

s_model = Seg_model()
data_path = "/mnt/data/webtoon/Manga109_released_2023_12_07/images"
save_path = "./manga512"

def make_data(data_path, save_path, model, resize = None):
    
    train_path = f"{save_path}/train"
    valid_path = f"{save_path}/val_source"
    test_path  = f"{save_path}/visual_test_source"
    os.makedirs(train_path, exist_ok = True); os.makedirs(valid_path, exist_ok = True); os.makedirs(test_path, exist_ok = True)
    im_paths = glob(f"{data_path}/*/*.jpg")
    for idx, im_path in tqdm(enumerate(im_paths)):
        # if idx == 3000: break
        dname = os.path.basename(os.path.dirname(im_path))
        fname = os.path.splitext(os.path.basename(im_path))[0]
        im = cv2.imread(im_path)
        try: overlayed, segments = s_model.segpanel(im)
        except: continue
        if len(segments) > 1:
            for i, segment in enumerate(segments):
                if ((idx * i) % 1200 == 0) and (i != 0) and (idx != 0): 
                    cv2.imwrite(filename = f"{valid_path}/{dname}_{fname}_{i}.jpg", img = cv2.resize(np.array(segment), resize)) if resize else cv2.imwrite(filename = f"{save_path}/{dname}_{fname}_{i}.jpg", img = np.array(segment))      
                elif ((idx * i) % 3000 == 0) and (i != 0) and (idx != 0): 
                    cv2.imwrite(filename = f"{test_path}/{dname}_{fname}_{i}.jpg", img = cv2.resize(np.array(segment), resize)) if resize else cv2.imwrite(filename = f"{save_path}/{dname}_{fname}_{i}.jpg", img = np.array(segment))      
                else: cv2.imwrite(filename = f"{train_path}/{dname}_{fname}_{i}.jpg", img = cv2.resize(np.array(segment), resize)) if resize else cv2.imwrite(filename = f"{save_path}/{dname}_{fname}_{i}.jpg", img = np.array(segment))
                
make_data(data_path = data_path, save_path = save_path, model = s_model, resize = (512, 512))
