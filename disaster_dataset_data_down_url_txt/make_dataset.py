import sys
sys.path.append('dataset_make__tools/')
from dataset import *
import os
from pathlib import Path
disaster_map = {'flooding': 0,  'Earthquake': 0,  'Fires': 0, 'Hurricane':0,  
             'Explosions':0, 'Volcan': 0, 'Landslide': 0, 'Oil':0,
             "dfc25_track2_trainval":0,'xview':0,"Cyclone":0,"Storm":0,
             'Typhoon':0, 'Tornado':0,"war":0,"Tsunami":0,"sar":0,
             "CRASAR-U-DROIDs":0,"flooding_a":0,'nodisaster1':0}  

disaster_img =  ['image_generation','image_test_data','image_video_data']

base_path = 'save_path'

data_map_config,data_area_config,data_img_config,data_config_labels  = get_map_img_config(base_path, 
                                                      disaster_map,disaster_img ,update_force = True)
disaster_dis_n = get_diasater_name(data_map_config,use_disasters= None)

print('num: ',len(disaster_dis_n))

def worker(disaster_name,N = 2,zooms = [17,18,19,20],layer_names= None):
        global data_map_config,data_config_labels
        disaster,dis_n = disaster_name[0],disaster_name[1]
        save_path = f"{base_path}/disasters_db_dataset/{disaster}/{dis_n}/"
        img_path = f'{save_path}/image'
        mask_path = f'{save_path}/mask'
    
        if Path(img_path).exists(): return 
        print('--------------: ',save_path) 
 
        layer_urls, layer_label_urls,area_config = get_url_area_mask(base_path,disaster,dis_n,data_map_config,data_config_labels,mask_index=True)
        data_index_lists = get_data_index_data_type(layer_urls,area_config,zooms=zooms,select_threshold=0,
                                                        select_num_check=5,data_type_use=["train",'test'],show =False,step=1)
       
        iter_data =        get_img_by_mask(data_index_lists,base_path, disaster,dis_n,layer_urls,layer_label_urls,mask_index=True,N=N,show=False,data_types=["train",'test'])
    
        for img,mask,(z,x,y),area,data_type,(disaster,dis_n,layer_name_zoom) in iter_data:
            mask = rgb_to_numeric_mask(mask)
            name = f'{layer_name_zoom}_{x}_{y}.png'
            if Path(f'{img_path}/{name}').exists(): continue
            os.makedirs(img_path ,exist_ok=True)
            os.makedirs(mask_path ,exist_ok=True)
            cv2.imwrite(f'{img_path}/{name}',cv2.cvtColor(img,cv2.COLOR_RGB2BGR))
            cv2.imwrite(f'{mask_path}/{name}',mask)


from multiprocessing.pool import ThreadPool
with ThreadPool(processes=1) as pool:
   pool.map(worker, disaster_dis_n[:])
