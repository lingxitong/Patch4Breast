import sdpc
import argparse
import os
import cv2
import numpy as np
from tqdm import tqdm
import glob
import json
import math
from shapely.affinity import translate
from shapely.geometry import Polygon,MultiPolygon
from utils import save_ROIs,crop_patches_and_move_to_dir,find_black_and_red,find_big_polygon,refine_black_ROI2Polygon,save_ROIs_black_polygon, crop_patches_and_move_to_dir_black_polygon
'''
-- /data_dir:
       /dir1
          /12345.sdpc
          /12345.sdpl
        /dir2
            /23456.sdpc
            /23456.sdpl
        /dir3
            /34567.sdpc
            /34567.sdpl
'''

parser = argparse.ArgumentParser(description='Code to patch WSI using .sdpc files')
parser.add_argument('--data_dir', type=str, default='/Path/to/your/data_dir', help='path of .sdpc files')
parser.add_argument('--save_dir', type=str, default='/Path/to/your/save_dir', help='path to store processed tiles')
parser.add_argument('--patch_size', type=int, default=128, help='size for processed tiles')
parser.add_argument('--patch_level', type=int, default=1, help='level for cutting patches')

args = parser.parse_args()
data_dir = args.data_dir
save_dir = args.save_dir
patch_size = args.patch_size
patch_level = args.patch_level

if __name__ == '__main__':
    sub_folder_list = os.listdir(data_dir)
    sub_folder_list_paths = [os.path.join(data_dir,sub_folder) for sub_folder in sub_folder_list]
    
    for sub_folder_path in sub_folder_list_paths:
        sdpc_path = glob.glob(os.path.join(sub_folder_path, '*.sdpc'))
        sdpl_path = glob.glob(os.path.join(sub_folder_path, '*.sdpl'))
        if len(sdpc_path) == 0 or len(sdpl_path) == 0:
            continue
        sdpc_path = sdpc_path[0]
        sdpl_path = sdpl_path[0]
        wsi = sdpc.Sdpc(sdpc_path)
        basename = os.path.basename(sub_folder_path)
        os.makedirs(os.path.join(save_dir, basename), exist_ok=True)
        now_wsi_save_dir = os.path.join(save_dir, basename)
        print('processing-----------------------', basename)
        print('切图参数：')
        print('patch_size:', patch_size)
        if patch_level == 0:
            mag = '40X'
        elif patch_level == 1:
            mag = '20X'
        elif patch_level == 2:
            mag = '10X'
        else:
            mag = None
        print('切图倍率:', mag)      
        # 开始处理
        with open(sdpl_path, 'r') as f:
            sdpl_file = json.load(f)
        ROI_list,ROI_color_list = find_black_and_red(sdpl_file)
        if len(ROI_list) != 0:
            save_ROIs(wsi,ROI_list,now_wsi_save_dir)
            for i,ROI_box in enumerate(ROI_list):
                if os.path.exists(os.path.join(now_wsi_save_dir, f'ROI-{i+1}_with_labels.png')):
                    print('ROI:', i+1, 'already processed')
                    continue
                print('processing ROI:', i+1)
                color = ROI_color_list[i]
                os.makedirs(os.path.join(now_wsi_save_dir, f'ROI-{i+1}'), exist_ok=True)
                now_ROI_patch_save_dir = os.path.join(now_wsi_save_dir, f'ROI-{i+1}')
                normal_dir = os.path.join(now_ROI_patch_save_dir, 'normal')
                os.makedirs(normal_dir, exist_ok=True)
                tumor_dir = os.path.join(now_ROI_patch_save_dir, 'tumor')
                os.makedirs(tumor_dir, exist_ok=True)
                middle_dir = os.path.join(now_ROI_patch_save_dir, 'middle')
                os.makedirs(middle_dir, exist_ok=True)
                crop_patches_and_move_to_dir(now_wsi_save_dir,i,sdpl_file, wsi, ROI_box,color, normal_dir, tumor_dir, middle_dir, patch_size, patch_level)
        else:
            black_ROI_list,black_ROI_index_list = find_big_polygon(sdpl_file)
            black_ROI_Polygon_list = refine_black_ROI2Polygon(black_ROI_list)
            print('Black不规则多边形标注:',len(black_ROI_Polygon_list))
            new_black_ROI_Polygon_list = []
            for i,black_ROI_Polygon in enumerate(black_ROI_Polygon_list):
                if isinstance(black_ROI_Polygon,MultiPolygon):
                    multipolygon_list = black_ROI_Polygon.geoms
                    MaxD = 0
                    MaxPolygon = None
                    for polygon in multipolygon_list:
                        bounds = polygon.bounds
                        xmin,ymin,xmax,ymax = bounds
                        w = xmax - xmin
                        h = ymax - ymin
                        D = w+h
                        if D >= MaxD:
                            MaxD = D
                            MaxPolygon = polygon
                    new_black_ROI_Polygon_list.append(MaxPolygon)
                else:
                    new_black_ROI_Polygon_list.append(black_ROI_Polygon)
            black_ROI_Polygon_list = new_black_ROI_Polygon_list
                    
            black_ROI_box_list = [box.bounds for box in black_ROI_Polygon_list]
            black_ROI_box_transformed = [(int(minx), int(miny), int(maxx - minx), int(maxy - miny)) for (minx, miny, maxx, maxy) in black_ROI_box_list]
            save_ROIs_black_polygon(wsi,black_ROI_box_transformed,black_ROI_Polygon_list,now_wsi_save_dir)
            
            for i,(black_ROI_box,black_ROI_Polygon) in enumerate(zip(black_ROI_box_transformed,black_ROI_Polygon_list)):
                print('processing black ROI:', i+1)
                if os.path.exists(os.path.join(now_wsi_save_dir, f'ROI-{i+1}_with_labels.png')):
                    print('ROI:', i+1, 'already processed')
                    continue
                os.makedirs(os.path.join(now_wsi_save_dir, f'Black_Polygon_ROI-{i+1}'), exist_ok=True)
                now_ROI_patch_save_dir = os.path.join(now_wsi_save_dir, f'Black_Polygon_ROI-{i+1}')
                normal_dir = os.path.join(now_ROI_patch_save_dir, 'normal')
                os.makedirs(normal_dir, exist_ok=True)
                tumor_dir = os.path.join(now_ROI_patch_save_dir, 'tumor')
                os.makedirs(tumor_dir, exist_ok=True)
                middle_dir = os.path.join(now_ROI_patch_save_dir, 'middle')
                os.makedirs(middle_dir, exist_ok=True)
                color = 'black'
                color = 0
                crop_patches_and_move_to_dir_black_polygon(now_wsi_save_dir,i,sdpl_file, wsi, black_ROI_box,black_ROI_Polygon,color, normal_dir, tumor_dir, middle_dir, patch_size, patch_level)



            
