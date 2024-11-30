import argparse
import os
import cv2
import numpy as np
from tqdm import tqdm
import glob
import sdpc
from PIL import Image, ImageDraw
import pickle
import json
from utils import find_blue_polygon, find_blue_polygon2
import math
from shapely.affinity import translate
from shapely.geometry import Polygon,MultiPolygon
'''
注意json中的标注不允许存在不闭合标注
'''
# Mask generation by OTSU algorithm
def whether_has_mask(patch_path,o_cut_x,o_cut_y,json_file,origin_demensions,wsi_name,zoom_value,polygon_list):
   one_patch_coco_json = []
   downsample_factor =  math.pow(zoom_value,patch_level)
   total_x,total_y = origin_demensions
   total_x = total_x/downsample_factor
   total_y = total_y/downsample_factor
   cut_x = o_cut_x/downsample_factor
   cut_y = o_cut_y/downsample_factor
   # 维护一个pylygon的列表
#    polygon_list = find_blue_polygon(json_file,downsample_factor)
   # 对应的patch的polygon
   square = Polygon([(cut_x, cut_y), (cut_x + patch_size, cut_y), (cut_x + patch_size, cut_y + patch_size), (cut_x, cut_y + patch_size)])
   has_mask = False
   mask = np.zeros((patch_size, patch_size), dtype=np.int8)
   for idx,polygon in enumerate(polygon_list):
       if polygon.intersection(square).area > 0:
           has_mask = True
           break
   return has_mask

def get_bg_mask(thumbnail, kernel_size=1):
    hsv = cv2.cvtColor(thumbnail, cv2.COLOR_BGR2HSV)
    _, threshold = cv2.threshold(hsv[:, :, 1], 0, 255, cv2.THRESH_OTSU)
    close_kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    image_close = cv2.morphologyEx(np.array(threshold), cv2.MORPH_CLOSE, close_kernel)
    open_kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    image_open = cv2.morphologyEx(np.array(image_close), cv2.MORPH_OPEN, open_kernel)
    return (image_open / 255.0).astype(np.uint8)




parser = argparse.ArgumentParser(description='Code to patch WSI using .sdpc/svs files')
parser.add_argument('--sdpc_and_sdpl_dir', type=str, default=r'E:\一些代码项目\中山一院11-27debug\wsi', help='path of .sdpc files')
parser.add_argument('--save_dir', type=str, default=r'E:\一些代码项目\中山一院11-27debug\save', help='path to store processed tiles')
parser.add_argument('--patch_size', type=int, default=1024, help='size for processed tiles')
parser.add_argument('--thumbnail_level', type=int, default=4, help='thumbnail level')
parser.add_argument('--patch_level', type=int, default=1, help='level for cutting patches')
parser.add_argument('--overlap_size', type=int, default=0, help='overlap size for processed tiles')
parser.add_argument('--blank_TH', type=float, default=0.8, help='cut patches with blank rate lower than blank_TH')

'''
-- /sdpc_and_sdpl_dir:
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



args = parser.parse_args()
sdpc_and_sdpl_dir = args.sdpc_and_sdpl_dir
thumbnail_level = args.thumbnail_level
save_dir = args.save_dir
patch_size = args.patch_size
patch_level = args.patch_level
overlap_size = args.overlap_size
blank_TH = args.blank_TH
ann = True

if __name__ == '__main__':
    wsi_dir = os.listdir(sdpc_and_sdpl_dir)
    wsi_dirs = [os.path.join(sdpc_and_sdpl_dir, wsi) for wsi in wsi_dir]
    
    for wsi_dir in tqdm(wsi_dirs,colour='green'):
        sdpc_path = glob.glob(os.path.join(wsi_dir, '*.sdpc'))[0]
        sdpl_path = glob.glob(os.path.join(wsi_dir, '*.sdpl'))[0]
        wsi = sdpc.Sdpc(sdpc_path)
        zoom_value = wsi.level_downsamples[1] / wsi.level_downsamples[0]
        origin_dimensions = wsi.level_dimensions[0]
        # 创建wsi对应的json文件
        with open(sdpl_path,"r") as f:
            json_file = json.load(f)
        downsample_factor =  math.pow(zoom_value,patch_level)
        polygon_list = find_blue_polygon(json_file,downsample_factor)
        wsi_name = os.path.basename(sdpc_path).split('.')[0]

        thumbnail = np.array(wsi.read_region((0, 0), thumbnail_level, wsi.level_dimensions[thumbnail_level]))
        # Obtain mask & generate mask image
        black_pixel = np.where((thumbnail[:, :, 0] < 50) & (thumbnail[:, :, 1] < 50) & (thumbnail[:, :, 2] < 50))
        thumbnail[black_pixel] = [255, 255, 255]

        bg_mask = get_bg_mask(thumbnail, kernel_size=5)
        if os.path.exists(os.path.join(f'{save_dir}/{wsi_name}')):
            print(wsi_name+ 'has been precessed')
            continue
        os.makedirs(f'{save_dir}/{wsi_name}/patches', exist_ok=True)
        save_cut_ans_thumbnail = wsi.read_region((0, 0), thumbnail_level, wsi.level_dimensions[thumbnail_level])
        save_cut_ans_thumbnail = Image.fromarray(save_cut_ans_thumbnail).convert('RGB')
        draw_downsample_factor =  math.pow(zoom_value,thumbnail_level)
        draw_polygon_list = find_blue_polygon2(json_file,draw_downsample_factor)
        draw = ImageDraw.Draw(save_cut_ans_thumbnail)
        for polygon in draw_polygon_list:
            draw.polygon(polygon, outline='blue', width=3)
        



        marked_img = thumbnail.copy()

        tile_x = int(patch_size / pow(zoom_value, thumbnail_level - patch_level))
        tile_y = int(patch_size / pow(zoom_value, thumbnail_level - patch_level))

        x_overlap = int(overlap_size / pow(zoom_value, thumbnail_level - patch_level))
        y_overlap = int(overlap_size / pow(zoom_value, thumbnail_level - patch_level))

        thumbnail_x, thumbnail_y = wsi.level_dimensions[thumbnail_level]

        total_num = int(np.floor((thumbnail_x - tile_x) / (tile_x - x_overlap) + 1)) * \
                    int(np.floor((thumbnail_y - tile_y) / (tile_y - y_overlap) + 1))

        with tqdm(total=total_num, ncols=100) as pbar:
            for i in range(int(np.floor((thumbnail_x - tile_x) / (tile_x - x_overlap) + 1))):
                for j in range(int(np.floor((thumbnail_y - tile_y) / (tile_y - y_overlap) + 1))):

                    start_x = int(np.floor(i * (tile_x - x_overlap) / thumbnail_x * bg_mask.shape[1]))
                    start_y = int(np.floor(j * (tile_y - y_overlap) / thumbnail_y * bg_mask.shape[0]))

                    end_x = int(np.ceil((i * (tile_x - x_overlap) + tile_x) / thumbnail_x * bg_mask.shape[1]))
                    end_y = int(np.ceil((j * (tile_y - y_overlap) + tile_y) / thumbnail_y * bg_mask.shape[0]))

                    mask = bg_mask[start_y:end_y, start_x:end_x]

                    if np.sum(mask == 0) / mask.size < blank_TH:
                        cv2.rectangle(marked_img, (end_x, end_y), (start_x, start_y), (255, 0, 0), 2)

                        cut_x = int(start_x * pow(zoom_value, thumbnail_level))  # Coordinate X of layer 0
                        cut_y = int(start_y * pow(zoom_value, thumbnail_level))  # Coordinate Y of layer 0

                        img = wsi.read_region((cut_x, cut_y), patch_level, (patch_size, patch_size))

                        save_patch_path = f'{save_dir}/{wsi_name}/patches/{wsi_name}-patch_level-{str(patch_level)}_X-{cut_x}_Y-{cut_y}.png'

                        if ann:
                            has_mask = whether_has_mask(save_patch_path,cut_x,cut_y,json_file,origin_dimensions,wsi_name,zoom_value,polygon_list)
                            if has_mask == True:
                                img = Image.fromarray(img).convert('RGB')
                                img.save(save_patch_path)
                                real_patch_size = math.pow(zoom_value,patch_level)*patch_size
                                points = [
                                    (cut_x / draw_downsample_factor, cut_y / draw_downsample_factor),
                                    ((cut_x + real_patch_size) / draw_downsample_factor, cut_y / draw_downsample_factor),
                                    ((cut_x + real_patch_size) / draw_downsample_factor, (cut_y + real_patch_size) / draw_downsample_factor),
                                    (cut_x / draw_downsample_factor, (cut_y + real_patch_size) / draw_downsample_factor)
                                ]
                                draw.polygon(points, outline='black', width=2)
                    pbar.update(1)
        save_cut_ans_thumbnail.save(f'{save_dir}/{wsi_name}/{wsi_name}-thumbnail_level-{str(thumbnail_level)}_with_patches.png')
                    



        

