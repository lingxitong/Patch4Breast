import os
from shapely.geometry import Polygon
from PIL import Image
from shapely.geometry import box
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union

def visualize_multi_polygons(multi_polygons):
    fig, ax = plt.subplots()
    ax.set_facecolor('white')

    for polygon in multi_polygons:
        x, y = polygon.exterior.xy
        ax.fill(x, y, alpha=0.5, fc='black', ec='black')

    plt.show()
def polygon_intersects_rectangle(polygon, X, Y, W, H):
    # 创建矩形对象
    rectangle = box(X, Y, X + W, Y + H)
    if polygon.intersects(rectangle):
        return True
    else:
        return False
    

import math

def calculate_distance(point1, point2):
    """
    计算两点之间的欧几里得距离
    :param point1: 第一个点的坐标 (x1, y1)
    :param point2: 第二个点的坐标 (x2, y2)
    :return: 两点之间的距离
    """
    x1, y1 = point1
    x2, y2 = point2
    distance = abs(x1 - x2) + abs(y1 - y2) 
    return distance

def get_polygons(sdpl_file, X, Y, W, H,ROI_Polygon=None):
    oripoints = []
    labels = []
    LabelInfoList = sdpl_file['LabelRoot']['LabelInfoList']
    for element in LabelInfoList:
        if str(element['PointsInfo']["ps"]) != "None":
            Pencolor = element['LabelInfo']["PenColor"]
            if Pencolor == 'Black':
                continue
            points = element['PointsInfo']["ps"]
            new_points = []
            for point in points:
                point1,point2 = point.split(', ')[0],point.split(', ')[1]
                point1 = int(point1)
                point2 = int(point2)
                new = (point1,point2)
                new_points.append(new)
            points = new_points
                
            CurPicRect = element['LabelInfo']["CurPicRect"]
            Ref_x, Ref_y = int(CurPicRect.split(',')[0]), int(CurPicRect.split(',')[1])
            Ref_x = int(Ref_x)
            Ref_y = int(Ref_y)
            left_top = element['LabelInfo']["ImgLeftTopPoint"]
            lt_x,lt_y = int(left_top.split(',')[0]),int(left_top.split(',')[1])
            zoom = element['LabelInfo']["ZoomScale"]
            new_points = []
            for point in points:
                new_point_1 = (point[0] + Ref_x - lt_x) // zoom
                new_point_2 = (point[1] + Ref_y - lt_y) // zoom
                new_point = (new_point_1,new_point_2)
                new_points.append(new_point)
            polygon_points = new_points.copy()
            polygon_points.append(polygon_points[0])
            polygon = Polygon(polygon_points).buffer(0.01)
            if ROI_Polygon  == None:
                if polygon_intersects_rectangle(polygon, X, Y, W, H):
                    oripoints.append(new_points)
                    color = Pencolor
                    labels.append(color)
                else:
                    continue
            else:
                if polygon.intersects(ROI_Polygon):
                    oripoints.append(new_points)
                    color = Pencolor
                    labels.append(color)
                else:
                    continue
    # 得到了在一个ORI内的所有线段的坐标和Pencolor
    return_polygons_part_1 = []
    return_labels_part_1 = []
    for i,(oripoint, label) in enumerate(zip(oripoints, labels)):
        if oripoint[0] == oripoint[-1]:
            if len(oripoint) < 3:
                oripoints.pop(i)
                labels.pop(i)
                continue
            polygon = Polygon(oripoint).buffer(0.01)
            return_polygons_part_1.append(polygon)
            return_labels_part_1.append(label)
            oripoints.pop(i)
            labels.pop(i)
    return_polygons_part_2 = []
    return_labels_part_2 = []
    
    L = len(oripoints)
    # print('L:',L)
    

    while L > 0:
        for oripoint, label in zip(oripoints, labels):
            Target_INDEX = -1
            Target_points = None

            INDEX = oripoints.index(oripoint)
            LABEL = label
            POINT = oripoint[0]
            witch = 'SELF'
            Min_D = 999999999999999
            for other_oripoint,other_label in zip(oripoints,labels):
                OTHER_INDEX = oripoints.index(other_oripoint)
                OTHER_LABEL = other_label
                if OTHER_LABEL != LABEL:
                    continue
                if INDEX == OTHER_INDEX:
                    OTHER_POINT = other_oripoint[-1]
                    D = calculate_distance(POINT, OTHER_POINT)
                    if D < Min_D:
                        Min_D = D
                        witch = 'SELF'
                        Target_INDEX = OTHER_INDEX
                else:
                    OTHER_POINT_START = other_oripoint[0]
                    OTHER_POINT_END = other_oripoint[-1]
                    D_START = calculate_distance(POINT, OTHER_POINT_START)
                    if D_START < Min_D:
                        Min_D = D_START
                        witch = 'start'
                        Target_INDEX = OTHER_INDEX
                        Target_points = other_oripoint
                    D_END = calculate_distance(POINT, OTHER_POINT_END)
                    if D_END < Min_D:
                        Min_D = D_END
                        witch = 'end'
                        Target_INDEX = OTHER_INDEX
                        Target_points = other_oripoint
            if witch == 'SELF':
                oripoint.append(oripoint[0])
                polygon = Polygon(oripoint).buffer(0.01)
                return_polygons_part_2.append(polygon)
                return_labels_part_2.append(LABEL)
                oripoints.pop(Target_INDEX)
                labels.pop(Target_INDEX)
                L -= 1
            if witch == 'start':
                oripoint.reverse()
                new_oripoint = oripoint + Target_points
                polygon = Polygon(oripoint).buffer(0.01)
                if Target_INDEX > INDEX:
                    del oripoints[Target_INDEX]
                    del labels[Target_INDEX]
                    del oripoints[INDEX]
                    del labels[INDEX]
                else:
                    del oripoints[INDEX]
                    del labels[INDEX]
                    del oripoints[Target_INDEX]
                    del labels[Target_INDEX]
                oripoints.append(new_oripoint)
                labels.append(LABEL)
                L -= 1
                
            if witch == 'end':
                new_oripoint = Target_points + oripoint
                polygon = Polygon(oripoint).buffer(0.01)
                if Target_INDEX > INDEX:
                    del oripoints[Target_INDEX]
                    del labels[Target_INDEX]
                    del oripoints[INDEX]
                    del labels[INDEX]
                else:
                    del oripoints[INDEX]
                    del labels[INDEX]
                    del oripoints[Target_INDEX]
                    del labels[Target_INDEX]
                oripoints.append(new_oripoint)
                labels.append(LABEL)
                L -= 1
                
        
    return return_polygons_part_1 + return_polygons_part_2, return_labels_part_1 + return_labels_part_2
                        
        

def get_label(sdpl_polygons,labels,color,x, y,patch_size,patch_level):
    patch_box = box(x, y, x + patch_size*(2**patch_level), y + patch_size*(2**patch_level))
    # 红色大边框的情况
    if color == 1:
        inter_id = []
        area = 0
        for i in range(len(sdpl_polygons)):
            if sdpl_polygons[i].intersects(patch_box):
                inter_id.append(i)
                area += sdpl_polygons[i].intersection(patch_box).area
        if len(inter_id) == 0:
            return 2
        else:
            ratio = area / patch_box.area
            if ratio > 0.5:
                return 1
            else:
                return 2
    elif color == 0:
        inter_id = []
        for i in range(len(sdpl_polygons)):
            if sdpl_polygons[i].intersects(patch_box):
                inter_id.append(i)
        if len(inter_id) == 0:
            return 0
        
        for i in inter_id:
            if labels[i] == 'Blue':
                if sdpl_polygons[i].contains(patch_box):
                    return 0
        area = 0
        red_id = []
        blue_id = []
        for i in inter_id:
            if labels[i] == 'Red':
                if sdpl_polygons[i].intersects(patch_box):
                    red_id.append(i)
                    area += sdpl_polygons[i].intersection(patch_box).area
                area += sdpl_polygons[i].intersection(patch_box).area
            if labels[i] == 'Blue':
                if sdpl_polygons[i].intersects(patch_box):
                    blue_id.append(i)
        for i in blue_id:
            for j in red_id:
                if sdpl_polygons[i].intersects(sdpl_polygons[j]):
                    area -= sdpl_polygons[i].intersection(sdpl_polygons[j]).area

        if area / patch_box.area > 0.5:
            return 2
        if area / patch_box.area <= 0.5:
            return 1        
        # print('不知道label')
def find_black_and_red(sdpl_file):  
    black_list = []
    red_list = []
    LabelInfoList = sdpl_file['LabelRoot']['LabelInfoList']
    for element in LabelInfoList:
        if str(element['PointsInfo']["ps"]) == "None":
            CurPicRect = element['LabelInfo']["CurPicRect"]
            Rect = element['LabelInfo']["Rect"]
            color = element['LabelInfo']["PenColor"]
            Ref_x, Ref_y = int(CurPicRect.split(',')[0]), int(CurPicRect.split(',')[1])
            left_top = element['LabelInfo']["ImgLeftTopPoint"]
            lt_x,lt_y = int(left_top.split(',')[0]),int(left_top.split(',')[1])
            x, y = int(Rect.split(',')[0]), int(Rect.split(',')[1])
            w, h = int(Rect.split(',')[2]), int(Rect.split(',')[3])
            zoom = element['LabelInfo']["ZoomScale"]
            if w == 0 or h == 0 or w//zoom == 0 or h//zoom == 0:
                continue
            read_region_info = [int((Ref_x + x - lt_x) // zoom), int((Ref_y + y - lt_y) // zoom), int(w // zoom), int(h // zoom)]
            if color == 'Black':
                black_list.append(read_region_info)
            elif color == 'Red':
                red_list.append(read_region_info)
    # Remove duplicate elements
    black_list = list(set(tuple(x) for x in black_list))
    red_list = list(set(tuple(x) for x in red_list))
    for black in black_list:
        for red in red_list:
            box_red = box(red[0],red[1],red[0]+red[2],red[1]+red[3])
            box_black = box(black[0],black[1],black[0]+black[2],black[1]+black[3])
            if box_red.intersects(box_black):
                black_list.remove(black)
    
    ROI_list = black_list + red_list
    ROI_color_list = [0]*len(black_list) + [1]*len(red_list)
    print('Black方框标注:',len(black_list))
    print('Red方框标注:',len(red_list))
    return ROI_list, ROI_color_list

def find_big_polygon(sdpl_file):
    black_list = [] 
    black_ROI_index_list = []
    LabelInfoList = sdpl_file['LabelRoot']['LabelInfoList']
    for i,element in enumerate(LabelInfoList):
        color = element['LabelInfo']["PenColor"]
        if color == 'Black':
            CurPicRect = element['LabelInfo']["CurPicRect"]
            Ref_x, Ref_y = int(CurPicRect.split(',')[0]), int(CurPicRect.split(',')[1])
            left_top = element['LabelInfo']["ImgLeftTopPoint"]
            lt_x,lt_y = int(left_top.split(',')[0]),int(left_top.split(',')[1])
            print(Ref_x,Ref_y)
            ps = element['PointsInfo']["ps"]
            zoom = element['LabelInfo']["ZoomScale"]
            print('zoom:',zoom)
            black_points = []
            for points in ps:
                point1,point2 = points.split(', ')[0],points.split(', ')[1]
                point1 = int(point1)
                point2 = int(point2)
                new_points = (point1,point2)
                black_points.append(new_points)
            new_points = []
            for points in black_points:
                new_point_1 = (points[0] + Ref_x - lt_x) // zoom
                new_point_2 = (points[1] + Ref_y - lt_y) // zoom
                new_point = (new_point_1,new_point_2)
                new_points.append(new_point)
            black_points = new_points
            black_list.append(black_points)
            black_ROI_index_list.append(i)


    return black_list,black_ROI_index_list  


def find_blue_polygon(sdpl_file,downsample_factor):
    black_list = [] 
    need_polygons = []
    black_ROI_index_list = []
    LabelInfoList = sdpl_file['LabelRoot']['LabelInfoList']
    for i,element in enumerate(LabelInfoList):
        color = element['LabelInfo']["PenColor"]
        if color == 'Blue':
            CurPicRect = element['LabelInfo']["CurPicRect"]
            Ref_x, Ref_y = int(CurPicRect.split(',')[0]), int(CurPicRect.split(',')[1])
            left_top = element['LabelInfo']["ImgLeftTopPoint"]
            lt_x,lt_y = int(left_top.split(',')[0]),int(left_top.split(',')[1])
            print(Ref_x,Ref_y)
            ps = element['PointsInfo']["ps"]
            zoom = element['LabelInfo']["ZoomScale"]
            print('zoom:',zoom)
            black_points = []
            for points in ps:
                point1,point2 = points.split(', ')[0],points.split(', ')[1]
                point1 = int(point1)
                point2 = int(point2)
                new_points = (point1,point2)
                black_points.append(new_points)
            new_points = []
            for points in black_points:
                new_point_1 = (points[0] + Ref_x - lt_x) // zoom//downsample_factor
                new_point_2 = (points[1] + Ref_y - lt_y) // zoom//downsample_factor
                new_point = (new_point_1,new_point_2)
                new_points.append(new_point)
            black_points = new_points
            black_list.append(black_points)
            black_ROI_index_list.append(i)
            need_polygons.append(Polygon(black_points))


    return need_polygons

def find_blue_polygon2(sdpl_file,downsample_factor):
    black_list = [] 
    need_polygons = []
    black_ROI_index_list = []
    LabelInfoList = sdpl_file['LabelRoot']['LabelInfoList']
    for i,element in enumerate(LabelInfoList):
        color = element['LabelInfo']["PenColor"]
        if color == 'Blue':
            CurPicRect = element['LabelInfo']["CurPicRect"]
            Ref_x, Ref_y = int(CurPicRect.split(',')[0]), int(CurPicRect.split(',')[1])
            left_top = element['LabelInfo']["ImgLeftTopPoint"]
            lt_x,lt_y = int(left_top.split(',')[0]),int(left_top.split(',')[1])
            print(Ref_x,Ref_y)
            ps = element['PointsInfo']["ps"]
            zoom = element['LabelInfo']["ZoomScale"]
            print('zoom:',zoom)
            black_points = []
            for points in ps:
                point1,point2 = points.split(', ')[0],points.split(', ')[1]
                point1 = int(point1)
                point2 = int(point2)
                new_points = (point1,point2)
                black_points.append(new_points)
            new_points = []
            for points in black_points:
                new_point_1 = (points[0] + Ref_x - lt_x) // zoom//downsample_factor
                new_point_2 = (points[1] + Ref_y - lt_y) // zoom//downsample_factor
                new_point = (new_point_1,new_point_2)
                new_points.append(new_point)
            black_points = new_points
            black_list.append(black_points)
            black_ROI_index_list.append(i)
            need_polygons.append(Polygon(black_points))


    return black_list   
            

def refine_black_ROI2Polygon(black_ROI_list):
    L = len(black_ROI_list)
    oripoints = black_ROI_list
    labels = ['Black']*L
    return_polygons_part = []
    return_labels_part = []
    while L > 0:
        for oripoint, label in zip(oripoints, labels):
            Target_INDEX = -1
            Target_points = None

            INDEX = oripoints.index(oripoint)
            LABEL = label
            POINT = oripoint[0]
            witch = 'SELF'
            Min_D = 999999999999999
            for other_oripoint,other_label in zip(oripoints,labels):
                OTHER_INDEX = oripoints.index(other_oripoint)
                OTHER_LABEL = other_label
                if OTHER_LABEL != LABEL:
                    continue
                if INDEX == OTHER_INDEX:
                    OTHER_POINT = other_oripoint[-1]
                    D = calculate_distance(POINT, OTHER_POINT)
                    if D < Min_D:
                        Min_D = D
                        witch = 'SELF'
                        Target_INDEX = OTHER_INDEX
                else:
                    OTHER_POINT_START = other_oripoint[0]
                    OTHER_POINT_END = other_oripoint[-1]
                    D_START = calculate_distance(POINT, OTHER_POINT_START)
                    if D_START < Min_D:
                        Min_D = D_START
                        witch = 'start'
                        Target_INDEX = OTHER_INDEX
                        Target_points = other_oripoint
                    D_END = calculate_distance(POINT, OTHER_POINT_END)
                    if D_END < Min_D:
                        Min_D = D_END
                        witch = 'end'
                        Target_INDEX = OTHER_INDEX
                        Target_points = other_oripoint
            if witch == 'SELF':
                oripoint.append(oripoint[0])
                polygon = Polygon(oripoint).buffer(0.01)
                return_polygons_part.append(polygon)
                return_labels_part.append(LABEL)
                oripoints.pop(Target_INDEX)
                labels.pop(Target_INDEX)
                L -= 1
            if witch == 'start':
                oripoint.reverse()
                new_oripoint = oripoint + Target_points
                polygon = Polygon(oripoint).buffer(0.01)
                if Target_INDEX > INDEX:
                    del oripoints[Target_INDEX]
                    del labels[Target_INDEX]
                    del oripoints[INDEX]
                    del labels[INDEX]
                else:
                    del oripoints[INDEX]
                    del labels[INDEX]
                    del oripoints[Target_INDEX]
                    del labels[Target_INDEX]
                oripoints.append(new_oripoint)
                labels.append(LABEL)
                L -= 1
                
            if witch == 'end':
                new_oripoint = Target_points + oripoint
                polygon = Polygon(oripoint).buffer(0.01)
                if Target_INDEX > INDEX:
                    del oripoints[Target_INDEX]
                    del labels[Target_INDEX]
                    del oripoints[INDEX]
                    del labels[INDEX]
                else:
                    del oripoints[INDEX]
                    del labels[INDEX]
                    del oripoints[Target_INDEX]
                    del labels[Target_INDEX]
                oripoints.append(new_oripoint)
                labels.append(LABEL)
                L -= 1
                
    return return_polygons_part
    

def save_ROIs(wsi,black_or_red_list,dir):
    for i,read_region_info in enumerate(black_or_red_list):
        w,h = read_region_info[2],read_region_info[3]
        level = 0
        x,y = read_region_info[0],read_region_info[1]
        
        if (3000>w>1000) or (3000>h>1000):
            level = 1
            w = w//2
            h = h//2
        elif (w>3000) or (h>3000):
            level = 3
            w = w//8
            h = h//8
            
        ROI = wsi.read_region((x,y),level,(w,h))
        image = Image.fromarray(ROI)
        image = image.convert('RGB')
        
        if not os.path.exists(dir):
            os.makedirs(dir)
        image.save(os.path.join(dir,'ROI'+str(i+1)+'.png'))
        
        

def save_ROIs_black_polygon(wsi,black_box_list,polygon_list,dir):
    for i,(read_region_info,black_polygon) in enumerate(zip(black_box_list,polygon_list)):
        w,h = read_region_info[2],read_region_info[3]
        level = 0
        x,y = read_region_info[0],read_region_info[1]
        
        if (3000>w>1000) or (3000>h>1000):
            level = 1
            w = w//2
            h = h//2
        elif (w>3000) or (h>3000):
            level = 3
            w = w//8
            h = h//8
            
        ROI = wsi.read_region((x,y),level,(w,h))
        image = Image.fromarray(ROI)
        image = image.convert('RGB')
        draw = ImageDraw.Draw(image)
        scaled_polygon = [(int((px - x)//(2**level)), int((py-y)//(2**level))) for px, py in black_polygon.exterior.coords]
        draw.polygon(scaled_polygon, outline="black", width=5)
        if not os.path.exists(dir):
            os.makedirs(dir)
        image.save(os.path.join(dir,'ROI'+str(i+1)+'.png'))
        

from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os
from tqdm import tqdm
def get_font_size(patch_size,patch_level,draw_level):
    draw_patch_size = patch_size*(2**patch_level)//(2**draw_level)
    font = (3/4)*draw_patch_size
    return int(font)
def crop_patches_and_move_to_dir(save_dir, num, sdpl_file, wsi, ROI_box, color, normal_dir, tumor_dir, middle_dir, patch_size, patch_level):
    X, Y, W, H = ROI_box
    draw_level = 0
    if (3000>W>1000) or (3000>H>1000):
        draw_level = 1
        w = W//2
        h = H//2
    elif (W>3000) or (H>3000):
        draw_level = 3
        w = W//8
        h = H//8
    else:
        draw_level = 0
        w = W
        h = H
    font_size = get_font_size(patch_size,patch_level,draw_level)
    # 构建polygons
    sdpl_polygons, labels = get_polygons(sdpl_file, X, Y, W, H)
    ROI = wsi.read_region((X, Y), draw_level, (w, h))
    ROI = Image.fromarray(np.array(ROI))  # 将 numpy.ndarray 转换为 PIL.Image
    draw = ImageDraw.Draw(ROI)
    font = ImageFont.truetype("arial.ttf", font_size)  # 使用更大的字体

    for x in tqdm(range(X, X + W, patch_size * (2 ** patch_level))):
        for y in range(Y, Y + H, patch_size*(2**patch_level)):
            patch = wsi.read_region((x, y), patch_level, (patch_size, patch_size))
            patch = Image.fromarray(np.array(patch))  # 将 numpy.ndarray 转换为 PIL.Image
            label = get_label(sdpl_polygons, labels, color, x, y, patch_size, patch_level)
            label_text = str(label)
            
            # 计算标签位置，放置在patch的中心
            text_width, text_height = draw.textsize(label_text, font=font)
            text_x = (int(x - X) + (patch_size - text_width)*(2**patch_level) // 2)//(2**draw_level)
            text_y = (int(y - Y) + (patch_size - text_height)*(2**patch_level)//2)//(2**draw_level)
            if label == 0:
                fill = 'black'
            elif label == 1:
                fill = 'blue'
            else:
                fill = 'red'
            draw.text((text_x, text_y), label_text, fill, font=font)  # 在ROI上绘制标签
            
            # # 绘制虚线边界
            # for i in range(x - X, x - X + patch_size*(2**patch_level), 10):
            #     draw.line([(i, y - Y), (i + 5, y - Y)], fill="black")
            #     draw.line([(i, y - Y + patch_size), (i + 5, y - Y + patch_size)], fill="black")
            # for i in range(y - Y, y - Y + patch_size, 10):
            #     draw.line([(x - X, i), (x - X, i + 5)], fill="black")
            #     draw.line([(x - X + patch_size, i), (x - X + patch_size, i + 5)], fill="black")

            if label == 0:
                patch.save(os.path.join(normal_dir, 'normal_' + f'{x}_{y}.png'))  # tumor占比0%
            elif label == 1:
                patch.save(os.path.join(middle_dir, 'middle_' + f'{x}_{y}.png'))  # tumor占比0-50%
            elif label == 2:
                patch.save(os.path.join(tumor_dir, 'tumor_' + f'{x}_{y}.png'))  # tumor占比50%以上

    ROI.save(os.path.join(save_dir, f'ROI-{num+1}_with_labels.png'))

def judge_patch_in_polygon(polygon, x, y, patch_size, patch_level):
    patch_box = box(x, y, x + patch_size*(2**patch_level), y + patch_size*(2**patch_level))
    if polygon.intersects(patch_box):
        return True
    else:
        return False
                
            
            
def crop_patches_and_move_to_dir_black_polygon(save_dir, num, sdpl_file, wsi, ROI_box,ROI_Polygon, color, normal_dir, tumor_dir, middle_dir, patch_size, patch_level):
    X, Y, W, H = ROI_box
    # 构建polygons
    draw_level = 0
    if (3000>W>1000) or (3000>H>1000):
        draw_level = 1
        w = W//2
        h = H//2
    elif (W>3000) or (H>3000):
        draw_level = 3
        w = W//8
        h = H//8
    font_size = get_font_size(patch_size,patch_level,draw_level)
    sdpl_polygons, labels = get_polygons(sdpl_file, X, Y, W, H,ROI_Polygon)
    ROI = wsi.read_region((X, Y), draw_level, (w, h))
    ROI = Image.fromarray(np.array(ROI))  # 将 numpy.ndarray 转换为 PIL.Image
    draw = ImageDraw.Draw(ROI)
    font = ImageFont.truetype("arial.ttf", font_size)  # 使用更大的字体
    scaled_polygon = [(int((px - X)//(2**draw_level)), int((py-Y)//(2**draw_level))) for px, py in ROI_Polygon.exterior.coords]
    draw.polygon(scaled_polygon, outline="black", width=5)

    for x in tqdm(range(X, X + W, patch_size * (2 ** patch_level))):
        for y in range(Y, Y + H, patch_size):
            corss = judge_patch_in_polygon(ROI_Polygon, x, y, patch_size, patch_level)
            if not corss:
                continue
            patch = wsi.read_region((x, y), patch_level, (patch_size, patch_size))
            patch = Image.fromarray(np.array(patch))  # 将 numpy.ndarray 转换为 PIL.Image
            label = get_label(sdpl_polygons, labels, color, x, y, patch_size, patch_level)
            label_text = str(label)

            # 计算标签位置，放置在patch的中心
            text_width, text_height = draw.textsize(label_text, font=font)
            text_x = (int(x - X) + (patch_size - text_width)*(2**patch_level) // 2)//(2**draw_level)
            text_y = (int(y - Y) + (patch_size - text_height)*(2**patch_level)//2)//(2**draw_level)
            if label == 0:
                fill = 'black'
            elif label == 1:
                fill = 'blue'
            else:
                fill = 'red'
            draw.text((text_x, text_y), label_text, fill, font=font)  # 在ROI上绘制标签

            # 绘制虚线边界

            if label == 0:
                patch.save(os.path.join(normal_dir, 'normal_' + f'{x}_{y}.png'))  # tumor占比0%
            elif label == 1:
                patch.save(os.path.join(middle_dir, 'middle_' + f'{x}_{y}.png'))  # tumor占比0-50%
            elif label == 2:
                patch.save(os.path.join(tumor_dir, 'tumor_' + f'{x}_{y}.png'))  # tumor占比50-100%

    ROI.save(os.path.join(save_dir, f'ROI-{num+1}_with_labels.png'))
