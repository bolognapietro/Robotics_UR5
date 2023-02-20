#!/usr/bin/env python

# Standard library modules
import math
from collections import Counter
from copy import deepcopy

import cv2
import numpy as np
from PIL import Image

# Local imports
import zed_params
import geometric_utils

def add_background(img: np.ndarray, width: int, height: int, color: tuple = (0,0,0,255)) -> np.ndarray:
    """
    Increases the size of the image by adding some background as contours

    Input parameters:
    - ``img`` the array of the image to be processed
    - ``width`` new width
    - ``height`` new height
    - ``color`` (optional, ``default = (0,0,0,255)``) background color

    Output parameter (``np.ndarray``):
    - ``img`` (``np.ndarray``):
        - the processed image
    """

    """
    Increases the size of the image by adding some background as contours.

    Args:
        img (np.ndarray): Image to be processed.
        width (int): New width.
        height (int): New height.
        color (tuple, optional): Background color. Defaults to (0,0,0255). 

    Returns:
        np.ndarray: Processed image.
    """

    image = img.copy()

    image = Image.fromarray(image)
    img_w, img_h = image.size

    background = Image.new('RGBA', (width, height), color)
    bg_w, bg_h = background.size

    offset = ((bg_w - img_w) // 2, (bg_h - img_h) // 2)
    background.paste(image, offset)

    background = np.asarray(background)

    return background

def all_black(img: np.ndarray) -> bool:

    image = img.copy()
    
    counter = 0

    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            if image[y][x].tolist() != [0,0,0]:
                counter = counter + 1
            
    return ((100 * counter) / (image.shape[0]*image.shape[1])) < 30

def main_color(img: np.ndarray, noblack: bool = False, notable: bool = False) -> tuple:
    """
    Retrieves the main color of the image

    Input parameters:
    - ``img`` the array of the image to be processed
    - ``noblack`` (optional, ``default = False``) if true discards black color
    - ``notable`` (optional, ``default = False``) if true discards table color

    Output parameter (``tuple``):
    - ``color`` (``tuple``):
        - the retrieved color
    """

    image = img.copy()

    image = Image.fromarray(image)
    image = image.resize((100, 100), Image.Resampling.LANCZOS)
    pixels = image.load()
    
    color_counts = Counter(pixels[i, j] for i in range(100) for j in range(100))

    index = 1

    while True:
        color = color_counts.most_common(index)[-1][0]

        if color != (0,0,0) or not noblack: 
            
            if notable and math.dist(color,(138,138,138)) > 20:
                return color
        
        index = index + 1

def one_color(img: np.ndarray,color: tuple = None) -> np.ndarray:
    """
    Fills the image with one color (excluding the black background)

    Input parameters:
    - ``img`` the array of the image to be processed
    - ``color`` fill color

    Output parameter (``np.ndarray``):
    - ``img`` (``np.ndarray``):
        - the processed image
    """

    image = img.copy()

    if color == None:
        color = main_color(image,True, True)

    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            
            pixel = image[y][x].tolist()

            if pixel != [0,0,0]:
                image[y][x] = color

    return image

def crop_img(img: np.ndarray, points: list = []) -> tuple:
    """
    Crops the image so as to remove any black borders

    Input parameters:
    - ``img`` the array of the image to be processed
    - ``points`` (optional, ``default = []``) list of the zed camera points of the bounding box of the object

    Output parameter (``tuple``):
    - ``img`` (``np.ndarray``):
        - the processed image
    - ``points`` (``list``):
        - the list of the processed points array
    """

    image = img.copy()

    #crop image
    cropped_height, cropped_width = image.shape[:2]

    #crop height top
    cropped_height_top = cropped_height

    for y in range(image.shape[0]):
        row = image[y]
        row = row.tolist()

        if all([rgb == [0,0,0] for rgb in row]):
            cropped_height_top = cropped_height_top - 1
        else:
            break

    cropped_height_top = abs(cropped_height - cropped_height_top)

    #crop height bottom
    cropped_height_bottom = cropped_height

    for y in range(image.shape[0]-1,-1,-1):
        row = image[y]
        row = row.tolist()

        if all([rgb == [0,0,0] for rgb in row]):
            cropped_height_bottom = cropped_height_bottom - 1
        else:
            break

    cropped_height_bottom = abs(cropped_height - cropped_height_bottom)

    #crop width left
    cropped_width_left = cropped_width

    for x in range(image.shape[1]):

        column = [image[y][x].tolist() for y in range(image.shape[0])]

        if all([rgb == [0,0,0] for rgb in column]):
            cropped_width_left = cropped_width_left - 1
        else:
            break

    cropped_width_left = abs(cropped_width - cropped_width_left)

    #crop width right
    cropped_width_right = cropped_width

    for x in range(image.shape[1]-1,-1,-1):

        column = [image[y][x].tolist() for y in range(image.shape[0])]

        if all([rgb == [0,0,0] for rgb in column]):
            cropped_width_right = cropped_width_right - 1
        else:
            break

    cropped_width_right = abs(cropped_width - cropped_width_right)

    cropped_img = image[cropped_height_top : image.shape[0] - cropped_height_bottom, cropped_width_left: image.shape[1] - cropped_width_right]
    
    cropped_points = []

    for point in points:
        cropped_points.append(point[cropped_height_top : image.shape[0] - cropped_height_bottom, cropped_width_left: image.shape[1] - cropped_width_right])

    return cropped_img, cropped_points

def right_side(img: np.ndarray) -> list:
    """
    Retrieves the right side of the object

    Input parameters:
    - ``img`` the array of the image to be processed
    
    Output parameter (``list``):
    - ``points`` (``list``):
        - the end points of the side
    """

    image = img.copy()

    y = image.shape[0]-1

    while True:

        bottom = image[y].tolist()
        bottom = [[pos,y] for pos, pixel in enumerate(bottom) if pixel != [0,0,0]]

        if len(bottom):
            break

        y = y - 1

    bottom = [bottom[0],bottom[-1]]

    if bottom[0][0] < int(image.shape[1] * 0.25) and bottom[1][0] > int(image.shape[1] * 0.75):
        return [bottom[0],bottom[1]]
    
    points = []
    end_point = None

    for y in range(bottom[1][1],-1,-1):
        row = image[y][bottom[1][0]:].tolist()

        x = len(row) - 1

        pixel = None

        while True:
            
            if row[x] != [0,0,0]:
                pixel = row[x]
                break
            
            if x == 0:
                break

            x = x - 1
        
        if pixel != None:
            #image[y][bottom[1][0]+x] = (255,255,255)

            point1 = (bottom[1][0]+x,y)
            point2 = bottom[1]

            distances = []

            for point in points:
                distance = geometric_utils.point_to_line_distance(point,point1,point2)
                distances.append(distance)

            points.append(point1)

            if len(distances):
                
                if max(distances) > 2:
                    end_point = points[-2]
                    break
    
    column = []

    for y in range(image.shape[0]):
        pixel = image[y][end_point[0]].tolist()

        if pixel == [0,0,0]:
            continue

        column.append([pixel,(end_point[0],y)])

    end_point = column[-1][1]
        
    return [list(bottom[1]),list(end_point)]

def left_side(img: np.ndarray) -> list:
    """
    Retrieves the left side of the object

    Input parameters:
    - ``img`` the array of the image to be processed
    
    Output parameter (``list``):
    - ``points`` (``list``):
        - the end points of the side
    """

    image = img.copy()

    y = image.shape[0]-1

    while True:

        bottom = image[y].tolist()
        bottom = [[pos,y] for pos, pixel in enumerate(bottom) if pixel != [0,0,0]]

        if len(bottom):
            break

        y = y - 1

    bottom = [bottom[0],bottom[-1]]

    if bottom[0][0] < int(image.shape[1] * 0.25) and bottom[1][0] > int(image.shape[1] * 0.75):
        return [bottom[0],bottom[1]]
    
    points = []
    end_point = None

    for y in range(bottom[0][1],-1,-1):
        row = image[y][:bottom[0][0]+1].tolist()

        x = 0

        pixel = None

        while True:
            
            if row[x] != [0,0,0]:
                pixel = row[x]
                break
            
            if x == len(row) - 1:
                break

            x = x + 1
        
        if pixel != None:

            point1 = (x,y)
            point2 = bottom[0]

            distances = []

            for point in points:
                distance = geometric_utils.point_to_line_distance(point,point1,point2)
                distances.append(distance)

            points.append(point1)

            if len(distances):
                
                if max(distances) > 2:
                    end_point = points[-2]
                    break
    
    column = []

    for y in range(image.shape[0]):
        pixel = image[y][end_point[0]].tolist()

        if pixel == [0,0,0]:
            continue

        column.append([pixel,(end_point[0],y)])

    end_point = column[-1][1]

    return [list(bottom[0]),list(end_point)]

def extract_table(img: np.ndarray, resolution: tuple = (1920,1080), table: list = zed_params.TABLE_DEFAULT) -> np.ndarray:
    """
    Extracts the table from a given image. In other words, it makes all the image black except for the table.

    Args:
        img (np.ndarray): Image to be processed.
        resolution (tuple, optional): Resolution of the zed camera. Defaults to (1920,1080)
        render (list, optional): Specifies the four vertices (x,y) of the table. Defaults to TABLE_DEFAULT.

    Returns:
        np.ndarray: Processed image
    """

    image = img.copy()

    _table = deepcopy(table)

    if table == zed_params.TABLE_DEFAULT:

        for coordinate in _table:
            coordinate[0] = int(coordinate[0] * resolution[0] / 1280)
            coordinate[1] = int(coordinate[1] * resolution[1] / 720)

    offset = 25

    min_x = min([coordinate[0] for coordinate in _table]) - offset
    min_x = min_x if min_x >= 0 else 0

    max_x = max([coordinate[0] for coordinate in _table]) + offset
    max_x = max_x if max_x < image.shape[1] else image.shape[1] - 1

    min_y = min([coordinate[1] for coordinate in _table]) - offset
    min_y = min_y if min_y >= 0 else 0
    
    max_y = max([coordinate[1] for coordinate in _table]) + offset
    max_y = max_y if max_y < image.shape[0] else image.shape[0] - 1

    zed_params.WINDOW_RECT = {
        "max_x": max_x,
        "min_x": min_x,
        "max_y": max_y,
        "min_y": min_y
    }

    # create the mask
    mask = np.array(_table, dtype = np.int32)

    # create a black background so, all the image except the table will be black
    background = np.zeros((image.shape[0], image.shape[1]), np.int8)

    # fill the mask with the black background
    cv2.fillPoly(background, [mask],255)

    mask_background = cv2.inRange(background, 1, 255)

    # apply the mask
    image = cv2.bitwise_and(image, image, mask = mask_background)

    return image

def extract_obj(img: np.ndarray, box: list = None, tolerance: int = 75) -> np.ndarray:
    """
    Fill the entire image except for the main object

    Input parameters:
    - ``img`` the array of the image to be processed
    - ``box`` (optional, ``default = None``) bounding box of the object. If None, it will be considered the entire image
    - ``tolerance`` (optional, ``default = 75``) each color with a difference >= tolerance from the main color, will be converted in black

    Output parameter (``np.ndarray``):
    - ``img`` (``np.ndarray``):
        - the processed image
    """

    image = img.copy()

    if box == None:
        box = [[0,image.shape[0]],[image.shape[1],image.shape[0]],[image.shape[1],0],[0,0]]

    img_min_x = min(box, key=lambda x:x[0])[0]
    img_max_x = max(box, key=lambda x:x[0])[0]

    img_min_y = min(box, key=lambda x:x[1])[1]
    img_max_y = max(box, key=lambda x:x[1])[1]

    image = image[img_min_y:img_max_y+1,img_min_x:img_max_x+1]

    mc = main_color(image, True, True)

    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            pixel = list(image[y][x])
            
            distance = math.dist(mc, pixel)

            if distance >= tolerance:
                image[y][x] = 0

    return image