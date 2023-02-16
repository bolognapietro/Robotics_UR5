import json
import math
from collections import Counter
from os.path import join

import cv2
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as compare_ssim

def add_background(img: np.ndarray, width: int, height: int, color: tuple = (0,0,0,255)):
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

    image = img.copy()

    image = Image.fromarray(image)
    img_w, img_h = image.size

    background = Image.new('RGBA', (width, height), color)
    bg_w, bg_h = background.size

    offset = ((bg_w - img_w) // 2, (bg_h - img_h) // 2)
    background.paste(image, offset)

    background = np.asarray(background)

    return background

def main_color(img: np.ndarray, noblack: bool = False) -> tuple:
    """
    Retrieves the main color of the image

    Input parameters:
    - ``img`` the array of the image to be processed
    - ``noblack`` (optional, ``default = False``) if true discards black color

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
            return color
        
        index = index + 1

def point_distance(point1: tuple, point2: tuple) -> float:
    """
    Calculates the euclidean distance between two points

    Input parameters:
    - ``poin1``,  ``poin2`` the points

    Output parameter (``float``):
    - ``distance`` (``np.ndarray``):
        - the distance between the two points
    """

    if type(point1) != tuple:
        point1 = tuple(point1)
    
    if type(point2) != tuple:
        point2 = tuple(point2)
    
    assert len(point1) == len(point2), f"{point1} is not compatible with {point2}. {len(point1)} != {len(point2)}"

    total = 0

    for i in range(len(point1)):
        total = total + (point1[i] - point2[i])**2

    return math.sqrt(total)

def one_color(img,color: tuple = None):
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
        color = main_color(image,True)

    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            
            pixel = image[y][x].tolist()

            if pixel != [0,0,0]:
                image[y][x] = color

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

    mc = main_color(image)

    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            pixel = list(image[y][x])
            
            distance = point_distance(mc, pixel)

            if distance >= tolerance:
                image[y][x] = 0

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

def directions(img: np.ndarray) -> np.ndarray:
    """
    Draws the main direction of the object

    Input parameters:
    - ``img`` the array of the image to be processed

    Output parameter (``tuple``):
    - ``img`` (``np.ndarray``):
        - the processed image
    """

    image = img.copy()

    top_axis = image[0].tolist()
    top_axis = [item[0] for item in [[[0,a],b] for a,b in enumerate(top_axis)] if item[1] != [0,0,0]]

    bottom_axis = image[image.shape[0]-1].tolist()
    bottom_axis = [item[0] for item in [[[image.shape[0]-1,a],b] for a,b in enumerate(bottom_axis)] if item[1] != [0,0,0]]

    max_vertical_distance = [
        [point_distance(top_axis[0],bottom_axis[0]),[top_axis[0],bottom_axis[0]]],
        [point_distance(top_axis[0],bottom_axis[-1]),[top_axis[0],bottom_axis[-1]]],
        [point_distance(top_axis[-1],bottom_axis[0]),[top_axis[-1],bottom_axis[0]]],
        [point_distance(top_axis[-1],bottom_axis[-1]),[top_axis[-1],bottom_axis[-1]]]
    ]

    max_vertical_distance = max(max_vertical_distance, key = lambda x: x[0])

    left_axis = [image[y][0].tolist() for y in range(image.shape[0])]
    left_axis = [item[0] for item in [[[a,0],b] for a,b in enumerate(left_axis)] if item[1] != [0,0,0]]

    right_axis = [image[y][-1].tolist() for y in range(image.shape[0])]
    right_axis = [item[0] for item in [[[a,image.shape[1]-1],b] for a,b in enumerate(right_axis)] if item[1] != [0,0,0]]

    max_horizontal_distance = [
        [point_distance(left_axis[0],right_axis[0]),[left_axis[0],right_axis[0]]],
        [point_distance(left_axis[0],right_axis[-1]),[left_axis[0],right_axis[-1]]],
        [point_distance(left_axis[-1],right_axis[0]),[left_axis[-1],right_axis[0]]],
        [point_distance(left_axis[-1],right_axis[-1]),[left_axis[-1],right_axis[-1]]]
    ]

    max_horizontal_distance = max(max_horizontal_distance, key = lambda x: x[0])

    max_distance = max([max_vertical_distance, max_horizontal_distance], key = lambda x: x[0])
    max_distance[1][0].reverse()
    max_distance[1][1].reverse()

    cv2.line(image, max_distance[1][0], max_distance[1][1], (0,0,255), 2)

    return image

def segment_orientation(img: np.ndarray) -> str:
    """
    Retrieves the object orientation by analyzing each side of the image

    Input parameters:
    - ``img`` the array of the image to be processed
    
    Output parameter (``string``):
    - ``tag`` (``string``):
        - the retrieved orientation
    """

    image = img.copy()
    image = one_color(img=image)

    # check top
    rows_with_black_top = 0

    for y in range(int(img.shape[0] / 2)):
        
        row = img[y]
        row = row.tolist()

        black_pixels = [rgb for rgb in row if rgb == [0,0,0]]
        rows_with_black_top = rows_with_black_top + len(black_pixels)

    rows_with_black_top

    # check bottom
    rows_with_black_bottom = 0

    for y in range(img.shape[0]-1,int(img.shape[0] / 2),-1):
        
        row = img[y]
        row = row.tolist()

        black_pixels = [rgb for rgb in row if rgb == [0,0,0]]
        rows_with_black_bottom = rows_with_black_bottom + len(black_pixels)

    rows_with_black_bottom

    # check left
    rows_with_black_left = 0

    for x in range(int(img.shape[1]/2)):
        
        column = [img[y][x].tolist() for y in range(img.shape[0])]

        black_pixels = [rgb for rgb in column if rgb == [0,0,0]]
        rows_with_black_left = rows_with_black_left + len(black_pixels)

    rows_with_black_left

    # check right
    rows_with_black_right = 0

    for x in range(img.shape[1]-1,int(img.shape[1] / 2),-1):
        
        column = [img[y][x].tolist() for y in range(img.shape[0])]

        black_pixels = [rgb for rgb in column if rgb == [0,0,0]]
        rows_with_black_right = rows_with_black_right + len(black_pixels)

    rows_with_black_right

    results = [rows_with_black_top, rows_with_black_bottom, rows_with_black_left, rows_with_black_right]
    duplicate = []

    for result in results:
        if result not in duplicate:
            duplicate.append(result)
        else:
            duplicate.remove(result)

    results = duplicate

    result = max(results)
    result_index = results.index(result)

    if result_index == 0:
        return "TOP"
    elif result_index == 1:
        return "BOTTOM"
    elif result_index == 2:
        return "LEFT"
    elif result_index == 3:
        return "RIGHT"
    
    return "ERROR"

def similarity_orientation(img: np.ndarray, ref: np.ndarray, fill_color: tuple = (255,255,255)) -> tuple:
    """
    Retrieves the object orientation by otating a reference image and selecting the rotation that is most similar to the original image

    Input parameters:
    - ``img`` target image
    - ``ref`` ref reference image
    - ``fill_color`` (optional, ``default = (255,255,255)``) fill objects color

    Output parameter (``tuple``):
    - ``best`` (``list``):
        - score
        - rotation angle
    - ``tag`` (``string``):
        - the retrieved orientation
    """

    image = img.copy()
    reference = ref.copy()

    image = one_color(image,fill_color)
    reference = one_color(reference,fill_color)

    width, height = max(image.shape[1],reference.shape[1]), max(image.shape[0],reference.shape[0])

    image = add_background(image,width,height)
    reference = add_background(reference,width,height)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    reference = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)

    ssim_score = []

    for i in range(0,360):
        angle = i

        rotation_matrix = cv2.getRotationMatrix2D((width/2, height/2), angle, 1)
        rotated_img = cv2.warpAffine(reference, rotation_matrix, (width, height))

        # Calculate the SSIM score
        ssim = compare_ssim(image, rotated_img)
        ssim_score.append([round(ssim*100,2),i])

    best = max(ssim_score, key=lambda x: x[0])

    rotation_matrix = cv2.getRotationMatrix2D((width/2, height/2), best[1], 1)
    rotated_img = cv2.warpAffine(reference, rotation_matrix, (width, height))

    rotation = best[1]

    tag = "ERROR"

    if rotation in range(0,45) or rotation in range(315,360):
        tag = "TOP"
    elif rotation in range(45, 135):
        tag = "LEFT"
    elif rotation in range(135, 225):
        tag = "BOTTOM"
    elif rotation in range(225, 315):
        tag = "RIGHT"

    return best, tag

def process_frame(objects: dict, frame: np.ndarray, dataset_path: str) -> dict:
    """
    Retrieves a superifical orientation of the given objects

    Input parameters:
    - ``objects`` dict of objects
    - ``frame`` captured image by zed camera
    - ``dataset_path`` path to the reference object images

    Output parameter (``dict``):
    - ``objects`` (``dict``):
        - processed objects
    """

    for obj in objects:

        image = extract_obj(img=frame, box=obj["box"])

        box_gazebo_world_frame = [point["3D"] for point in obj["box_gazebo_world_frame"]]
        box_gazebo_world_frame = np.array(box_gazebo_world_frame).reshape(image.shape[0], image.shape[1], -1)

        box_robot_world_frame = [point["3D"] for point in obj["box_robot_world_frame"]]
        box_robot_world_frame = np.array(box_robot_world_frame).reshape(image.shape[0], image.shape[1], -1)

        points = [box_gazebo_world_frame, box_robot_world_frame]

        image, points = crop_img(img=image, points=points)

        box_gazebo_world_frame, box_robot_world_frame = points

        reference = cv2.imread(join(dataset_path,f"{obj['label_name']}.png"))

        tag1 = segment_orientation(img=image)
        best, tag2 = similarity_orientation(image, reference)

        obj["orientation"] = {
            "segment_orientation": tag1,
            "similarity_orientation": tag2
        }

    return obj