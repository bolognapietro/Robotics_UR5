#!/usr/bin/env python

# Standard library modules
from copy import deepcopy

import cv2
import numpy as np

# Local imports
import zed_params

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

def extract_objects(img: np.ndarray) -> np.ndarray:
    """
    Fill the entire image of black except for the main objects.

    Args:
        img (np.ndarray): Image to be processed.

    Returns:
        np.ndarray: The processed image.
    """

    image = img.copy()

    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # define the color ranges for red, green, blue, yellow, and fuchsia in HSV
    color_ranges = {
        'red': [(0, 50, 50), (10, 255, 255)], # Hue range: 0-10
        'green': [(36, 50, 50), (70, 255, 255)], # Hue range: 36-70
        'blue': [(90, 50, 50), (130, 255, 255)], # Hue range: 90-130
        'yellow': [(20, 50, 50), (35, 255, 255)], # Hue range: 20-35
        'fuchsia': [(145, 50, 50), (175, 255, 255)], # Hue range: 145-175
        'orange': [(11, 50, 50), (25, 255, 255)] # Hue range: 11-25
    }

    # create an empty mask
    mask = np.zeros(image.shape[:2], dtype=np.uint8)

    # iterate over color ranges and add masks
    for color_range in color_ranges.values():
        lower_color = np.array(color_range[0])
        upper_color = np.array(color_range[1])
        color_mask = cv2.inRange(hsv_image, lower_color, upper_color)
        mask = cv2.bitwise_or(mask, color_mask)

    # apply the mask to the original image to extract the colored objects
    result = cv2.bitwise_and(image, image, mask=mask)

    # set the background pixels to black
    background = np.zeros_like(image)
    # set background color to black
    background[mask == 0] = [0, 0, 0]

    # combine the objects and background
    return cv2.add(result, background)

def remove_noise_by_color(img: np.ndarray) -> np.ndarray:
    """
    Remove any colored interference inside the image box

    Args:
        img (np.ndarray): Image to be processed.

    Returns:
        np.ndarray: The processed image.
    """

    image = img.copy()

    # convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # threshold the image to obtain a binary mask
    _, binary_mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

    # find contours in the binary mask
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    contours.pop(0)

    # iterate over the contours and fill each contour area with black color
    for contour in contours:
        cv2.drawContours(image, [contour], 0, (0, 0, 0), -1)

    return image
