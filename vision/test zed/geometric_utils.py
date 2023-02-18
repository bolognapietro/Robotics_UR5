import math

def point_to_line_distance(point: tuple, line_start: tuple, line_end: tuple) -> float:
    """
    Calculates the distance between a point and a line

    Input parameters:
    - ``point``
    - ``line_start``
    - ``line_end``

    Output parameter (``float``):
    - ``distance`` (``np.ndarray``):
        - the distance between the point and the line
    """

    # Check if the line segment is a point
    if line_start == line_end:
        return math.sqrt((point[0] - line_start[0])**2 + (point[1] - line_start[1])**2)

    # Calculate the length of the line segment
    line_length = math.sqrt((line_end[0] - line_start[0])**2 + (line_end[1] - line_start[1])**2)

    # Calculate the distance from the start of the line segment to the point
    point_to_start_distance = math.sqrt((point[0] - line_start[0])**2 + (point[1] - line_start[1])**2)

    # Calculate the distance from the end of the line segment to the point
    point_to_end_distance = math.sqrt((point[0] - line_end[0])**2 + (point[1] - line_end[1])**2)

    # Calculate the area of the triangle formed by the point and the two endpoints of the line
    semiperimeter = (line_length + point_to_start_distance + point_to_end_distance) / 2
    triangle_area = math.sqrt(semiperimeter * (semiperimeter - line_length) * (semiperimeter - point_to_start_distance) * (semiperimeter - point_to_end_distance))

    # Calculate the distance between the point and the line
    distance = 2 * triangle_area / line_length

    return distance

def point_distance(point1: tuple, point2: tuple) -> float:
    """
    Calculates the euclidean distance between two points

    Input parameters:
    - ``point1``,  ``point2`` the points

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