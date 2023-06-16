import math

def point_to_line_distance(point: tuple, line_start: tuple, line_end: tuple, recursion_flag: bool = False) -> float:
    """
    Calculates the distance between a point and a line.

    Args:
        point (tuple): Point.
        line_start (tuple): End point of the line.
        line_end (tuple): End point of the line

    Returns:
        float: Distance.
    """

    # Check if the line segment is a point
    if line_start == line_end:
        return math.sqrt((point[0] - line_start[0])**2 + (point[1] - line_start[1])**2)

    if list(point) in [list(line_start), list(line_end)]:
        return 0
    
    # Calculate the length of the line segment
    line_length = math.sqrt((line_end[0] - line_start[0])**2 + (line_end[1] - line_start[1])**2)

    # Calculate the distance from the start of the line segment to the point
    point_to_start_distance = math.sqrt((point[0] - line_start[0])**2 + (point[1] - line_start[1])**2)

    # Calculate the distance from the end of the line segment to the point
    point_to_end_distance = math.sqrt((point[0] - line_end[0])**2 + (point[1] - line_end[1])**2)

    # Calculate the area of the triangle formed by the point and the two endpoints of the line
    semiperimeter = (line_length + point_to_start_distance + point_to_end_distance) / 2

    try:
        sqrt_value = semiperimeter * (semiperimeter - line_length) * (semiperimeter - point_to_start_distance) * (semiperimeter - point_to_end_distance)
        sqrt_value = round(sqrt_value,5)

        triangle_area = math.sqrt(sqrt_value)
    except:

        if not recursion_flag:
            return point_to_line_distance(point,line_end,line_start,True)
        else:
            raise Exception(f"Unable to calculate the distance of {point} from the given line.\npoint: {point}\nline_start: {line_start}\nline_end: {line_end}")

    # Calculate the distance between the point and the line
    distance = 2 * triangle_area / line_length

    return distance

def calculate_line(point1: tuple, point2: tuple) -> tuple:
    """
    Calculates the angular coefficient and the quote of the line that has poin1 and point2 as end points.

    Args:
        point1 (tuple): First point.
        point2 (tuple): Second point.

    Returns:
        tuple: Angular coefficient and quote respectively.
    """
    
    m = (point2[1] - point1[1]) / (point2[0] - point1[0])
    q = point1[1] - m*point1[0]

    return m,q