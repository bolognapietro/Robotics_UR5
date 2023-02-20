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

    '''p1 = line_start
    p2 = line_end
    p3 = point

    try:
        x = (p3[1] - p3[0]*((p2[0] - p1[0]) / (p2[1] - p1[1])) - p1[1] + p1[0]*((p2[1] - p1[1]) / (p2[0] - p1[0])))
        x = x / (((p2[1] - p1[1]) / (p2[0] - p1[0])) - ((p2[0] - p1[0]) / (p2[1] - p1[1])))

        y = (x*((p2[1] - p1[1]) / (p2[0] - p1[0])) + p1[1] - p1[0]*((p2[1] - p1[1]) / (p2[0] - p1[0])))

        p4 = (x,y)

        distance =  math.dist(p3,p4)
    except:
        distance = 0'''

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
        triangle_area = math.sqrt(semiperimeter * (semiperimeter - line_length) * (semiperimeter - point_to_start_distance) * (semiperimeter - point_to_end_distance))
    except:
        return point_to_line_distance(point,line_end,line_start)

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