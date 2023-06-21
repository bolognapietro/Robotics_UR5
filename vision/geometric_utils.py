def calculate_line(point1: tuple, point2: tuple) -> tuple:
    """
    Calculates the angular coefficient and the quote of the line that has poin1 and point2 as end points.

    Args:
        point1 (tuple): First point.
        point2 (tuple): Second point.

    Returns:
        tuple: Angular coefficient and quote respectively.
    """
    
    if point1[0] == point2[0]:
        return 1,0
    
    m = (point2[1] - point1[1]) / (point2[0] - point1[0])
    q = point1[1] - m*point1[0]

    return m,q