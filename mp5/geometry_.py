# geometry.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by James Gao (jamesjg2@illinois.edu) on 9/03/2021
# Inspired by work done by Jongdeog Lee (jlee700@illinois.edu)

"""
This file contains geometry functions necessary for solving problems in MP5
"""

import numpy as np
from alien import Alien
from typing import List, Tuple
from copy import deepcopy


def does_alien_touch_wall(alien: Alien, walls: List[Tuple[int]]):
    """Determine whether the alien touches a wall

        Args:
            alien (Alien): Instance of Alien class that will be navigating our map
            walls (list): List of endpoints of line segments that comprise the walls in the maze in the format
                         [(startx, starty, endx, endx), ...]

        Return:
            True if touched, False if not
    """
    for startx, starty, endx, endy in walls:
        if alien.is_circle():
            if point_segment_distance(alien.get_centroid(),
                                      [[startx, starty], [endx, endy]]) <= alien.get_width():
                return True
        else:
            head, tail = alien.get_head_and_tail()
            if segment_distance([[startx, starty], [endx, endy]],
                                [head, tail]) <= alien.get_width():
                return True
    return False


def is_alien_within_window(alien: Alien, window: Tuple[int]):
    """Determine whether the alien stays within the window

        Args:
            alien (Alien): Alien instance
            window (tuple): (width, height) of the window
    """
    if alien.is_circle():
        x, y = alien.get_centroid()
        r = alien.get_width()
        if x - r <= 0 or x + r >= window[0]:
            return False

        if y - r <= 0 or y + r >= window[1]:
            return False
    else:
        head, tail = alien.get_head_and_tail()
        head_x, head_y = head
        tail_x, tail_y = tail
        r = alien.get_width()
        if head_x - r <= 0 or tail_x - r <= 0:
            return False
        if head_x + r >= window[0] or tail_x + r >= window[0]:
            return False
        if head_y - r <= 0 or tail_y - r <= 0:
            return False
        if head_y + r >= window[1] or tail_y + r >= window[1]:
            return False
    return True


def is_point_in_polygon(point, polygon):
    """
    Determine whether a point is in a parallelogram.
    Note: The vertex of the parallelogram should be clockwise or counter-clockwise.

        Args:
            point (tuple): shape of (2, ). The coordinate (x, y) of the query point.
            polygon (tuple): shape of (4, 2). The coordinate (x, y) of 4 vertices of the parallelogram.
    """
    assert len(point) == 2
    assert len(polygon) == 4
    Ax, Ay = polygon[0][0], polygon[0][1]
    Bx, By = polygon[1][0], polygon[1][1]
    Cx, Cy = polygon[2][0], polygon[2][1]
    Dx, Dy = polygon[3][0], polygon[3][1]
    if Ax == Bx == Cx == Dx:
        min_y = min([Ay, By, Cy, Dy])
        max_y = max([By, By, Cy, Dy])
        return point_segment_distance(point, ((Ax, min_y), (Ax, max_y))) == 0
    elif Ay == By == Cy == Dy:
        min_x = min([Ax, Bx, Cx, Dx])
        max_x = max([Bx, Bx, Cx, Dx])
        return point_segment_distance(point, ((min_x, Ay), (max_x, Ay))) == 0
    else:
        x, y = point[0], point[1]
        a = (Bx - Ax) * (y - Ay) - (By - Ay) * (x - Ax)
        b = (Cx - Bx) * (y - By) - (Cy - By) * (x - Bx)
        c = (Dx - Cx) * (y - Cy) - (Dy - Cy) * (x - Cx)
        d = (Ax - Dx) * (y - Dy) - (Ay - Dy) * (x - Dx)
        if (a >= 0 and b >= 0 and c >= 0 and d >= 0) or (a <= 0 and b <= 0 and c <= 0 and d <= 0):
            return True
        else:
            return False


def does_alien_path_touch_wall(alien: Alien, walls: List[Tuple[int]], waypoint: Tuple[int, int]):
    """
    Determine whether the alien's straight-line path from its current position to the waypoint touches a wall

        Args:
            alien (Alien): the current alien instance
            walls (List of tuple): List of endpoints of line segments that comprise the walls in the maze in the format
                         [(startx, starty, endx, endx), ...]
            waypoint (tuple): the coordinate of the waypoint where the alien wants to move

        Return:
            True if touched, False if not
    """
    # If the alien changes shape, it can't change position
    # if start[1] != end[1]:
    #     raise ValueError("Alien cannot change shape on straight line path")

    # Case 1: it's a circle, so we check if initial and final are okay, then reformulate the rest of the path as
    # a line segment of length 2r that is perpendicular to the direction of the path

    new_alien = deepcopy(alien)
    new_alien.set_alien_pos(waypoint)
    min_dist = alien.get_width()
    if alien.is_circle():
        if does_alien_touch_wall(alien, walls) or does_alien_touch_wall(new_alien, walls):
            return True
        # Check area carved out by circle's path
        for startx, starty, endx, endy in walls:
            # Distance between wall and path
            if segment_distance([[startx, starty], [endx, endy]],
                                [alien.get_centroid(), waypoint]) <= min_dist:
                return True
    else:
        for startx, starty, endx, endy in walls:
            wall = [(startx, starty), (endx, endy)]
            head1, tail1 = alien.get_head_and_tail()
            head2, tail2 = new_alien.get_head_and_tail()
            # case 1: at least one endpoint of the wall is inside the parallelogram
            polygon = (head1, tail1, tail2, head2)
            # order in polygon is important!
            if is_point_in_polygon(wall[0], polygon) or is_point_in_polygon(wall[1], polygon):
                return True
            # case 2: the wall is outside the parallelogram
            sides = [[head1, tail1], [head2, tail2], [head1, head2], [tail1, tail2]]
            if any((segment_distance(wall, side) <= min_dist) for side in sides):
                return True
    return False


def point_segment_distance(p, s):
    """Compute the distance from the point to the line segment.

        Args:
            p: A tuple (x, y) of the coordinates of the point.
            s: A tuple ((x1, y1), (x2, y2)) of coordinates indicating the endpoints of the segment.

        Return:
            Euclidean distance from the point to the line segment.
    """
    p = np.array(p)  # shape = (2,)
    s = np.array(s)  # shape = (2, 2)

    # Center s.t. s[0] is the origin
    p = p - s[0]

    # q represents centered s using centered endpoint
    q = s[1] - s[0]

    # Compute projection of point onto segment
    proj = np.dot(p, q) / np.linalg.norm(q)

    if proj <= 0:  # Check if point is to the left of the segment
        return np.linalg.norm(p)
    elif proj >= np.linalg.norm(q):  # Check if point is to the right of the segment
        return np.linalg.norm(p - q)
    else:
        if np.isclose(np.linalg.norm(p), proj):  # Check if point is on the segment
            return 0
        else:
            return (np.linalg.norm(p) ** 2 - proj ** 2) ** 0.5


def do_segments_intersect(s1, s2):
    """
    Determine whether segment1 intersects segment2.

        Args:
            s1: A tuple of coordinates indicating the endpoints of segment1.
            s2: A tuple of coordinates indicating the endpoints of segment2.

        Return:
            True if line segments intersect, False if not.
    """
    s1 = np.array(s1)  # shape = (2, 2)
    s2 = np.array(s2)  # shape = (2, 2)

    # Check if the endpoints of one line lands on the other
    if (any(point_segment_distance(p, s2) == 0 for p in s1)
            or any(point_segment_distance(p, s1) == 0 for p in s2)):
        return True

    # Otherwise, the endpoints do not land on the lines
    # Center s.t. s1[0] is the origin
    q = s1[1] - s1[0]
    s = s2 - s1[0]
    # Check if s[0] and s[1] are on same side of q
    if np.cross(s[0], q) * np.cross(s[1], q) >= 0:
        return False
    # Center s.t. s2[0] is the origin
    q = s2[1] - s2[0]
    s = s1 - s2[0]
    # Check if s[0] and s[1] are on same side of q
    if np.cross(s[0], q) * np.cross(s[1], q) >= 0:
        return False
    return True


def segment_distance(s1, s2):
    """Compute the distance from segment1 to segment2.  You will need `do_segments_intersect`.

        Args:
            s1: A tuple of coordinates indicating the endpoints of segment1.
            s2: A tuple of coordinates indicating the endpoints of segment2.

        Return:
            Euclidean distance between the two line segments.
    """
    if do_segments_intersect(s1, s2):
        return 0
    else:
        return min(point_segment_distance(s1[0], s2),
                   point_segment_distance(s1[1], s2),
                   point_segment_distance(s2[0], s1),
                   point_segment_distance(s2[1], s1))


if __name__ == '__main__':

    from geometry_test_data import walls, goals, window, alien_positions, alien_ball_truths, alien_horz_truths, \
        alien_vert_truths, point_segment_distance_result, segment_distance_result, is_intersect_result, waypoints

    # Here we first test your basic geometry implementation
    def test_point_segment_distance(points, segments, results):
        num_points = len(points)
        num_segments = len(segments)
        for i in range(num_points):
            p = points[i]
            for j in range(num_segments):
                seg = ((segments[j][0], segments[j][1]), (segments[j][2], segments[j][3]))
                cur_dist = point_segment_distance(p, seg)
                assert abs(cur_dist - results[i][j]) <= 10 ** -3, \
                    f'Expected distance between {points[i]} and segment {segments[j]} is {results[i][j]}, ' \
                    f'but get {cur_dist}'


    def test_do_segments_intersect(center: List[Tuple[int]], segments: List[Tuple[int]],
                                   result: List[List[List[bool]]]):
        for i in range(len(center)):
            for j, s in enumerate([(40, 0), (0, 40), (100, 0), (0, 100), (0, 120), (120, 0)]):
                for k in range(len(segments)):
                    cx, cy = center[i]
                    st = (cx + s[0], cy + s[1])
                    ed = (cx - s[0], cy - s[1])
                    a = (st, ed)
                    b = ((segments[k][0], segments[k][1]), (segments[k][2], segments[k][3]))
                    if do_segments_intersect(a, b) != result[i][j][k]:
                        if result[i][j][k]:
                            assert False, f'Intersection Expected between {a} and {b}.'
                        if not result[i][j][k]:
                            assert False, f'Intersection not expected between {a} and {b}.'


    def test_segment_distance(center: List[Tuple[int]], segments: List[Tuple[int]], result: List[List[float]]):
        for i in range(len(center)):
            for j, s in enumerate([(40, 0), (0, 40), (100, 0), (0, 100), (0, 120), (120, 0)]):
                for k in range(len(segments)):
                    cx, cy = center[i]
                    st = (cx + s[0], cy + s[1])
                    ed = (cx - s[0], cy - s[1])
                    a = (st, ed)
                    b = ((segments[k][0], segments[k][1]), (segments[k][2], segments[k][3]))
                    distance = segment_distance(a, b)
                    assert abs(result[i][j][k] - distance) <= 10 ** -3, f'The distance between segment {a} and ' \
                                                                        f'{b} is expected to be {result[i]}, but your' \
                                                                        f'result is {distance}'


    def test_helper(alien: Alien, position, truths):
        alien.set_alien_pos(position)
        config = alien.get_config()

        touch_wall_result = does_alien_touch_wall(alien, walls)
        in_window_result = is_alien_within_window(alien, window)

        assert touch_wall_result == truths[
            0], f'does_alien_touch_wall(alien, walls) with alien config {config} returns {touch_wall_result}, ' \
                f'expected: {truths[0]}'
        assert in_window_result == truths[
            2], f'is_alien_within_window(alien, window) with alien config {config} returns {in_window_result}, ' \
                f'expected: {truths[2]}'


    def test_check_path(alien: Alien, position, truths, waypoints):
        alien.set_alien_pos(position)
        config = alien.get_config()
        new_walls = walls + [(50, 119, 50, 121)]

        for i, waypoint in enumerate(waypoints):
            path_touch_wall_result = does_alien_path_touch_wall(alien, new_walls, waypoint)

            assert path_touch_wall_result == truths[
                i], f'does_alien_path_touch_wall(alien, walls, waypoint) with alien config {config} ' \
                    f'and waypoint {waypoint} returns {path_touch_wall_result}, ' \
                    f'expected: {truths[i]}'
    
    def test_is_point_in_polygon():
        def assert_point_in_polygon(point, polygon, expected):
            result = is_point_in_polygon(point, polygon)
            assert result == expected, \
            f'is_point_in_polygon with point {point} and polygon {polygon} ' \
            f'returns {result}, expected:{expected}'

        # Test case 1: Point inside a rectangle
        point1 = (5, 5)
        polygon1 = [(0, 0), (10, 0), (10, 10), (0, 10)]
        assert_point_in_polygon(point1, polygon1, True)

        # Test case 2: Point outside the rectangle
        point2 = (-1, 5)
        assert_point_in_polygon(point2, polygon1, False)

        # Test case 3: Point on the edge of the rectangle
        point3 = (10, 5)
        assert_point_in_polygon(point3, polygon1, True)

        # Test case 4: Point inside a parallelogram
        point4 = (3, 2)
        polygon2 = [(1, 1), (6, 1), (4, 3), (-1, 3)]
        assert_point_in_polygon(point4, polygon2, True)

        # Test case 5: Point outside the parallelogram
        point5 = (6, 2)
        assert_point_in_polygon(point5, polygon2, False)

        # Test case 6: Point on the vertex of the polygon
        point6 = (10, 10)
        assert_point_in_polygon(point6, polygon1, True)

        # Test case 7-8: Horizontal line, inside/outside
        point7, point8 = (2, 0), (5, 0)
        polygon3 = [(0, 0), (1, 0), (3, 0), (4, 0)]
        assert_point_in_polygon(point7, polygon3, True)
        assert_point_in_polygon(point8, polygon3, False)

        # Test case 9-10: Vertical line, inside
        point9, point10 = (0, 2), (0, 5)
        polygon4 = [(0, 0), (0, 1), (0, 3), (0, 4)]
        assert_point_in_polygon(point9, polygon4, True)
        assert_point_in_polygon(point10, polygon4, False)


    # Initialize Aliens and perform simple sanity check.
    alien_ball = Alien((30, 120), [40, 0, 40], [11, 25, 11], ('Horizontal', 'Ball', 'Vertical'), 'Ball', window)
    test_helper(alien_ball, alien_ball.get_centroid(), (False, False, True))

    alien_horz = Alien((30, 120), [40, 0, 40], [11, 25, 11], ('Horizontal', 'Ball', 'Vertical'), 'Horizontal', window)
    test_helper(alien_horz, alien_horz.get_centroid(), (False, False, True))

    alien_vert = Alien((30, 120), [40, 0, 40], [11, 25, 11], ('Horizontal', 'Ball', 'Vertical'), 'Vertical', window)
    test_helper(alien_vert, alien_vert.get_centroid(), (True, False, True))

    edge_horz_alien = Alien((50, 100), [100, 0, 100], [11, 25, 11], ('Horizontal', 'Ball', 'Vertical'), 'Horizontal',
                            window)
    edge_vert_alien = Alien((200, 70), [120, 0, 120], [11, 25, 11], ('Horizontal', 'Ball', 'Vertical'), 'Vertical',
                            window)
    
    # Test point in polygon
    test_is_point_in_polygon()

    # Test validity of straight line paths between an alien and a waypoint
    alien_ball1 = Alien((30, 120), [20, 0, 20], [11, 20, 11], ('Horizontal', 'Ball', 'Vertical'), 'Ball', window)
    alien_horz1 = Alien((30, 120), [20, 0, 20], [11, 20, 11], ('Horizontal', 'Ball', 'Vertical'), 'Horizontal', window)
    alien_vert1 = Alien((30, 120), [20, 0, 20], [11, 20, 11], ('Horizontal', 'Ball', 'Vertical'), 'Vertical', window)
    alien_horz2 = Alien((30, 120), [30, 0, 30], [11, 20, 11], ('Horizontal', 'Ball', 'Vertical'), 'Horizontal', window)

    test_check_path(alien_ball, (30, 120), (False, True, True), waypoints)
    test_check_path(alien_horz, (30, 120), (False, True, False), waypoints)
    test_check_path(alien_vert, (30, 120), (True, True, True), waypoints)
    centers = alien_positions
    segments = walls
    test_point_segment_distance(centers, segments, point_segment_distance_result)
    test_do_segments_intersect(centers, segments, is_intersect_result)
    test_segment_distance(centers, segments, segment_distance_result)

    for i in range(len(alien_positions)):
        test_helper(alien_ball, alien_positions[i], alien_ball_truths[i])
        test_helper(alien_horz, alien_positions[i], alien_horz_truths[i])
        test_helper(alien_vert, alien_positions[i], alien_vert_truths[i])

    # Edge case coincide line endpoints
    test_helper(edge_horz_alien, edge_horz_alien.get_centroid(), (True, False, False))
    test_helper(edge_horz_alien, (110, 55), (True, True, True))
    test_helper(edge_vert_alien, edge_vert_alien.get_centroid(), (True, False, True))

    print("Geometry tests passed\n")
