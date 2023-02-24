from typing import List

import math
import numpy as np
from sumolib.net.edge import Edge


def sort_roads_clockwise(roads: List[Edge], switch_src_dest: bool = False) -> List[Edge]:
    dest_coords = [np.array(list(road.getToNode().getCoord())) for road in roads]
    src_coords = [np.array(list(road.getFromNode().getCoord())) for road in roads]
    src_coords, dest_coords = (dest_coords, src_coords) if switch_src_dest else (src_coords, dest_coords)
    twelve_pm = np.array([0.0, 1.0])
    angles = [rotation_angle(twelve_pm, src_coord - dest_coord) for src_coord, dest_coord in
              zip(src_coords, dest_coords)]
    return [edge for _, edge in sorted(zip(angles, roads))]


def rotation_angle(a: np.ndarray, b: np.ndarray) -> float:
    unit_a = a / np.linalg.norm(a)
    unit_b = b / np.linalg.norm(b)
    dot = np.dot(unit_a, unit_b)
    cross = np.cross(b, a)
    angle = math.atan2(cross, dot)
    if angle < 0:
        angle = 2 * math.pi + angle
    return angle
