from typing import List

import math
import numpy as np
from sumolib.net.edge import Edge


def angle_approach(approach: Edge, reference_vec: np.ndarray = np.array([0.0, 1.0]),
                   switch_src_dest: bool = False):
    dest_coord = np.array(list(approach.getToNode().getCoord()))
    src_coord = np.array(list(approach.getFromNode().getCoord()))
    src_coord, dest_coord = (dest_coord, src_coord) if switch_src_dest else (src_coord, dest_coord)
    approach_vec = src_coord - dest_coord
    return rotation_angle(reference_vec, approach_vec)

def angle_between_approaches(approach_a: Edge, approach_b: Edge):
    dest_coord_a = np.array(list(approach_a.getToNode().getCoord()))
    src_coord_a = np.array(list(approach_a.getFromNode().getCoord()))
    dest_coord_b = np.array(list(approach_b.getToNode().getCoord()))
    src_coord_b = np.array(list(approach_b.getFromNode().getCoord()))
    approach_a_vec = src_coord_a - dest_coord_a
    approach_b_vec = src_coord_b - dest_coord_b
    return rotation_angle(approach_a_vec, approach_b_vec)

def sort_approaches_clockwise(approaches: List[Edge], switch_src_dest: bool = False) -> List[Edge]:
    angles = [angle_approach(approach, switch_src_dest=switch_src_dest) for approach in approaches]
    return [edge for _, edge in sorted(zip(angles, approaches))]


def rotation_angle(a: np.ndarray, b: np.ndarray) -> float:
    unit_a = a / np.linalg.norm(a)
    unit_b = b / np.linalg.norm(b)
    dot = np.dot(unit_a, unit_b)
    cross = np.cross(b, a)
    angle = math.atan2(cross, dot)
    if angle < 0:
        angle = 2 * math.pi + angle
    return angle
