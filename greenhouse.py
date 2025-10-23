#!/usr/bin/python3
import numpy as np
import matplotlib.pyplot as plt


def glazing_grid():
    """
    Defines the geometry of the glazing surface.

    Returns
    -------
    np.ndarray of glazing surface represented as [N, 3] array of [South, East, Up] coordinates from some reference.
    """
    x_coordinates = np.array((25), dtype=np.float32)
    y_coordinates = np.arange(-206, -40, 1, dtype=np.float32)
    z_coordinates = np.arange(0, 80, dtype=np.float32)

    x, y, z = np.meshgrid(x_coordinates, y_coordinates, z_coordinates)
    combined = np.vstack((x.flatten(), y.flatten(), z.flatten())).T
    return combined


def roof_edge():
    """
    Defines the geometry of the roof edge.

    Returns
    -------
    start_point: np.ndarray of shape [3] giving coordinates of one end of roof edge
    end_point: np.ndarray of shape [3] giving coordinates of other end of roof edge
    slope: np.ndarray of shape [3] giving slope of the roof edge
    """
    start_point = np.array((0, 0, 141.35), dtype=np.float32)
    slope = np.array((1, 0, -2.0/13), dtype=np.float32)
    end_point = start_point + 220 * slope
    slope_norm = np.linalg.norm(slope)
    
    return start_point, end_point, slope / slope_norm


def solar_points():
    """
    Gets the solar positions as an array of [South, East, Up] coordinates.
    """
    point = np.array((0, 1, 1), dtype=np.float32).reshape(1, 3)
    norm = np.linalg.norm(point, axis=-1)
    return point / norm


def sun_zenith_plane_normal(sun_points: np.ndarray):
    """
    Calculates the normal for the plane defined by zenith and the sun points.
    """
    normals = np.stack((-sun_points[..., 1], sun_points[..., 0], np.zeros(sun_points.shape[:-1]))).reshape(-1, 3)
    norms = np.linalg.norm(normals, axis=-1)
    return normals / norms


def line_intersects_plane(line_start: np.ndarray, line_slope: np.ndarray, plane_point: np.ndarray, plane_normal: np.ndarray):
    """
    Gets the point along `line` that intersects the plane defined by `plane_point` and `plane_normal`.
    """
    numerator = np.einsum("ix,jx->ij", plane_point - line_start, plane_normal)
    denominator = np.einsum("x,ix->i", line_slope, plane_normal)
    distance_along_line = numerator / denominator
    intersection = line_start + distance_along_line * line_slope
    return intersection


def angle_with_zenith(points: np.ndarray):
    """
    Calculates the angle between zenith and points [radians].

    Parameters
    ----------
    points: np.ndarray of shape [..., 3] of (South, East, Up) coordinates of the points

    Returns
    -------
    Angles between zenith and points, np.ndarray of shape [...]
    """
    norm = np.linalg.norm(points, axis=-1)
    angle = np.arccos(points[..., 2] / norm)
    return angle
    

if __name__ == "__main__":
    sun_points = solar_points()
    sun_zenith_angles = angle_with_zenith(sun_points)
    sun_normals = sun_zenith_plane_normal(sun_points)
    roof_start, roof_end, roof_slope = roof_edge()
    glazing_points = glazing_grid()
    roof_intersection_points = line_intersects_plane(roof_start - glazing_points, roof_slope, sun_points, sun_normals)
    roof_zenith_angles = angle_with_zenith(roof_intersection_points)
    is_sunlit = roof_zenith_angles > sun_zenith_angles
    fraction_lit = np.count_nonzero(is_sunlit) / is_sunlit.size
    print(fraction_lit)
