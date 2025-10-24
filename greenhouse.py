#!/usr/bin/python3
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def glazing_grid():
    """
    Defines the geometry of the glazing surface.

    Returns
    -------
    np.ndarray of glazing surface represented as [N, 3] array of [South, East, Up] coordinates from some reference.
    """
    rise = 54
    run = 32
    span = 63
    z = np.flip(np.arange(0, rise, rise/span, dtype=np.float32), axis=0)
    y = np.arange(-206, -40, 1, dtype=np.float32)
    x = np.array((0))
    x, y, z = np.meshgrid(x, y, z)
    x_vals = np.arange(18.5, 18.5 + run, run/span, dtype=np.float32)
    x[:, :, :] = x_vals[:]
    combined = np.vstack((x.flatten(), y.flatten(), z.flatten())).T
    
    step = combined[1, :] - combined[0, :]
    normal = np.array((rise, 0, run), dtype=np.float32)
    norm = np.linalg.norm(normal)

    return combined, normal / norm


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
    az_el = np.genfromtxt("solar_az_el.csv", skip_header=1, dtype=np.float32, delimiter=",")
    new_points = np.zeros((az_el.shape[0], 3), dtype=np.float32)
    new_points[:, 0] = -np.cos(np.deg2rad(az_el[:, 1]))
    new_points[:, 1] = np.sin(np.deg2rad(az_el[:, 1]))
    new_points[:, 2] = np.sin(np.deg2rad(az_el[:, 0]))
    norm = np.linalg.norm(new_points, axis=-1)
    return new_points / norm[:, np.newaxis]


def sun_zenith_plane_normal(sun_points: np.ndarray):
    """
    Calculates the normal for the plane defined by zenith and the sun points.
    """
    normals = np.stack((-sun_points[..., 1], sun_points[..., 0], np.zeros(sun_points.shape[0])), axis=1)
    norms = np.linalg.norm(normals, axis=-1)
    return normals / norms[:, np.newaxis]


def line_intersects_plane(line_start: np.ndarray, line_slope: np.ndarray, plane_point: np.ndarray, plane_normal: np.ndarray):
    """
    Gets the point along `line` that intersects the plane defined by `plane_point` and `plane_normal`.
    """
    numerator = np.einsum("ijx,ix->ij", plane_point[:, np.newaxis] - line_start[np.newaxis, :], plane_normal)
    denominator = np.einsum("x,jx->j", line_slope, plane_normal)
    distance_along_line = numerator / denominator[:, np.newaxis]
    intersection = line_start[np.newaxis, :, :] + distance_along_line[..., np.newaxis] * line_slope[np.newaxis, np.newaxis, :]
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

    glazing_points, glazing_normal = glazing_grid()
    print(glazing_normal)
    roof_start, roof_end, roof_slope = roof_edge()
    adjusted_roof_start = roof_start - glazing_points
    adjusted_roof_end = roof_end - glazing_points
    tstamps = []
    for minute in range(0, 1440, 6):
        tstamps.append(dt.datetime(2025, 1, 1) + dt.timedelta(minutes=minute))
    
    illuminated_cells = []
    num_days = 365
    start_day = 0
    for i in range(start_day, start_day + num_days):
        solar_positions = sun_points[i*240 : (i+1)*240]
        sun_zenith_angles = angle_with_zenith(solar_positions)
        sun_normals = sun_zenith_plane_normal(solar_positions)
        
        glazing_sun_alignment = np.dot(solar_positions, glazing_normal)
        glazing_sun_alignment[glazing_sun_alignment < 0] = 0.0
         
        roof_intersection_points = line_intersects_plane(adjusted_roof_start, roof_slope, solar_positions, sun_normals)
        roof_zenith_angles = angle_with_zenith(roof_intersection_points)
        print(roof_zenith_angles.shape)
        no_intersections = np.nonzero(~np.isfinite(roof_intersection_points[..., 0]))  # calculate points where no intersection found
        intersection_past_end = np.nonzero(np.linalg.norm(roof_intersection_points, axis=-1) > np.linalg.norm(adjusted_roof_end, axis=-1))  # past the end of the roof line
        roof_zenith_angles[no_intersections] = np.pi / 2.0  # set to 90 degrees, only larger than sun angle if sun is set
        roof_zenith_angles[intersection_past_end] = np.pi / 2.0
        roof_zenith_angles[solar_positions[:, 1] < 0] = np.pi / 2.0  # when sun is to the west (roof is east)

        is_sunlit = roof_zenith_angles > sun_zenith_angles[:, np.newaxis]
        #if i == start_day:
        #    fig, ax = plt.subplots(1, 1)
        #    ax.plot(np.rad2deg(np.arccos(glazing_sun_alignment)))
        #    ax.plot(np.rad2deg(roof_zenith_angles[:, 0]))
        #    ax.plot(np.rad2deg(sun_zenith_angles))
        #    plt.show()
        #    plt.close()
        illuminated_cells.append(np.sum(is_sunlit, axis=1))
    
    SQM_PER_SQIN = 0.0254 * 0.0254
    SOLAR_FLUX_W_PER_SQM = 1000.0  # per wikipedia, at the surface on clear day
    
    sqm_illuminated = np.array(illuminated_cells) * SQM_PER_SQIN
    insolation = SOLAR_FLUX_W_PER_SQM * sqm_illuminated * glazing_sun_alignment[np.newaxis, :]
    daily_cumulative = np.sum(insolation, axis=1) * 360

    fig, ax = plt.subplots(1, 2, width_ratios=[1, 0.2], sharey='all', figsize=(8, 6))
    img = ax[0].imshow(insolation / 1000, origin='lower', aspect='auto', interpolation='none', extent=(tstamps[0], tstamps[-1], start_day + 0.5, start_day + num_days + 0.5))
    fig.colorbar(img, ax=ax[0], label="Solar Power [kW]")
    ax[0].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax[0].set_ylabel("Day of Year")
    ax[0].set_xlabel("Time of Day")
    ax[1].plot(daily_cumulative / 1e6, np.arange(start_day + 1, start_day + num_days+1))
    ax[1].set_xlabel("Insolation [MJ]")
    ax[1].set_xlim(0)
    plt.savefig("solar_flux.png", bbox_inches='tight')
    plt.show()
    plt.close()

