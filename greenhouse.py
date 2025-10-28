#!/usr/bin/python3
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def solar_angles(start_date: dt.datetime, lat: float, lon: float):
    """
    Calculate the solar position every 6 minutes for a year at a location, starting on `start_date`.
    """
    assert -90 <= lat <= 90
    assert -180 <= lon <= 180
    
    date = dt.date(start_date.year, start_date.month, start_date.day)
    
    tz = start_date.tzinfo
    if tz is None:
        time_shift = 0
    else:
        time_shift = tz.utcoffset(start_date).total_seconds() / 3600  # local time zone shift, in hours
    julian_date = (date - dt.date(1900, 1, 1)).days /  + 2415018.5 - (time_shift / 24)

    times = np.arange(1, 10 * 24 * 365 + 1, dtype=float) * 6 / 1440 # every 6 minutes, every day, for a year

    timestamps = times + julian_date
    julian_century = (timestamps - 2451545) / 36525
    
    geometric_mean_longitude_sun_deg = np.fmod(280.46646 + julian_century*(36000.76983 + julian_century * 0.0003032), 360.0)
    geometric_mean_anomaly_sun_deg = 357.52911 + julian_century * (35999.05029 - 0.0001537 * julian_century)
    earth_orbit_eccentricity = 0.016708634 - julian_century * (0.000042037 + 0.0000001267 * julian_century)
    sun_equation_of_center = (
        np.sin(np.deg2rad(geometric_mean_anomaly_sun_deg)) * (
            1.914602 - julian_century * (0.004817 + 0.000014 * julian_century)
        ) + np.sin(np.deg2rad(2 * geometric_mean_anomaly_sun_deg)) * (
            0.019993 - 0.000101 * julian_century
        ) + np.sin(np.deg2rad(3 * geometric_mean_anomaly_sun_deg)) * 0.000289
    )
    sun_true_longitude_deg = geometric_mean_longitude_sun_deg + sun_equation_of_center
    sun_true_anomaly_deg = geometric_mean_anomaly_sun_deg + sun_equation_of_center
    sun_rad_vector_au = (1.000001018 * (1 - earth_orbit_eccentricity * earth_orbit_eccentricity)) / (1 + earth_orbit_eccentricity * np.cos(np.deg2rad(sun_true_anomaly_deg)))
    sun_apparent_long_deg = sun_true_longitude_deg - 0.00569 - 0.00478 * np.sin(np.deg2rad(125.04 - 1937.136 * julian_century))
    mean_oblique_ecliptic_deg = 23 + (26 + ((21.448 - julian_century * (46.815 + julian_century * (0.00059 - julian_century * 0.001813)))) / 60 ) / 60
    oblique_correction_deg = mean_oblique_ecliptic_deg + 0.00256 * np.cos(np.deg2rad(125.04 - 1934.136 * julian_century))
    sun_right_ascension_deg = np.rad2deg(np.arctan2(np.cos(np.deg2rad(sun_apparent_long_deg)), np.cos(np.deg2rad(oblique_correction_deg)) * np.sin(np.deg2rad(sun_apparent_long_deg))))
    sun_declination_deg = np.rad2deg(np.arcsin(np.sin(np.deg2rad(oblique_correction_deg)) * np.sin(np.deg2rad(sun_apparent_long_deg))))
    var_y = np.tan(np.deg2rad(oblique_correction_deg / 2.0)) * np.tan(np.deg2rad(oblique_correction_deg / 2.0))
    eq_of_time_min = 4*np.rad2deg(var_y*np.sin(2*np.deg2rad(geometric_mean_longitude_sun_deg))-2*earth_orbit_eccentricity*np.sin(np.deg2rad(geometric_mean_anomaly_sun_deg))+4*earth_orbit_eccentricity*var_y*np.sin(np.deg2rad(geometric_mean_anomaly_sun_deg))*np.cos(2*np.deg2rad(geometric_mean_longitude_sun_deg))-0.5*var_y*var_y*np.sin(4*np.deg2rad(geometric_mean_longitude_sun_deg))-1.25*earth_orbit_eccentricity*earth_orbit_eccentricity*np.sin(2*np.deg2rad(geometric_mean_anomaly_sun_deg)))
    ha_sunrise_deg = np.rad2deg(np.arccos(np.cos(np.deg2rad(90.833)) / (np.cos(np.deg2rad(lat)) * np.cos(np.deg2rad(sun_declination_deg))) - np.tan(np.deg2rad(lat)) * np.tan(np.deg2rad(sun_declination_deg))))
    solar_noon_lst = (720 - 4 * lon - eq_of_time_min + time_shift * 60) / 1440
    sunrise_time_lst = solar_noon_lst - ha_sunrise_deg * 4 / 1440
    sunset_time_lst = solar_noon_lst + ha_sunrise_deg * 4 / 1440
    sunrise_duration_min = 8 * ha_sunrise_deg

    true_solar_time_min = np.fmod(np.fmod(times, 1.0) * 1440 + eq_of_time_min + 4 * lon - 60 * time_shift, 1440.0)
    hour_angle_deg = np.zeros(true_solar_time_min.shape, dtype=float)
    mask = true_solar_time_min / 4 < 0
    hour_angle_deg[mask] = true_solar_time_min[mask] / 4 + 180.0
    hour_angle_deg[~mask] = true_solar_time_min[~mask] / 4 - 180.0
    solar_zenith_angle_deg = np.rad2deg(np.arccos(np.sin(np.deg2rad(lat)) * np.sin(np.deg2rad(sun_declination_deg)) + np.cos(np.deg2rad(lat)) * np.cos(np.deg2rad(sun_declination_deg)) * np.cos(np.deg2rad(hour_angle_deg))))
    solar_elevation_angle_deg = 90.0 - solar_zenith_angle_deg
    
    atmospheric_refraction_deg = np.zeros(solar_elevation_angle_deg.shape, dtype=float)
    low_elv_mask = np.logical_and(solar_elevation_angle_deg > 5.0, solar_elevation_angle_deg < 85.0)
    atmospheric_refraction_deg[low_elv_mask] = 58.1 / np.tan(np.deg2rad(solar_elevation_angle_deg[low_elv_mask])) - 0.07 / np.power(np.tan(np.deg2rad(solar_elevation_angle_deg[low_elv_mask])), 3) + 0.000086 / np.power(np.tan(np.deg2rad(solar_elevation_angle_deg[low_elv_mask])), 5)
    near_horizon_mask = np.logical_and(solar_elevation_angle_deg > -0.575, solar_elevation_angle_deg < 5.0)
    atmospheric_refraction_deg[near_horizon_mask] = 1735 + solar_elevation_angle_deg[near_horizon_mask] * (-518.2 + solar_elevation_angle_deg[near_horizon_mask]* (103.4 + solar_elevation_angle_deg[near_horizon_mask] * (-12.79 + solar_elevation_angle_deg[near_horizon_mask] * 0.711)))
    atmospheric_refraction_deg[solar_elevation_angle_deg < -0.575] = -20.772 / np.tan(np.deg2rad(solar_elevation_angle_deg[solar_elevation_angle_deg < -0.575]))
    atmospheric_refraction_deg /= 3600.0

    solar_elv_corrected_deg = atmospheric_refraction_deg + solar_elevation_angle_deg
    solar_az_deg = np.zeros(solar_elv_corrected_deg.shape, dtype=float)
    mask = hour_angle_deg > 0.0
    solar_az_deg[mask] = np.rad2deg(np.arccos(((np.sin(np.deg2rad(lat))*np.cos(np.deg2rad(solar_zenith_angle_deg[mask]))) - np.sin(np.deg2rad(sun_declination_deg[mask]))) / (np.cos(np.deg2rad(lat)) * np.sin(np.deg2rad(solar_zenith_angle_deg[mask]))))) + 180
    solar_az_deg[~mask] = 540 - np.rad2deg(np.arccos(((np.sin(np.deg2rad(lat)) * np.cos(np.deg2rad(solar_zenith_angle_deg[~mask]))) - np.sin(np.deg2rad(sun_declination_deg[~mask]))) / (np.cos(np.deg2rad(lat)) * np.sin(np.deg2rad(solar_zenith_angle_deg[~mask])))))
    solar_az_deg = np.fmod(solar_az_deg, 360.0)
    
    return solar_elv_corrected_deg, solar_az_deg


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
    el, az = solar_angles(dt.datetime(2025, 1, 1, tzinfo=dt.timezone(-dt.timedelta(hours=6))), 52.1093, -106.59)
    assert np.allclose(az, az_el[:, 1]) is True, f"az doesn't match, max_diff {np.max(np.abs(np.rad2deg(np.angle(np.exp(1j * np.deg2rad(az - az_el[:, 1]))))))}"
    assert np.allclose(el, az_el[:, 0]) is True, f"el doesn't match, max_diff {np.max(np.abs(el - az_el[:, 0]))}"
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

