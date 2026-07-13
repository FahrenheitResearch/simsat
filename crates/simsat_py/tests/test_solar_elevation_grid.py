"""Focused installed-extension tests for ``simsat.solar_elevation_grid``."""

from __future__ import annotations

import unittest

import numpy as np
import simsat


TIME = "1974-04-03T23:12:00Z"
LAT = np.array(
    [[35.0, 35.0, 35.0], [37.5, 37.5, 37.5], [40.0, 40.0, 40.0]],
    dtype=np.float32,
)
LON = np.array(
    [[-95.0, -92.5, -90.0], [-95.0, -92.5, -90.0], [-95.0, -92.5, -90.0]],
    dtype=np.float32,
)


class SolarElevationGridTests(unittest.TestCase):
    def test_noaa_center_and_corners_are_pinned(self) -> None:
        got = simsat.solar_elevation_grid(TIME, LAT, LON)
        self.assertEqual(got.shape, LAT.shape)
        self.assertEqual(got.dtype, np.dtype(np.float32))
        np.testing.assert_array_equal(
            got,
            np.array(
                [
                    [17.741222, 15.708194, 13.673027],
                    [17.470839, 15.504449, 13.535097],
                    [17.166819, 15.270934, 13.3712845],
                ],
                dtype=np.float32,
            ),
        )

    def test_full_and_partial_overrides_use_the_same_center_path(self) -> None:
        full = simsat.solar_elevation_grid(
            TIME, LAT, LON, sun_elev=12.5, sun_az=210.0
        )
        partial = simsat.solar_elevation_grid(TIME, LAT, LON, sun_elev=5.0)
        self.assertEqual(float(full[1, 1]), 12.5)
        self.assertEqual(float(partial[1, 1]), 5.0)
        np.testing.assert_array_equal(
            full[[0, 0, 2, 2], [0, 2, 0, 2]],
            np.array([15.665485, 13.599671, 11.254318, 9.354468], np.float32),
        )

    def test_invalid_pixels_are_nan(self) -> None:
        lat = LAT.copy()
        lon = LON.copy()
        lat[0, 0] = np.nan
        lon[2, 2] = np.nan
        got = simsat.solar_elevation_grid(TIME, lat, lon)
        self.assertTrue(np.isnan(got[0, 0]))
        self.assertTrue(np.isnan(got[2, 2]))
        self.assertTrue(np.isfinite(got[1, 1]))

    def test_malformed_time_and_shapes_raise_value_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "time_iso"):
            simsat.solar_elevation_grid("not-a-time", LAT, LON)
        with self.assertRaisesRegex(ValueError, "time_iso"):
            simsat.solar_elevation_grid("1974-02-30T23:12:00Z", LAT, LON)
        with self.assertRaisesRegex(ValueError, "time_iso"):
            simsat.solar_elevation_grid("1974-04-03T23:12:oops", LAT, LON)
        with self.assertRaisesRegex(ValueError, "same shape"):
            simsat.solar_elevation_grid(TIME, LAT, LON[:, :2])
        with self.assertRaisesRegex(ValueError, "non-zero"):
            empty = np.empty((0, 3), dtype=np.float32)
            simsat.solar_elevation_grid(TIME, empty, empty)
        with self.assertRaisesRegex(ValueError, "no finite coordinate pair"):
            invalid = np.full((2, 2), np.nan, dtype=np.float32)
            simsat.solar_elevation_grid(TIME, invalid, invalid)
        with self.assertRaisesRegex(ValueError, "sun_elev"):
            simsat.solar_elevation_grid(TIME, LAT, LON, sun_elev=float("nan"))


if __name__ == "__main__":
    unittest.main()
