"""Reference-sector and night-quality regressions; no network required."""
import argparse
import datetime as dt
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from netCDF4 import Dataset


def module(name, filename):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(filename))
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value

fetch = module("fetch_reference", "fetch-goes-abi-reference.py")
validate = module("reference_validator", "simsat-validate-goes.py")
NOW = dt.datetime(2026, 9, 4, 19, 30, tzinfo=dt.timezone.utc)


def item(product, region="", instant=NOW):
    return fetch.S3Object(product, f"{product}/OR_{product}{region}-M6_G19_test.nc", 0, "", "", instant, instant)


class ReferenceTests(unittest.TestCase):
    def test_sector_products_and_meso_cod(self):
        self.assertEqual(fetch.products_for_sector("full-disk"), ["ABI-L2-MCMIPF", "ABI-L2-ACMF", "ABI-L2-CODF"])
        self.assertEqual(fetch.products_for_sector("meso2"), ["ABI-L2-MCMIPM", "ABI-L2-ACMM", "ABI-L2-CODF"])

    def test_meso_selection_does_not_choose_nearer_other_sector(self):
        a = item("ABI-L2-MCMIPM", "1")
        b = item("ABI-L2-MCMIPM", "2", NOW + dt.timedelta(seconds=12))
        with patch.object(fetch, "list_prefix", return_value=[a, b]):
            selected = fetch.choose_objects("noaa-goes19", [a.product], NOW, "G19", 600, 1, "start", "meso2")
        self.assertEqual(selected, [b])

    def test_ambiguous_family_fails(self):
        with self.assertRaisesRegex(RuntimeError, "exactly one"):
            fetch.find_selected([item("ABI-L2-MCMIPF"), item("ABI-L2-MCMIPC")], "ABI-L2-MCMIP")

    def test_full_disk_night_alignment_and_thermal_quality(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            lat = np.array([[1., 1., 1.], [0., 0., 0.], [-1., -1., -1.]])
            lon = np.tile([-76., -75., -74.], (3, 1))
            np.savez(root / "grid.npz", lat=lat, lon=lon)
            selected = item("ABI-L2-MCMIPF")
            with Dataset(root / Path(selected.key).name, "w") as d:
                d.createDimension("x", 5); d.createDimension("y", 5)
                d.createVariable("x", "f8", ("x",))[:] = np.linspace(-.01, .01, 5)
                d.createVariable("y", "f8", ("y",))[:] = np.linspace(-.01, .01, 5)
                proj = d.createVariable("goes_imager_projection", "i4")
                proj.setncatts(dict(grid_mapping_name="geostationary", perspective_point_height=35786023.,
                                   semi_major_axis=6378137., semi_minor_axis=6356752.31414,
                                   longitude_of_projection_origin=-75., latitude_of_projection_origin=0., sweep_angle_axis="x"))
                for band in ("C01", "C02", "C03", "C13"):
                    v = d.createVariable("CMI_" + band, "f4", ("y", "x"), fill_value=-999.)
                    v[:] = 250. if band == "C13" else -999.
                    d.createVariable("DQF_" + band, "u1", ("y", "x"))[:] = 0
            args = argparse.Namespace(target_grid=root / "grid.npz", target_grib=None, output_dir=root,
                                      aligned_name="ref.npz", preview=False)
            fetch.align_reference(args, [selected])
            with np.load(root / "ref.npz") as f:
                arrays = dict(f)
            self.assertTrue(np.all(arrays["valid"] == 1))
            self.assertTrue(np.all(arrays["valid_visible"] == 0))
            self.assertTrue(np.all(arrays["valid_c13"] == 1))
            arrays["dqf_c13"][0, 0] = 1
            masks = validate.thermal_regime_masks(np, arrays, np.ones(lat.shape, dtype=bool))
            self.assertEqual(int(masks["valid"]["mask"].sum()), 8)

    def test_flagged_bilinear_neighbor_cannot_contaminate_good_nearest_pixel(self):
        axis = np.array([0., 1.]); target = np.array([[.1]])
        field = np.array([[.2, .2], [.2, 99.]])
        dqf = np.array([[0, 0], [0, 1]])
        self.assertEqual(float(fetch.sample_regular(np, dqf, axis, axis, target, target, nearest=True)[0, 0]), 0.)
        quality_field = np.where(dqf == 0, field, np.nan)
        self.assertTrue(np.isnan(fetch.sample_regular(np, quality_field, axis, axis, target, target)[0, 0]))

    def test_legacy_thermal_validity_unchanged(self):
        arrays = {"valid": np.array([[1, 0]]), "cmi_c13": np.array([[250., 250.]])}
        masks = validate.thermal_regime_masks(np, arrays, np.ones((1, 2), dtype=bool))
        np.testing.assert_array_equal(masks["valid"]["mask"], [[True, False]])

if __name__ == "__main__":
    unittest.main()
