import importlib.util
from pathlib import Path
import unittest
import numpy as np

spec = importlib.util.spec_from_file_location('material_prep', Path(__file__).with_name('simsat-prepare-material-indices.py'))
prep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prep)

class MaterialPreparationTests(unittest.TestCase):
    def fixture(self):
        return '\n'.join(f'      DATA ({name}(I), I = 1, 3 ) /\n     &{values}/'
                         for name, values in [('WLTAB', '.39,.5,1.1'), ('REREF', '1.34,1.33,1.32'), ('IMREF', '1D-10,2D-9,3D-7')])

    def test_fortran_data_is_parsed_as_numbers_and_retains_boundary_knots(self):
        data = prep.parse_water(self.fixture(), count=3)
        np.testing.assert_allclose(data[:, 2], [1e-10, 2e-9, 3e-7])
        np.testing.assert_array_equal(prep.visible_subset(data), data)

    def test_parser_rejects_missing_overlapping_and_nonpositive_data(self):
        for source in [self.fixture().replace('DATA (IMREF', 'DATA (UNKNOWN'),
                       self.fixture()+'\n'+self.fixture(),
                       self.fixture().replace('1D-10', '-1D-10')]:
            with self.subTest(source=source):
                with self.assertRaises(ValueError):
                    prep.parse_water(source, count=3)

    def test_subset_rejects_extrapolation_unsorted_and_nonfinite_data(self):
        data = prep.parse_water(self.fixture(), count=3)
        corruptions = [data[1:], data[::-1], data.copy()]
        corruptions[-1][1, 2] = np.nan
        for bad in corruptions:
            with self.subTest(data=bad):
                with self.assertRaises(ValueError):
                    prep.visible_subset(bad)

if __name__ == '__main__':
    unittest.main()
