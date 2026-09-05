"""Surface-input invariants; no network or observational radiance fitting."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from netCDF4 import Dataset

SPEC=importlib.util.spec_from_file_location('spectral_surface', Path(__file__).with_name('simsat-prepare-spectral-surface.py'))
surface=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(surface)


class SpectralSurfaceTests(unittest.TestCase):
    def test_affine_spectra_and_target_orientation(self):
        y=np.array([0.,1.,2.]);x=np.array([10.,11.,12.])
        values=np.stack([0.1 + 0.01*y[:,None] + 0.02*(x[None,:]-10),
                         0.4 + 0.03*y[:,None] - 0.01*(x[None,:]-10)],axis=-1)
        lat=np.array([[1.75,1.25],[0.25,0.75]]);lon=np.array([[10.5,11.5],[11.25,10.75]])
        a,valid=surface.resample_spectra(y,x,values,lat,lon)
        np.testing.assert_allclose(a[...,0],0.1+0.01*lat+0.02*(lon-10),rtol=1e-6)
        np.testing.assert_allclose(a[...,1],0.4+0.03*lat-0.01*(lon-10),rtol=1e-6)
        b,v2=surface.resample_spectra(y[::-1],x[::-1],values[::-1,::-1],lat,lon)
        np.testing.assert_array_equal(a,b);np.testing.assert_array_equal(valid,v2)
        self.assertTrue(valid.all())

    def test_invalid_and_ocean_corners_cannot_be_filled_or_clipped(self):
        for bad in ([0.,0.], [0.2,np.nan], [0.2,1.01], [-0.1,0.2]):
            values=np.full((2,2,2),0.2);values[1,1]=bad
            a,v=surface.resample_spectra([0,1],[0,1],values,[[0.25]],[[0.25]])
            self.assertFalse(v.any());self.assertTrue(np.isnan(a).all())

    def test_outside_and_nonfinite_targets_are_missing(self):
        a,v=surface.resample_spectra([0,1],[0,1],np.full((2,2,2),0.2),
                                   [[-0.1,1.1,np.nan,1]],[[0,0,0,1]])
        np.testing.assert_array_equal(v,[[0,0,0,1]])
        self.assertTrue(np.isnan(a[0,:3]).all());np.testing.assert_allclose(a[0,3],0.2)

    def test_malformed_coordinate_and_shape_rejected(self):
        for axis in ([0,0],[0,np.nan],[0,2,1]):
            with self.assertRaises(ValueError):surface.ascending_axis(axis,'test')
        with self.assertRaisesRegex(ValueError,'shape'):
            surface.resample_spectra([0,1],[0,1],np.ones((3,3,2)),[[0]],[[0]])

    def fixture(self, folder):
        src=folder/'source.nc';grid=folder/'grid.npz'
        with Dataset(src,'w') as f:
            for k,values in [('Latitude',[2.,1.,0.]),('Longitude',[10.,11.,12.]),('Wavelength',[470.,640.,860.])]:
                f.createDimension(k,3);f.createVariable(k,'f4',(k,))[:]=values
            a=f.createVariable('Black_Sky_Albedo','f4',('Latitude','Longitude','Wavelength'))
            a[:]=np.broadcast_to([0.1,0.2,0.4],(3,3,3));f.source_doy=247
        # Hostile unrelated observation arrays must have no effect or be read.
        np.savez(grid,lat=[[1.5,1.0],[0.5,0.0]],lon=[[10.5,11],[10.5,11]],
                 land_mask=[[1,0],[1,1]],cmi_c01=np.zeros((7,7)),valid=np.zeros((7,7)),bcm=np.zeros((7,7)))
        return src,grid

    def test_file_pipeline_preserves_units_spectra_and_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);src,grid=self.fixture(p)
            r=surface.prepare(src,grid,p/'out',247)
            self.assertEqual(r['valid_count'],3)
            self.assertEqual(r['target_grid']['fields_used'],['lat','lon','land_mask'])
            with np.load(p/'out/spectral-surface-aligned.npz') as a:
                np.testing.assert_allclose(a['wavelength_um'],[0.47,0.64,0.86])
                np.testing.assert_allclose(a['black_sky_albedo'][a['valid']==1],np.broadcast_to([0.1,0.2,0.4],(3,3)))
                self.assertTrue(np.isnan(a['black_sky_albedo'][0,1]).all())
                self.assertFalse(any(k.startswith('cmi') for k in a.files))
                self.assertEqual(a['latitude'][0,0],1.5)
            self.assertEqual(r['output_sha256'],surface.sha256(p/'out/spectral-surface-aligned.npz'))
            with self.assertRaisesRegex(ValueError,'new'):surface.prepare(src,grid,p/'out',247)

    def test_land_mask_is_mandatory_and_binary(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);src,grid=self.fixture(p)
            np.savez(grid,lat=[[1.]],lon=[[11.]],land_mask=[[2]])
            with self.assertRaisesRegex(ValueError,'land_mask'):
                surface.prepare(src,grid,p/'out',247)
            np.savez(grid,lat=[[1.]],lon=[[11.]])
            with self.assertRaises(KeyError):surface.prepare(src,grid,p/'out',247)
            self.assertFalse((p/'out').exists())

    def test_wrong_day_and_empty_overlap_fail_without_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);src,grid=self.fixture(p)
            with self.assertRaisesRegex(ValueError,'day'):surface.prepare(src,grid,p/'out',246)
            self.assertFalse((p/'out').exists())
            np.savez(grid,lat=[[80.]],lon=[[170.]],land_mask=[[1]])
            with self.assertRaisesRegex(ValueError,'overlap'):surface.prepare(src,grid,p/'out',247)
            self.assertFalse((p/'out').exists())


if __name__=='__main__':unittest.main()
