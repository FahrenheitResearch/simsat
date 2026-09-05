"""Exporter boundary checks for a measured RGB display source; no network."""
import datetime, hashlib, importlib.util, json
from pathlib import Path
import tempfile, unittest
import numpy as np

spec=importlib.util.spec_from_file_location('nbar_export',Path(__file__).with_name('simsat-export-nbar-surface.py'))
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

class NbarExportTests(unittest.TestCase):
    def fixture(self,p):
        p.mkdir()
        rgb=np.broadcast_to(np.array([0.12,0.7,0.05,0.16,0.4,0.3,0.2]),(2,3,7)).copy()
        qa=np.zeros((2,3,7),dtype='u1')
        qa[0,1,0]=1
        rgb[0,2,:]=np.nan
        rgb[1,1,0]=1.2
        qa[1,2,2]=255
        source=p/'aligned-nbar.npz'
        np.savez(source,latitude=[[1,1,1],[0,0,0]],longitude=[[0,1,2],[0,1,2]],
            land_mask=[[1,1,1],[0,1,1]],band=np.arange(1,8),nbar=rgb,mandatory_quality=qa)
        provenance=dict(quantity='MODIS_7_band_nadir_BRDF_adjusted_surface_reflectance',
            date='2024-04-03',output_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
        (p/'provenance.json').write_text(json.dumps(provenance),encoding='utf-8')
        return p

    def test_band_mapping_preserves_source_values_orientation_quality_and_explicit_fallback(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp);src=self.fixture(p/'source');out=p/'out'
            h=module.export(src,out,datetime.date(1974,4,3),'full-and-magnitude')
            self.assertEqual(h['counts'],dict(full=1,magnitude=1,fallback=3))
            self.assertEqual(h['rgb_bands'],[1,4,3])
            rgb=np.fromfile(out/'nbar-rgb.bin',dtype='<f4').reshape(2,3,3)
            np.testing.assert_allclose(rgb[0,0],[.12,.16,.05])
            self.assertTrue(np.isnan(rgb[0,2]).all())
            self.assertGreater(rgb[1,1,0],1) # preserve source, do not clip to albedo
            coords=np.fromfile(out/'coordinates.bin',dtype='<f8').reshape(2,3,2)
            np.testing.assert_array_equal(coords[:,:,0],[[1,1,1],[0,0,0]])
            for name,record in h['files'].items():
                self.assertEqual(hashlib.sha256((out/name).read_bytes()).hexdigest(),record['sha256'])
            strict=module.export(src,p/'strict',datetime.date(1974,4,3),'full-only')
            self.assertEqual(strict['counts'],dict(full=1,magnitude=0,fallback=4))
            with self.assertRaises(FileExistsError):
                module.export(src,out,datetime.date(1974,4,3),'full-only')

    def test_wrong_date_and_corrupt_source_fail_before_output(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp);src=self.fixture(p/'source');out=p/'out'
            with self.assertRaisesRegex(ValueError,'calendar'):
                module.export(src,out,datetime.date(1974,4,4),'full-only')
            self.assertFalse(out.exists())
            with (src/'aligned-nbar.npz').open('ab') as f:f.write(b'corrupt')
            with self.assertRaisesRegex(ValueError,'checksum'):
                module.export(src,out,datetime.date(1974,4,3),'full-only')
            self.assertFalse(out.exists())

if __name__=='__main__':unittest.main()
