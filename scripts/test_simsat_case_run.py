"""Focused case-runner tests: no network, Rust build, or real renderer needed.

Run: python -B -m unittest discover -s scripts -p test_simsat_case_run.py -v
"""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

import numpy as np
from netCDF4 import Dataset
from test_simsat_case_score import report_fixture

SPEC = importlib.util.spec_from_file_location('simsat_case_run', Path(__file__).with_name('simsat-case-run.py'))
case = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(case)


class TimeAndGridTests(unittest.TestCase):
    def test_time_requires_explicit_zone_and_exact_unique_match(self):
        with self.assertRaisesRegex(ValueError, 'explicit UTC'):
            case.parse_time('2026-09-04T19:30:00')
        target = case.parse_time('2026-09-04T12:30:00-07:00')
        self.assertEqual(case.select_wrf_time(['2026-09-04_18:30:00', '2026-09-04_19:30:00'], target), 1)
        with self.assertRaisesRegex(ValueError, 'exactly one'):
            case.select_wrf_time(['2026-09-04_18:30:00'], target)
        with self.assertRaisesRegex(ValueError, 'requested'):
            case.select_wrf_time(['2026-09-04_18:30:00', '2026-09-04_19:30:00'], target, 0)
        with self.assertRaisesRegex(ValueError, 'found 2'):
            case.select_wrf_time(['2026-09-04_19:30:00'] * 2, target)

    def test_wrf_times_and_native_grid_boundary_convention(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / 'prepared-wrf'
            with Dataset(source, 'w') as d:
                for name, size in [('Time', 2), ('DateStrLen', 19), ('south_north', 5), ('west_east', 7)]:
                    d.createDimension(name, size)
                times = d.createVariable('Times', 'S1', ('Time', 'DateStrLen'))
                times[:] = np.array([list('2026-09-04_18:30:00'), list('2026-09-04_19:30:00')], dtype='S1')
                lat = np.repeat((30 + np.arange(5) * .01)[:, None], 7, axis=1).astype('f4')
                lon = np.repeat((-100 + np.arange(7) * .01)[None, :], 5, axis=0).astype('f4')
                for name, values in [('XLAT', lat), ('XLONG', lon), ('LANDMASK', np.ones_like(lat))]:
                    d.createVariable(name, 'f4', ('Time', 'south_north', 'west_east'))[:] = np.stack((values, values))
                d.MAP_PROJ, d.STAND_LON, d.TRUELAT1, d.TRUELAT2 = 6, -100., 30., 60.
                d.TITLE = 'HRRR-prepared WRF test fixture'
            args = case.parser().parse_args(['--input', str(source), '--time', '2026-09-04T19:30Z', '--out', str(root/'out'), '--bin', str(root)])
            metadata, grid, _ = case.prepare_input(args, root / 'target-grid.npz')
            self.assertEqual(metadata['timestep'], 1)
            self.assertEqual(metadata['kind'], 'wrf-netcdf')
            self.assertIn('HRRR-prepared', metadata['source_attributes']['TITLE'])
            self.assertEqual(grid['lat'].shape, (5, 7))
            self.assertGreater(float(grid['lat'][0, 3]), float(grid['lat'][-1, 3]))
            self.assertEqual(float(grid['lat'][2, 3]), float(lat[2, 3]))
            # The map boundary is deliberately inset; raw XLAT[::-1] is not substituted.
            self.assertLess(float(grid['lat'][0, 3]), float(lat[-1, 3]))
            self.assertEqual(metadata['geometry']['boundary_inset_cells'], .01)
            args.time = '2026-09-04T20:30Z'
            with self.assertRaisesRegex(ValueError, 'exactly one'):
                case.prepare_input(args, root / 'bad-grid.npz')
            self.assertFalse((root / 'bad-grid.npz').exists())

    def test_lambert_adapter_matches_independent_proj_coordinates(self):
        from pyproj import Proj
        p = {'map_proj': 1, 'truelat1': 30., 'truelat2': 60., 'stand_lon': -97.5, 'cen_lat': 39., 'dx': 3000., 'dy': 3000.}
        adapter = case.ModelProjection(np, p)
        independent = Proj('+proj=lcc +lat_1=30 +lat_2=60 +lat_0=90 +lon_0=-97.5 +R=6370000 +units=m')
        lat, lon = np.array([12., 39., 50.]), np.array([-88., -97.5, -110.])
        u, v = adapter.forward(lat, lon)
        expected_u, expected_v = independent(lon, lat)
        np.testing.assert_allclose(u, expected_u, atol=1e-7, rtol=0)
        np.testing.assert_allclose(v, expected_v, atol=1e-7, rtol=0)
        actual_lat, actual_lon = adapter.inverse(u, v)
        np.testing.assert_allclose(actual_lat, lat, atol=1e-12)
        np.testing.assert_allclose(actual_lon, lon, atol=1e-12)


class ReferenceReuseTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.reference = self.root / 'abi-reference-aligned.npz'
        params = {'map_proj':6,'truelat1':30.,'truelat2':60.,'stand_lon':-100.,'cen_lat':30.}
        self.projection = case.ModelProjection(np,params)
        self.target = {'lat':np.array([[31.,31.],[30.,30.]],np.float32),
                       'lon':np.array([[-100.,-99.],[-100.,-99.]],np.float32)}
        np.savez_compressed(self.reference, **self.target)
        self.score = case.helpers()
        self.manifest = {'target_time':'2026-09-04T19:30:00Z','bucket':'noaa-goes19','sector':'conus','platform':'G19',
                         'objects':[{'product':'ABI-L2-MCMIPC','key':'ABI-L2-MCMIPC/2026/OR_ABI-L2-MCMIPC-M6_G19_s2026247.nc'}]}
        self.args = case.parser().parse_args(['--input','unused','--time','2026-09-04T19:30Z','--out','unused','--bin','unused'])
        self.write_bundle(self.manifest)

    def write_bundle(self, manifest, **companion_changes):
        self.score.write_json(self.root/'source-manifest.json', manifest)
        companion = {'source_manifest':'source-manifest.json', 'source_manifest_sha256':self.score.sha256(self.root/'source-manifest.json'),
                     'alignment':{'sha256':self.score.sha256(self.reference)}}
        companion.update(companion_changes)
        self.score.write_json(self.reference.with_suffix('.json'), companion)

    def verify(self, target=None):
        return case.verify_reference(self.args, self.reference, self.target if target is None else target, self.projection, {'dx':1.,'dy':1.})

    def test_bundle_time_sector_hash_and_grid_are_all_checked(self):
        self.assertEqual(self.verify()['grid_max_offset_cells'],0)
        for key, wrong, expected in [('target_time','2026-09-04T20:30Z','target_time'),('sector','full-disk','sector'),
                                     ('bucket','noaa-goes18','satellite'),('platform','G18','platform')]:
            self.write_bundle(dict(self.manifest, **{key:wrong}))
            with self.assertRaisesRegex(ValueError,expected):
                self.verify()
        self.write_bundle(self.manifest)
        displaced = {key:values.copy() for key,values in self.target.items()}
        displaced['lon'] += .1
        with self.assertRaisesRegex(ValueError,'mesh differs'):
            self.verify(displaced)
        self.write_bundle(self.manifest, alignment={'sha256':'wrong'})
        with self.assertRaisesRegex(ValueError,'hash differs'):
            self.verify()

    def test_companion_must_bind_exact_manifest_name_and_bytes(self):
        # Replacing the trusted label with another valid manifest cannot relabel this NPZ.
        self.score.write_json(self.root/'source-manifest.json',dict(self.manifest,target_time='2026-09-04T20:30Z'))
        with self.assertRaisesRegex(ValueError,'name/hash binding'):
            self.verify()
        self.write_bundle(self.manifest, source_manifest='unrelated.json')
        with self.assertRaisesRegex(ValueError,'name/hash binding'):
            self.verify()

    def test_legacy_sector_is_inferred_only_from_unambiguous_mcmip(self):
        for code, sector in [('C','conus'),('F','full-disk'),('M1','meso1'),('M2','meso2')]:
            legacy = {key:value for key,value in self.manifest.items() if key != 'sector'}
            legacy['objects'] = [{'product':'ABI-L2-MCMIP' + (code if code in ('C','F') else 'M'),
                                  'key':f'ABI-L2-MCMIP{code}/OR_ABI-L2-MCMIP{code}-M6_G19_s2026247.nc'}]
            self.write_bundle(legacy)
            self.args.sector = sector
            result = self.verify()
            self.assertEqual(result['sector'],sector)
            self.assertIn('inferred',result['sector_source'])
        for objects in [[],[{'product':'ABI-L2-MCMIPM'}],[{'product':'ABI-L2-MCMIPC','key':'ABI-L2-MCMIPF/foo'}]]:
            self.write_bundle(dict(legacy,objects=objects))
            with self.assertRaisesRegex(ValueError,'ambiguous'):
                self.verify()

    def test_selected_object_platform_must_match_manifest_bucket(self):
        objects = [{'product':'ABI-L2-MCMIPC','key':'ABI-L2-MCMIPC/OR_ABI-L2-MCMIPC-M6_G18_s2026247.nc'}]
        self.write_bundle(dict(self.manifest, objects=objects))
        with self.assertRaisesRegex(ValueError,'object platform'):
            self.verify()


class OrchestrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / 'input-wrf'
        self.source.write_bytes(b'input fixture')
        self.bin = self.root / 'bin'
        self.bin.mkdir()
        for name in ('simsat-render-ir.exe', 'simsat-render-frame.exe'):
            (self.bin/name).write_bytes(b'never execute fixture binary')
        self.args = case.parser().parse_args(['--input', str(self.source), '--time', '2026-09-04T19:30Z',
            '--out', str(self.root/'output'), '--bin', str(self.bin), '--scoreboard', str(self.root/'scoreboard.md')])
        self.calls = []

    def prepare(self, args, target):
        target.write_bytes(b'grid fixture')
        return {'kind': 'wrf-netcdf', 'input_provenance': 'HRRR-prepared WRF fixture', 'timestep': 2,
                'geometry': {}, 'valid_time': '2026-09-04T19:30:00Z'}, {}, None

    def verify(self, args, ref, arrays, projection, geometry):
        return {'path': str(ref), 'sha256': case.helpers().sha256(ref)}

    def runner(self, command, stdout, **kwargs):
        self.calls.append(command)
        self.assertEqual(kwargs['env']['RAYON_NUM_THREADS'], str(self.args.threads))
        if '--download' in command:
            dest = Path(command[command.index('--output-dir') + 1])
            dest.mkdir()
            (dest/'abi-reference-aligned.npz').write_bytes(b'aligned fixture')
        elif '--product' in command:
            product = command[command.index('--product') + 1]
            dest = Path(command[command.index('--output-dir') + 1])
            dest.mkdir()
            case.helpers().write_json(dest/'validation.json', report_fixture(product))
        else:
            opts = dict(part.split('=', 1) for part in command[1:])
            for key in ('out', 'bt-out', 'cloud-mask-out', 'rgb-reflectance-out'):
                if key in opts:
                    Path(opts[key]).write_bytes(b'render fixture')
        stdout.write('fixture stage completed\n')
        return subprocess.CompletedProcess(command, 0)

    def execute(self, runner=None):
        with contextlib.redirect_stdout(io.StringIO()):
            return case.run(self.args, runner or self.runner, self.prepare, self.verify)

    def test_day_command_contracts_and_cloud_proxy_disclosure(self):
        scores = self.execute()
        self.assertEqual(set(scores), {'ir','vis','vis_sensor_fast_gray'})
        fetch = self.calls[0]
        for flag in ('--download','--align','--preview','--target-grid','--sector','--satellite'):
            self.assertIn(flag, fetch)
        ir = next(c for c in self.calls if c[0].endswith('simsat-render-ir.exe'))
        self.assertIn('threads=6', ir)
        self.assertIn('view=topdown', ir)
        self.assertIn('sensor=goes-r-abi-band13-fm4', ir)
        self.assertIn('timestep=2', ir)
        self.assertTrue(any(arg.startswith('cloud-mask-out=') for arg in ir))
        visible = [c for c in self.calls if c[0].endswith('simsat-render-frame.exe')]
        self.assertEqual(len(visible), 2)
        for command in visible:
            for option in ('view=geo','raster=model-grid','resolution=native','geo-navigation=goes-r-abi','threads=6'):
                self.assertIn(option, command)
        validations = [c for c in self.calls if '--product' in c]
        self.assertEqual(len(validations), 3)
        self.assertTrue(all('--synthetic-cloud-mask' in c for c in validations))
        self.assertIn('f32le-rgb-unclipped', validations[-1])
        manifest = case.helpers().read_json(self.args.out/'run.json')
        self.assertEqual(manifest['status'], 'complete')
        self.assertEqual(manifest['scoreboard_update'], 'appended')
        self.assertIn('neighboring columns', manifest['visible_cloud_regime_limitation'])
        self.assertIn('HRRR-prepared WRF fixture', self.args.scoreboard.read_text())
        self.assertFalse(self.args.scoreboard.with_suffix('.md.lock').exists())

    def test_night_ir_skips_visible_and_can_reuse_reference_bundle(self):
        self.args.night_ir = True
        (self.bin/'simsat-render-frame.exe').unlink()
        reference = self.root/'already-aligned.npz'
        reference.write_bytes(b'aligned fixture')
        self.args.reference = reference
        scores = self.execute()
        self.assertEqual(set(scores), {'ir'})
        self.assertFalse(any('--download' in c for c in self.calls))
        self.assertFalse(any('simsat-render-frame' in c[0] for c in self.calls))
        self.assertIn('visible rendering and scoring deliberately skipped', self.args.scoreboard.read_text())

    def test_later_failure_preserves_ir_checkpoint_and_does_not_append(self):
        def fail_visible(command, stdout, **kwargs):
            if command[0].endswith('simsat-render-frame.exe'):
                stdout.write('fixture visible failure\n')
                return subprocess.CompletedProcess(command, 23)
            return self.runner(command, stdout, **kwargs)
        with self.assertRaisesRegex(RuntimeError, 'exit 23'):
            self.execute(fail_visible)
        self.assertIn('ir', case.helpers().read_json(self.args.out/'scores.json'))
        manifest = case.helpers().read_json(self.args.out/'run.json')
        self.assertEqual(manifest['status'], 'failed')
        self.assertEqual(manifest['stages'][-1]['returncode'], 23)
        self.assertFalse(self.args.scoreboard.exists())

    def test_goes18_requires_explicit_response_approximation_and_records_it(self):
        self.args.satellite = 'goes18'
        with self.assertRaisesRegex(ValueError,'no matched FM3'):
            self.execute()
        self.assertFalse(self.args.out.exists())
        self.args.allow_goes18_fm4_approximation = True
        self.execute()
        self.assertIn('APPROXIMATION',self.args.scoreboard.read_text())
        manifest = case.helpers().read_json(self.args.out/'run.json')
        self.assertIn('not a matched GOES18',manifest['ir_response_note'])
        ir = next(c for c in self.calls if c[0].endswith('simsat-render-ir.exe'))
        self.assertIn('sat=goes-west',ir)

    def test_existing_output_is_never_touched(self):
        self.args.out.mkdir()
        marker = self.args.out/'scores.json'
        marker.write_text('baseline marker')
        with self.assertRaises(FileExistsError):
            self.execute()
        self.assertEqual(marker.read_text(), 'baseline marker')
        self.assertEqual(self.calls, [])


class ScoreboardTests(unittest.TestCase):
    def test_duplicate_row_is_idempotent_but_conflicting_scores_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'scoreboard.md'
            metadata = {'valid_time':'2026-09-04T19:30Z','label':'test','input_kind':'wrf-netcdf',
                        'input_provenance':'prepared WRF','output':'case-output','satellite':'goes19','sector':'conus','night_ir':True}
            scores = {'ir':{'clear_bias':-2.,'both_cloudy_bias':8.,'cloudy_bias':10.,'bias':4.,'corr':.2}}
            self.assertEqual(case.update_scoreboard(path,'identity',metadata,scores), 'appended')
            original = path.read_bytes()
            self.assertEqual(case.update_scoreboard(path,'identity',metadata,scores), 'already-recorded')
            self.assertEqual(original,path.read_bytes())
            scores['ir']['bias'] = 5
            with self.assertRaisesRegex(ValueError,'different scores'):
                case.update_scoreboard(path,'identity',metadata,scores)
            self.assertEqual(original,path.read_bytes())
            self.assertFalse(path.with_suffix('.md.lock').exists())

    def test_another_writer_lock_is_never_removed(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'scoreboard.md'
            lock = path.with_suffix('.md.lock')
            lock.write_text('another writer')
            with self.assertRaises(FileExistsError):
                case.update_scoreboard(path,'identity',{}, {})
            self.assertEqual(lock.read_text(),'another writer')


if __name__ == '__main__':
    unittest.main()
