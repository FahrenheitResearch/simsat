"""Reject corrupted, unconverged, or incorrectly paired scientific references."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

SCRIPT = Path(__file__).with_name('simsat-validate-cloud-slab.py')

class CloudSlabValidationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.simulated = {'schema': 'simsat-cloud-slab-source-audit-v1', 'limitations': ['fixture'],
                          'cases': [{'id': 0, 'within_lut_tau_mu_domain': True,
                                     'rho_f': {'candidate': 0.3},
                                     'reference_request': dict(tau=1, ssa=1, kind=1, g1=.85, g2=-.15,
                                                               weight=.9, gamma=0, mu0=.65, muv=1,
                                                               relative_azimuth_deg=0, albedo=0)}]}
        self.requests = '0 1 1 1 .85 -.15 .9 0 .65 1 0 0 32\n0 1 1 1 .85 -.15 .9 0 .65 1 0 0 68\n'
        self.rows = ['0,32,.08,.25,.38,.4,.6,0', '0,68,.08,.25,.38,.4,.6,0']

    def run_validator(self):
        (self.root/'simulated.json').write_text(json.dumps(self.simulated), encoding='utf-8')
        (self.root/'requests.txt').write_text(self.requests, encoding='utf-8')
        (self.root/'reference.csv').write_text('id,nstr,L_over_E_sr1,rho_f,BRF,R,T_down,atmospheric_absorptance\n' + '\n'.join(self.rows) + '\n', encoding='utf-8')
        return subprocess.run([sys.executable, '-B', str(SCRIPT), '--simulated', str(self.root/'simulated.json'),
                               '--reference-csv', str(self.root/'reference.csv'), '--reference-input', str(self.root/'requests.txt'),
                               '--output-dir', str(self.root/'out')], capture_output=True, text=True)

    def test_replay_scores_signed_error_and_records_provenance(self):
        result = self.run_validator()
        self.assertEqual(result.returncode, 0, result.stderr)
        summary = json.loads((self.root/'out/summary.json').read_text(encoding='utf-8'))
        score = summary['groups']['all_converged']['scores']['candidate']
        self.assertAlmostEqual(score['bias'], .05)
        self.assertAlmostEqual(score['mae'], .05)
        self.assertEqual(len(summary['reference_provenance']['request_sha256']), 64)

    def test_rejects_wrong_geometry_with_matching_ids(self):
        self.requests = self.requests.replace('.65', '.66')
        result = self.run_validator()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('geometry/physics does not match', result.stderr)

    def test_rejects_duplicate_missing_and_nonfinite_rows(self):
        original = list(self.rows)
        for rows, expected in [(original+[original[0]], 'duplicate'),
                               (original[:1], 'do not exactly match'),
                               ([original[0], original[1].replace(',.25,', ',nan,')], 'non-finite')]:
            with self.subTest(expected=expected):
                self.rows = rows
                # Failed validation may have created an output directory; use a
                # fresh temporary location for every independent input corruption.
                with tempfile.TemporaryDirectory() as temp:
                    saved = self.root
                    self.root = Path(temp)
                    result = self.run_validator()
                    self.root = saved
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(expected, result.stderr)

    def test_unconverged_reference_is_excluded_and_reported(self):
        self.rows[1] = self.rows[1].replace(',.25,', ',.30,')
        result = self.run_validator()
        self.assertNotEqual(result.returncode, 0)
        summary = json.loads((self.root/'out/summary.json').read_text(encoding='utf-8'))
        self.assertEqual(summary['unconverged_ids'], [0])
        self.assertEqual(summary['groups']['all_converged']['count'], 0)
        self.assertEqual(summary['groups']['all_converged']['scores'], {})

if __name__ == '__main__':
    unittest.main()
