import importlib.util
from pathlib import Path
import unittest
import numpy as np

spec=importlib.util.spec_from_file_location('solar_weights',Path(__file__).with_name('simsat-prepare-solar-weights.py'))
solar=importlib.util.module_from_spec(spec)
spec.loader.exec_module(solar)

class SolarWeightsTests(unittest.TestCase):
    def test_cubic_integral_matches_analytic_antiderivative(self):
        # E=1+x, R=x/2, f=2-0.3x over x=0..2 nm. Different E/R knots.
        x=np.array([400.,400.3,400.9,401.5,402.])
        t=np.array([400.,400.2,401.1,402.])
        w=solar.solar_weights(x,1+x-400,t/1000,(t-400)/2)
        f=2-0.3*(t-400)
        p=np.polynomial.Polynomial([1,1])*np.polynomial.Polynomial([0,.5])*np.polynomial.Polynomial([2,-.3])
        a=p.integ()
        self.assertAlmostEqual(float(w@f),float(a(2)-a(0)),places=12)
        self.assertTrue(np.all(w>=0))

    def test_narrow_solar_line_energy_is_preserved(self):
        # A triangular line with area 0.1 W/m^2 between coarse response knots.
        x=np.array([400.,400.4,400.5,400.6,402.])
        w=solar.solar_weights(x,np.array([1.,1.,0.,1.,1.]),np.array([.4,.402]),np.ones(2))
        self.assertAlmostEqual(float(w.sum()),1.9,places=12)
        # Naive resampling sees only E=1 at both ends and incorrectly gives 2.
        self.assertGreater(abs(float(w.sum())-2.0),0.09)

    def test_subdividing_response_does_not_change_linear_transfer(self):
        x=np.array([400.,400.4,400.5,400.6,402.]);e=np.array([1.,1.,0.,1.,1.])
        t=np.array([.4,.402]);r=np.array([.1,.8])
        u=np.array([.4,.40025,.4007,.4015,.402])
        a=solar.solar_weights(x,e,t,r)@(3+2*t)
        b=solar.solar_weights(x,e,u,np.interp(u,t,r))@(3+2*u)
        self.assertAlmostEqual(float(a),float(b),places=12)

    def test_wavelength_units_and_flat_solar_source(self):
        w=solar.solar_weights([400,500],[2,2],[.42,.46,.48],[0,1,0])
        self.assertAlmostEqual(float(w.sum()),60.0,places=12)
        self.assertAlmostEqual(float(w.sum()/np.trapezoid([0,1,0],[.42,.46,.48])),2000.,places=9)

    def test_invalid_spectra_and_partial_coverage_rejected(self):
        with self.assertRaises(ValueError): solar.solar_weights([401,500],[1,1],[.4,.5],[1,1])
        with self.assertRaises(ValueError): solar.solar_weights([400,500],[1,-1],[.4,.5],[1,1])
        with self.assertRaises(ValueError): solar.solar_weights([400,500],[1,1],[.4,.5],[0,0])
        with self.assertRaises(ValueError): solar.solar_weights([400,400],[1,1],[.4,.5],[1,1])
        with self.assertRaises(ValueError): solar.solar_weights([400,500],[1,np.nan],[.4,.5],[1,1])

if __name__=='__main__':unittest.main()
