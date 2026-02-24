import sys
import types
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _install_scipy_shim_if_missing() -> None:
    """Provide a tiny scipy shim for environments without scipy."""
    try:
        import scipy  # noqa: F401
        return
    except Exception:
        pass

    scipy_module = types.ModuleType("scipy")
    interpolate_module = types.ModuleType("scipy.interpolate")
    optimize_module = types.ModuleType("scipy.optimize")

    class _LinearInterpolator:
        def __init__(self, x, y):
            self.x = np.asarray(x, dtype=float)
            self.y = np.asarray(y)

        def __call__(self, x_new):
            x_new = np.asarray(x_new, dtype=float)
            if np.iscomplexobj(self.y):
                real = np.interp(x_new, self.x, np.real(self.y))
                imag = np.interp(x_new, self.x, np.imag(self.y))
                return real + 1j * imag
            return np.interp(x_new, self.x, self.y)

    def _least_squares_unavailable(*_args, **_kwargs):
        raise NotImplementedError("scipy.optimize.least_squares is unavailable in this environment.")

    interpolate_module.CubicSpline = _LinearInterpolator
    interpolate_module.PchipInterpolator = _LinearInterpolator
    optimize_module.least_squares = _least_squares_unavailable
    scipy_module.interpolate = interpolate_module
    scipy_module.optimize = optimize_module

    sys.modules["scipy"] = scipy_module
    sys.modules["scipy.interpolate"] = interpolate_module
    sys.modules["scipy.optimize"] = optimize_module


_install_scipy_shim_if_missing()

from ellipsometry import AirLayer, CauchyLayer, DrudeLorentzLayer, EllModel, IsoLayer  # noqa: E402


def _const_nk_layer(n: float, k: float, wl: np.ndarray, thickness=None) -> IsoLayer:
    df = pd.DataFrame(
        {
            "wl": wl,
            "n": np.full(wl.shape, n, dtype=float),
            "k": np.full(wl.shape, k, dtype=float),
        }
    )
    return IsoLayer(df, thickness=thickness)


class TestEllipsometerModeling(unittest.TestCase):
    def setUp(self):
        self.wl = np.array([400.0, 550.0, 700.0])
        self.air = AirLayer()
        self.film = _const_nk_layer(1.8, 0.02, self.wl, thickness=100.0)
        self.substrate = _const_nk_layer(3.5, 0.0, self.wl, thickness=None)
        self.model = EllModel([self.air, self.film, self.substrate], [55, 65], self.wl)

    def test_rejects_single_layer_model(self):
        with self.assertRaises(ValueError):
            EllModel([self.air], [60], self.wl)

    def test_transfer_matrix_2_shape(self):
        mx = self.model.transfer_matrix_2(self.film, self.air, pol="s")
        self.assertEqual(mx.shape, (2, 3, 2, 2))
        self.assertTrue(np.iscomplexobj(mx))

    def test_transfer_matrix_2_rejects_invalid_polarization(self):
        with self.assertRaises(ValueError):
            self.model.transfer_matrix_2(self.film, self.air, pol="x")

    def test_psi_delta_computation_populates_dataframes(self):
        psi = self.model.Psi_calc()
        delta = self.model.Delta_calc()

        self.assertEqual(psi.shape, (2, 3, 1))
        self.assertEqual(delta.shape, (2, 3, 1))
        self.assertIn("55", self.model.Psi.columns)
        self.assertIn("65", self.model.Delta.columns)
        self.assertTrue(np.all((psi >= 0) & (psi <= 90)))
        self.assertTrue(np.all((delta >= 0) & (delta < 360)))

    def test_get_nk_interpolates_without_missing_helpers(self):
        out = self.film.get_nk([500.0, 600.0])
        self.assertEqual(list(out.columns), ["wl", "n", "k"])
        self.assertEqual(len(out), 2)
        self.assertTrue(np.allclose(out["n"].to_numpy(), 1.8))
        self.assertTrue(np.allclose(out["k"].to_numpy(), 0.02))

    def test_get_epsilon_returns_expected_column_names(self):
        out = self.film.get_epsilon([500.0])
        self.assertEqual(list(out.columns), ["wl", "e1", "e2"])
        self.assertAlmostEqual(float(out["e1"].iloc[0]), self.film.e1[0], places=10)
        self.assertAlmostEqual(float(out["e2"].iloc[0]), self.film.e2[0], places=10)

    def test_transfer_mx_returns_2x2_matrix(self):
        mx = self.film.transfer_mx(self.air, wl=550.0, angle=60.0, pol="p")
        self.assertEqual(mx.shape, (2, 2))
        self.assertTrue(np.iscomplexobj(mx))

    def test_transfer_mx_rejects_infinite_layer(self):
        with self.assertRaises(ValueError):
            self.air.transfer_mx(self.air, wl=550.0, angle=60.0, pol="s")

    def test_psi_plot_honors_save_path_argument(self):
        import matplotlib.pyplot as plt

        calls = {}
        original_savefig = plt.savefig
        original_show = plt.show
        try:
            plt.savefig = lambda path, *args, **kwargs: calls.setdefault("path", path)
            plt.show = lambda *args, **kwargs: None
            self.model.Psi_calc_plot(save_path="/tmp/psi_plot.png")
        finally:
            plt.savefig = original_savefig
            plt.show = original_show

        self.assertEqual(calls.get("path"), "/tmp/psi_plot.png")

    def test_delta_plot_honors_save_path_argument(self):
        import matplotlib.pyplot as plt

        calls = {}
        original_savefig = plt.savefig
        original_show = plt.show
        try:
            plt.savefig = lambda path, *args, **kwargs: calls.setdefault("path", path)
            plt.show = lambda *args, **kwargs: None
            self.model.Delta_calc_plot(save_path="/tmp/delta_plot.png")
        finally:
            plt.savefig = original_savefig
            plt.show = original_show

        self.assertEqual(calls.get("path"), "/tmp/delta_plot.png")

    def test_two_layer_structure_produces_finite_results(self):
        model = EllModel([self.air, self.substrate], [60], self.wl)
        psi = model.Psi_calc()
        delta = model.Delta_calc()
        self.assertEqual(psi.shape, (1, 3, 1))
        self.assertEqual(delta.shape, (1, 3, 1))
        self.assertTrue(np.isfinite(psi).all())
        self.assertTrue(np.isfinite(delta).all())

    def test_zero_thickness_interlayer_matches_bare_interface(self):
        zero_layer = _const_nk_layer(2.2, 0.1, self.wl, thickness=0.0)
        model_with_zero_layer = EllModel([self.air, zero_layer, self.substrate], [60], self.wl)
        model_bare = EllModel([self.air, self.substrate], [60], self.wl)

        r_s_with_layer = model_with_zero_layer.reflect_coeff("s")[0, :, 0]
        r_s_bare = model_bare.reflect_coeff("s")[0, :, 0]
        r_p_with_layer = model_with_zero_layer.reflect_coeff("p")[0, :, 0]
        r_p_bare = model_bare.reflect_coeff("p")[0, :, 0]

        self.assertTrue(np.allclose(r_s_with_layer, r_s_bare, atol=1e-12, rtol=1e-12))
        self.assertTrue(np.allclose(r_p_with_layer, r_p_bare, atol=1e-12, rtol=1e-12))

    def test_cauchy_layer_stack_runs(self):
        cauchy = CauchyLayer(
            n_0=1.452,
            n_1=36,
            n_2=0,
            k_0=0,
            k_1=0,
            k_2=0,
            wl=self.wl,
            thickness=90.0,
        )
        model = EllModel([self.air, cauchy, self.substrate], [55, 70], self.wl)
        psi = model.Psi_calc()
        delta = model.Delta_calc()

        self.assertEqual(psi.shape, (2, 3, 1))
        self.assertEqual(delta.shape, (2, 3, 1))
        self.assertTrue(np.isfinite(psi).all())
        self.assertTrue(np.isfinite(delta).all())

    def test_drude_lorentz_multilayer_stack_runs(self):
        drude_params = {"omega_p": 6.0, "gamma": 0.4}
        lorentz_params = [{"f": 3.0, "omega_0": 4.0, "gamma": 0.6}]
        metal = DrudeLorentzLayer(
            epsilon_inf=2.0,
            drude_params=drude_params,
            lorentz_params=lorentz_params,
            wl=self.wl,
            thickness=30.0,
        )
        cauchy = CauchyLayer(
            n_0=1.45,
            n_1=20.0,
            n_2=0.0,
            k_0=0.0,
            k_1=0.0,
            k_2=0.0,
            wl=self.wl,
            thickness=15.0,
        )

        model = EllModel([self.air, metal, cauchy, self.substrate], [50, 65, 75], self.wl)
        r_s = model.reflect_coeff("s")
        r_p = model.reflect_coeff("p")
        psi = model.Psi_calc()
        delta = model.Delta_calc()

        self.assertEqual(r_s.shape, (3, 3, 1))
        self.assertEqual(r_p.shape, (3, 3, 1))
        self.assertEqual(psi.shape, (3, 3, 1))
        self.assertEqual(delta.shape, (3, 3, 1))
        self.assertTrue(np.isfinite(np.abs(r_s)).all())
        self.assertTrue(np.isfinite(np.abs(r_p)).all())
        self.assertTrue(np.isfinite(psi).all())
        self.assertTrue(np.isfinite(delta).all())


if __name__ == "__main__":
    unittest.main()
