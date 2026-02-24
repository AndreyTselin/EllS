"""PyQt UI for quick ellipsometry model setup and plotting."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pyqtgraph as pg
from PyQt5.QtWidgets import (
    QApplication,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def _install_scipy_shim_if_missing() -> None:
    """Provide minimal interpolation functionality if scipy is unavailable."""
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
        raise NotImplementedError("scipy.optimize.least_squares is unavailable.")

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


pg.setConfigOptions(antialias=True)


def _parse_angles(text: str) -> list[float | int]:
    parts = [p.strip() for p in text.split(",") if p.strip()]
    if not parts:
        raise ValueError("Angles list is empty.")

    angles: list[float | int] = []
    for token in parts:
        value = float(token)
        if value <= 0 or value >= 90:
            raise ValueError(f"Angle must be between 0 and 90 deg, got {value}.")
        if value.is_integer():
            angles.append(int(value))
        else:
            angles.append(value)
    return angles


def _load_nk_file(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    else:
        df = pd.read_csv(path, sep=r"\s+")

    normalized = {col.lower().strip(): col for col in df.columns}
    if "wl(mu)" in normalized:
        wl = df[normalized["wl(mu)"]].to_numpy(dtype=float) * 1000.0
    elif "wl" in normalized:
        wl = df[normalized["wl"]].to_numpy(dtype=float)
    else:
        raise ValueError("Expected wavelength column named 'wl' or 'wl(mu)'.")

    if "n" not in normalized:
        raise ValueError("Expected refractive index column 'n'.")

    n = df[normalized["n"]].to_numpy(dtype=float)
    if "k" in normalized:
        k = df[normalized["k"]].to_numpy(dtype=float)
    else:
        k = np.zeros_like(n)

    out = pd.DataFrame({"wl": wl, "n": n, "k": k}).dropna()
    if out.empty:
        raise ValueError("No valid rows found after parsing substrate file.")
    out = out.sort_values("wl").drop_duplicates("wl", keep="last")
    return out.reset_index(drop=True)


def _spin(min_v: float, max_v: float, value: float, step: float, decimals: int = 4) -> QDoubleSpinBox:
    widget = QDoubleSpinBox()
    widget.setRange(min_v, max_v)
    widget.setDecimals(decimals)
    widget.setSingleStep(step)
    widget.setValue(value)
    return widget


class EllipsometryWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Ellipsometry Model UI")
        self.resize(1280, 780)
        self._build_ui()
        self._apply_style()

    def _build_ui(self) -> None:
        root = QWidget()
        self.setCentralWidget(root)
        main_layout = QHBoxLayout(root)
        main_layout.setContentsMargins(12, 12, 12, 12)
        main_layout.setSpacing(12)

        controls = self._build_controls_panel()
        visual = self._build_visual_panel()

        main_layout.addWidget(controls, 0)
        main_layout.addWidget(visual, 1)

    def _build_controls_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setSpacing(10)

        substrate_group = QGroupBox("Substrate Data")
        substrate_form = QGridLayout(substrate_group)
        self.substrate_path = QLineEdit(
            str(PROJECT_ROOT / "assets" / "optical_cosnt_nk" / "Si_Aspnes.txt")
        )
        browse_button = QPushButton("Browse")
        browse_button.clicked.connect(self._browse_substrate_file)
        substrate_form.addWidget(QLabel("nk file"), 0, 0)
        substrate_form.addWidget(self.substrate_path, 0, 1)
        substrate_form.addWidget(browse_button, 0, 2)
        layout.addWidget(substrate_group)

        sweep_group = QGroupBox("Sweep")
        sweep_form = QFormLayout(sweep_group)
        self.wl_start = _spin(100.0, 10000.0, 190.0, 10.0, decimals=1)
        self.wl_stop = _spin(120.0, 12000.0, 1200.0, 10.0, decimals=1)
        self.wl_step = _spin(0.1, 500.0, 2.0, 0.5, decimals=2)
        self.angles = QLineEdit("50, 60, 70")
        sweep_form.addRow("Start wl (nm)", self.wl_start)
        sweep_form.addRow("Stop wl (nm)", self.wl_stop)
        sweep_form.addRow("Step (nm)", self.wl_step)
        sweep_form.addRow("Angles (deg)", self.angles)
        layout.addWidget(sweep_group)

        model_group = QGroupBox("Film Model")
        model_layout = QVBoxLayout(model_group)
        self.model_type = QComboBox()
        self.model_type.addItems(["Constant n,k", "Cauchy", "Drude-Lorentz"])
        self.model_type.currentIndexChanged.connect(self._on_model_type_changed)

        self.model_stack = QStackedWidget()
        self.constant_page = self._build_constant_page()
        self.cauchy_page = self._build_cauchy_page()
        self.drude_page = self._build_drude_page()
        self.model_stack.addWidget(self.constant_page)
        self.model_stack.addWidget(self.cauchy_page)
        self.model_stack.addWidget(self.drude_page)

        model_layout.addWidget(self.model_type)
        model_layout.addWidget(self.model_stack)
        layout.addWidget(model_group)

        run_button = QPushButton("Compute Psi/Delta")
        run_button.setObjectName("runButton")
        run_button.clicked.connect(self._run_model)
        layout.addWidget(run_button)

        self.status = QLabel("Ready")
        self.status.setObjectName("statusLabel")
        layout.addWidget(self.status)
        layout.addStretch(1)
        return panel

    def _build_visual_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setSpacing(10)

        self.graph_widget = pg.GraphicsLayoutWidget()
        self.graph_widget.setBackground("w")
        self.psi_plot = self.graph_widget.addPlot(row=0, col=0, title="Psi")
        self.delta_plot = self.graph_widget.addPlot(row=1, col=0, title="Delta")
        self.delta_plot.setXLink(self.psi_plot)
        self.psi_plot.getAxis("left").setTextPen("#1f2933")
        self.delta_plot.getAxis("left").setTextPen("#1f2933")
        self.delta_plot.getAxis("bottom").setTextPen("#1f2933")

        self.psi_plot.setLabel("left", "Psi", units="deg")
        self.delta_plot.setLabel("left", "Delta", units="deg")
        self.delta_plot.setLabel("bottom", "Wavelength", units="nm")
        self.psi_plot.showGrid(x=True, y=True, alpha=0.25)
        self.delta_plot.showGrid(x=True, y=True, alpha=0.25)
        self.psi_plot.setMouseEnabled(x=True, y=True)
        self.delta_plot.setMouseEnabled(x=True, y=True)

        self.psi_legend = self.psi_plot.addLegend(offset=(10, 10))
        self.delta_legend = self.delta_plot.addLegend(offset=(10, 10))

        self.summary = QPlainTextEdit()
        self.summary.setReadOnly(True)
        self.summary.setPlaceholderText("Computation summary will appear here.")

        layout.addWidget(self.graph_widget, 4)
        layout.addWidget(self.summary, 2)
        return panel

    def _build_constant_page(self) -> QWidget:
        page = QWidget()
        form = QFormLayout(page)
        self.const_n = _spin(1.0, 8.0, 1.46, 0.01, decimals=4)
        self.const_k = _spin(0.0, 5.0, 0.0, 0.01, decimals=4)
        self.const_t = _spin(0.0, 5000.0, 109.4, 1.0, decimals=2)
        form.addRow("n", self.const_n)
        form.addRow("k", self.const_k)
        form.addRow("Thickness (nm)", self.const_t)
        return page

    def _build_cauchy_page(self) -> QWidget:
        page = QWidget()
        form = QFormLayout(page)
        self.c_n0 = _spin(0.0, 10.0, 1.452, 0.01, decimals=6)
        self.c_n1 = _spin(-1e5, 1e5, 36.0, 1.0, decimals=4)
        self.c_n2 = _spin(-1e5, 1e5, 0.0, 1.0, decimals=4)
        self.c_k0 = _spin(-10.0, 10.0, 0.0, 0.01, decimals=6)
        self.c_k1 = _spin(-1e5, 1e5, 0.0, 1.0, decimals=4)
        self.c_k2 = _spin(-1e5, 1e5, 0.0, 1.0, decimals=4)
        self.c_t = _spin(0.0, 5000.0, 109.4, 1.0, decimals=2)
        form.addRow("n0", self.c_n0)
        form.addRow("n1", self.c_n1)
        form.addRow("n2", self.c_n2)
        form.addRow("k0", self.c_k0)
        form.addRow("k1", self.c_k1)
        form.addRow("k2", self.c_k2)
        form.addRow("Thickness (nm)", self.c_t)
        return page

    def _build_drude_page(self) -> QWidget:
        page = QWidget()
        form = QFormLayout(page)
        self.d_eps_inf = _spin(-100.0, 100.0, 2.0, 0.1, decimals=4)
        self.d_omega_p = _spin(0.0, 200.0, 6.0, 0.1, decimals=4)
        self.d_gamma = _spin(0.0, 50.0, 0.4, 0.1, decimals=4)
        self.d_f = _spin(0.0, 200.0, 3.0, 0.1, decimals=4)
        self.d_omega_0 = _spin(0.0, 200.0, 4.0, 0.1, decimals=4)
        self.d_lor_gamma = _spin(0.0, 50.0, 0.6, 0.1, decimals=4)
        self.d_t = _spin(0.0, 5000.0, 30.0, 1.0, decimals=2)
        form.addRow("epsilon_inf", self.d_eps_inf)
        form.addRow("omega_p (eV)", self.d_omega_p)
        form.addRow("drude gamma (eV)", self.d_gamma)
        form.addRow("Lorentz f", self.d_f)
        form.addRow("Lorentz omega0 (eV)", self.d_omega_0)
        form.addRow("Lorentz gamma (eV)", self.d_lor_gamma)
        form.addRow("Thickness (nm)", self.d_t)
        return page

    def _apply_style(self) -> None:
        self.setStyleSheet(
            """
            QWidget { background: #f5f7fa; color: #1f2933; font-size: 12px; }
            QGroupBox {
                font-weight: bold;
                border: 1px solid #d0d7de;
                border-radius: 8px;
                margin-top: 8px;
                padding-top: 10px;
                background: #ffffff;
            }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; }
            QLineEdit, QDoubleSpinBox, QComboBox, QPlainTextEdit {
                border: 1px solid #c7d0d9;
                border-radius: 6px;
                padding: 4px 6px;
                background: #ffffff;
            }
            QPushButton {
                border: 1px solid #2f855a;
                border-radius: 8px;
                background: #38a169;
                color: white;
                padding: 6px 10px;
                font-weight: 600;
            }
            QPushButton:hover { background: #2f855a; }
            #runButton { min-height: 34px; }
            #statusLabel { color: #2f855a; font-weight: 600; padding: 4px; }
            """
        )

    def _on_model_type_changed(self, index: int) -> None:
        self.model_stack.setCurrentIndex(index)

    def _browse_substrate_file(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select nk Data File",
            str(PROJECT_ROOT / "assets"),
            "Data Files (*.txt *.csv *.dat);;All Files (*)",
        )
        if file_path:
            self.substrate_path.setText(file_path)

    def _wavelength_grid(self) -> np.ndarray:
        start = self.wl_start.value()
        stop = self.wl_stop.value()
        step = self.wl_step.value()
        if stop <= start:
            raise ValueError("Stop wavelength must be larger than start wavelength.")
        points = int(np.floor((stop - start) / step)) + 1
        if points < 2:
            raise ValueError("Wavelength sweep must contain at least 2 points.")
        return start + np.arange(points) * step

    def _build_film_layer(self, wl: np.ndarray):
        model_name = self.model_type.currentText()
        if model_name == "Constant n,k":
            n = self.const_n.value()
            k = self.const_k.value()
            thickness = self.const_t.value()
            df = pd.DataFrame(
                {
                    "wl": wl,
                    "n": np.full(wl.shape, n, dtype=float),
                    "k": np.full(wl.shape, k, dtype=float),
                }
            )
            return IsoLayer(df, thickness=thickness), f"Constant n,k | n={n}, k={k}, t={thickness} nm"

        if model_name == "Cauchy":
            layer = CauchyLayer(
                n_0=self.c_n0.value(),
                n_1=self.c_n1.value(),
                n_2=self.c_n2.value(),
                k_0=self.c_k0.value(),
                k_1=self.c_k1.value(),
                k_2=self.c_k2.value(),
                wl=wl,
                thickness=self.c_t.value(),
            )
            summary = (
                f"Cauchy | n0={self.c_n0.value()}, n1={self.c_n1.value()}, n2={self.c_n2.value()}, "
                f"k0={self.c_k0.value()}, k1={self.c_k1.value()}, k2={self.c_k2.value()}, "
                f"t={self.c_t.value()} nm"
            )
            return layer, summary

        layer = DrudeLorentzLayer(
            epsilon_inf=self.d_eps_inf.value(),
            drude_params={"omega_p": self.d_omega_p.value(), "gamma": self.d_gamma.value()},
            lorentz_params=[
                {
                    "f": self.d_f.value(),
                    "omega_0": self.d_omega_0.value(),
                    "gamma": self.d_lor_gamma.value(),
                }
            ],
            wl=wl,
            thickness=self.d_t.value(),
        )
        summary = (
            f"Drude-Lorentz | eps_inf={self.d_eps_inf.value()}, omega_p={self.d_omega_p.value()}, "
            f"drude_gamma={self.d_gamma.value()}, f={self.d_f.value()}, omega0={self.d_omega_0.value()}, "
            f"lor_gamma={self.d_lor_gamma.value()}, t={self.d_t.value()} nm"
        )
        return layer, summary

    def _run_model(self) -> None:
        try:
            wl = self._wavelength_grid()
            angles = _parse_angles(self.angles.text())
            substrate_df = _load_nk_file(Path(self.substrate_path.text()))
            substrate = IsoLayer(substrate_df, thickness=None)
            air = AirLayer()
            film, film_summary = self._build_film_layer(wl)

            model = EllModel([air, film, substrate], angles, wl)
            model.Psi_calc()
            model.Delta_calc()

            self._draw_results(model, angles)
            self._update_summary(model, angles, film_summary)
            self.status.setText(f"Computed {len(wl)} wavelengths across {len(angles)} angle(s).")

        except Exception as exc:  # noqa: BLE001
            self.status.setText("Computation failed.")
            QMessageBox.critical(self, "Model Error", str(exc))

    def _draw_results(self, model: EllModel, angles: list[float | int]) -> None:
        self.psi_plot.clear()
        self.delta_plot.clear()
        self.psi_legend.clear()
        self.delta_legend.clear()

        for idx, angle in enumerate(angles):
            key = str(angle)
            color = pg.intColor(idx, hues=max(6, len(angles)))
            pen = pg.mkPen(color=color, width=2)
            self.psi_plot.plot(model.wl, model.Psi[key].to_numpy(), pen=pen, name=f"{angle} deg")
            self.delta_plot.plot(model.wl, model.Delta[key].to_numpy(), pen=pen, name=f"{angle} deg")

        self.psi_plot.enableAutoRange()
        self.delta_plot.enableAutoRange()

    def _update_summary(self, model: EllModel, angles: list[float | int], film_summary: str) -> None:
        psi_cols = ["wl"] + [str(a) for a in angles]
        delta_cols = ["wl"] + [str(a) for a in angles]
        psi_preview = model.Psi[psi_cols].head(6).to_string(index=False)
        delta_preview = model.Delta[delta_cols].head(6).to_string(index=False)

        text = (
            f"Film: {film_summary}\n"
            f"Substrate file: {self.substrate_path.text()}\n"
            f"Wavelength range: {model.wl[0]:.2f} .. {model.wl[-1]:.2f} nm ({len(model.wl)} points)\n"
            f"Angles: {', '.join(str(a) for a in angles)}\n\n"
            "Psi preview:\n"
            f"{psi_preview}\n\n"
            "Delta preview:\n"
            f"{delta_preview}\n"
        )
        self.summary.setPlainText(text)


def main() -> int:
    app = QApplication(sys.argv)
    window = EllipsometryWindow()
    window.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
