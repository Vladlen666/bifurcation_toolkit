# -*- coding: utf-8 -*-
"""Dynamic System Sandbox — Hopf bifurcations (robust, 2025-07-04)."""

from __future__ import annotations
import sys
import itertools
from collections.abc import Callable

import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp
from scipy.optimize import root
import matplotlib
matplotlib.use("QtAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.lines import Line2D

from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QSplitter,
    QPushButton, QFormLayout, QDialog, QDialogButtonBox,
    QComboBox, QDoubleSpinBox, QLabel, QPlainTextEdit, QMessageBox,
    QTabWidget, QTableWidget, QTableWidgetItem
)

# --------------------------------------------------------------------------- #
#                          1.  «Безопасная» экспонента                        #
# --------------------------------------------------------------------------- #
EXP_CLIP = 50.0  # exp(±50) ≈ 5·10²¹
def safe_exp(z):
    return np.exp(np.clip(z, -EXP_CLIP, EXP_CLIP))

SAFE_MODULE = {'exp': safe_exp, 'safe_exp': safe_exp, 'np': np}

# --------------------------------------------------------------------------- #
#                           2.  Вспомогательные диалоги                       #
# --------------------------------------------------------------------------- #
class SystemDialog(QDialog):
    def __init__(self, f_txt: str, g_txt: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Edit system")
        self.f_edit = QPlainTextEdit(f_txt)
        self.g_edit = QPlainTextEdit(g_txt)
        form = QFormLayout(self)
        form.addRow("f(x, y; μ1, μ2) =", self.f_edit)
        form.addRow("g(x, y; μ1, μ2) =", self.g_edit)
        box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok |
            QDialogButtonBox.StandardButton.Cancel
        )
        box.accepted.connect(self.accept)
        box.rejected.connect(self.reject)
        form.addRow(box)

    def texts(self) -> tuple[str, str]:
        return self.f_edit.toPlainText(), self.g_edit.toPlainText()

class RangeDialog(QDialog):
    def __init__(self, rng: dict[str, float], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Parameter range")
        self.edits: dict[str, QDoubleSpinBox] = {}
        form = QFormLayout(self)
        for k in ("mu1_min", "mu1_max", "mu2_min", "mu2_max"):
            spb = QDoubleSpinBox(minimum=-1e6, maximum=1e6, decimals=4)
            spb.setValue(rng[k])
            form.addRow(k, spb)
            self.edits[k] = spb
        box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok |
            QDialogButtonBox.StandardButton.Cancel
        )
        box.accepted.connect(self.accept)
        box.rejected.connect(self.reject)
        form.addRow(box)

    def values(self) -> dict[str, float]:
        return {k: w.value() for k, w in self.edits.items()}

class PhaseRangeDialog(QDialog):
    def __init__(self, pr: dict[str, float], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Phase range")
        self.edits: dict[str, QDoubleSpinBox] = {}
        form = QFormLayout(self)
        for k, l in (("x_min", "x min"), ("x_max", "x max"),
                     ("y_min", "y min"), ("y_max", "y max")):
            spb = QDoubleSpinBox(minimum=-1e6, maximum=1e6, decimals=4)
            spb.setValue(pr[k])
            form.addRow(l, spb)
            self.edits[k] = spb
        box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok |
            QDialogButtonBox.StandardButton.Cancel
        )
        box.accepted.connect(self.accept)
        box.rejected.connect(self.reject)
        form.addRow(box)

    def values(self) -> dict[str, float]:
        return {k: w.value() for k, w in self.edits.items()}

# --------------------------------------------------------------------------- #
#                                  3.  Canvas                                 #
# --------------------------------------------------------------------------- #
class MplCanvas(QWidget):
    def __init__(self):
        super().__init__()
        self.fig, self.ax = plt.subplots()
        self.canvas = FigureCanvasQTAgg(self.fig)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.canvas)
        lay.addWidget(NavigationToolbar2QT(self.canvas, self))
        self.canvas.mpl_connect("scroll_event", self._on_scroll)

    def _on_scroll(self, e):
        if e.inaxes is None:
            return
        scale = 1.2 if e.button == "up" else 0.8
        xm, ym = e.xdata, e.ydata
        xmin, xmax = e.inaxes.get_xlim()
        ymin, ymax = e.inaxes.get_ylim()
        e.inaxes.set_xlim(xm + (xmin - xm) * scale, xm + (xmax - xm) * scale)
        e.inaxes.set_ylim(ym + (ymin - ym) * scale, ym + (ymax - ym) * scale)
        self.canvas.draw_idle()

# --------------------------------------------------------------------------- #
#                               4.  Main Window                               #
# --------------------------------------------------------------------------- #
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Dynamic System Sandbox — Hopf")
        self.resize(1100, 650)

        # -- default system ----------------------------------------------------
        self.f_expr = "-2*exp(-x) + exp(-2*x) + y"
        self.g_expr = "-x + mu1*y + mu2"

        # Precompute Hopf branches container before compile
        self.hopf_branches: list[tuple[Callable, Callable, Callable]] = []

        # compile everything
        self._compile_system()

        # -- state -------------------------------------------------------------
        self.range = dict(mu1_min=-3, mu1_max=3, mu2_min=-3, mu2_max=3)
        self.phase_range = dict(x_min=-3, x_max=3, y_min=-3, y_max=3)
        self.current_mu: tuple[float, float] | None = None

        self.traj_lines: list[Line2D] = []
        self.xt_data: list[tuple[np.ndarray, np.ndarray]] = []

        self.nullcline_art = []
        self.nullcline_pts = []
        self.field_art = []
        self.sep_lines = []
        self.equilibria = []

        self.show_field = False
        self.param_marker = None
        self.color_cycle = itertools.cycle(
            plt.rcParams["axes.prop_cycle"].by_key()["color"]
        )

        # -- interface ---------------------------------------------------------
        self.param_canvas = MplCanvas()
        self.phase_canvas = MplCanvas()
        self.hopf_canvas = MplCanvas()
        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        self.splitter.addWidget(self.param_canvas)
        self.splitter.addWidget(self.phase_canvas)

        tabs = QTabWidget()
        # Plots tab
        page = QWidget()
        QVBoxLayout(page).addWidget(self.splitter)
        tabs.addTab(page, "Plots")
        # Equilibria tab
        self.eq_table = QTableWidget(0, 5)
        self.eq_table.setHorizontalHeaderLabels(["x", "y", "type", "λ₁", "λ₂"])
        tabs.addTab(self.eq_table, "Equilibria")
        # Hopf tab
        tabs.addTab(self.hopf_canvas, "Hopf")
        self.tabs = tabs
        self.hopf_tab_index = tabs.count() - 1

        self.setCentralWidget(tabs)

        self._build_toolbar()
        self._configure_param_axes()
        self._configure_phase_axes()
        self._connect_events()

        # for x(t) window
        self.xt_fig = None
        self.xt_ax = None

    def _compile_system(self):
        # --- symbolic setup ---
        x, y, m1, m2 = sp.symbols("x y mu1 mu2")
        f_sym = sp.sympify(self.f_expr)
        g_sym = sp.sympify(self.g_expr)
        J = sp.Matrix([f_sym, g_sym]).jacobian([x, y])
        detJ = sp.simplify(J.det())
        trJ_sym = sp.simplify(J.trace())

        # lambdas for RHS and Jacobian entries
        self.f_lam = sp.lambdify((x, y, m1, m2), f_sym,
                                 modules=[SAFE_MODULE, 'numpy'])
        self.g_lam = sp.lambdify((x, y, m1, m2), g_sym,
                                 modules=[SAFE_MODULE, 'numpy'])
        self.J11 = sp.lambdify((x, y, m1, m2), J[0, 0],
                               modules=[SAFE_MODULE, 'numpy'])
        self.J12 = sp.lambdify((x, y, m1, m2), J[0, 1],
                               modules=[SAFE_MODULE, 'numpy'])
        self.J21 = sp.lambdify((x, y, m1, m2), J[1, 0],
                               modules=[SAFE_MODULE, 'numpy'])
        self.J22 = sp.lambdify((x, y, m1, m2), J[1, 1],
                               modules=[SAFE_MODULE, 'numpy'])
        self.detJ_lam = sp.lambdify((x, y, m1, m2), detJ,
                                    modules=[SAFE_MODULE, 'numpy'])

        # RHS function
        self.rhs_func: Callable = lambda t, s, μ1, μ2: np.array([
            self.f_lam(s[0], s[1], μ1, μ2),
            self.g_lam(s[0], s[1], μ1, μ2)
        ])

        # --- Precompute Hopf branches: solve f=0, g=0, trJ=0 -> μ2 = φ(μ1) ---
        self.hopf_branches.clear()
        try:
            eqs = sp.solve([f_sym, g_sym], [x, y], dict=True)
        except (NotImplementedError, ValueError):
            eqs = []

        for sol in eqs:
            xi = sol[x]
            yi = sol[y]
            trJ_i = trJ_sym.subs({x: xi, y: yi})
            sol_m2 = sp.solve(trJ_i, m2)
            if not sol_m2:
                continue
            φ = sol_m2[0]
            phi_fun = sp.lambdify(m1, φ, modules=[SAFE_MODULE, 'numpy'])
            xi_fun  = sp.lambdify((m1, m2), xi, modules=[SAFE_MODULE, 'numpy'])
            yi_fun  = sp.lambdify((m1, m2), yi, modules=[SAFE_MODULE, 'numpy'])
            self.hopf_branches.append((phi_fun, xi_fun, yi_fun))

    def _build_toolbar(self):
        bar = self.addToolBar("Controls")
        bar.setIconSize(QSize(16, 16))

        for txt, slot in [
            ("Edit system",     self._edit_system),
            ("Set param range", self._edit_range),
            ("Set phase range", self._edit_phase_range)
        ]:
            act = QAction(txt, self)
            act.triggered.connect(slot)
            bar.addAction(act)

        bar.addSeparator()
        bar.addWidget(QLabel("μ₁:"))
        self.mu1_spin = QDoubleSpinBox(decimals=3)
        self.mu1_spin.valueChanged.connect(self._on_spin_changed)
        bar.addWidget(self.mu1_spin)

        bar.addWidget(QLabel("μ₂:"))
        self.mu2_spin = QDoubleSpinBox(decimals=3)
        self.mu2_spin.valueChanged.connect(self._on_spin_changed)
        bar.addWidget(self.mu2_spin)

        bar.addSeparator()
        self.integrator_cb = QComboBox()
        self.integrator_cb.addItems(["RK45", "RK23", "DOP853", "LSODA", "Radau", "BDF"])
        bar.addWidget(QLabel("Integrator:"))
        bar.addWidget(self.integrator_cb)

        bar.addSeparator()
        btn_clear = QPushButton("Clear")
        btn_clear.clicked.connect(self._clear_trajectories)
        bar.addWidget(btn_clear)

        bar.addSeparator()
        self.act_vector = QAction("Vector field", self, checkable=True)
        self.act_vector.triggered.connect(self._toggle_vector_field)
        bar.addAction(self.act_vector)

        self.act_separatrices = QAction("Separatrices", self, checkable=True)
        self.act_separatrices.triggered.connect(self._toggle_separatrices)
        bar.addAction(self.act_separatrices)

        bar.addSeparator()
        self.act_hopf = QAction("Hopf", self, checkable=True)
        self.act_hopf.triggered.connect(self._toggle_hopf)
        bar.addAction(self.act_hopf)

        bar.addSeparator()
        self.act_xt = QAction("Plot x(t)", self)
        self.act_xt.setEnabled(False)
        self.act_xt.triggered.connect(self._plot_xt)
        bar.addAction(self.act_xt)

    def _configure_param_axes(self):
        ax = self.param_canvas.ax
        ax.clear()
        ax.set_title("Parameter plane (μ1, μ2)")
        ax.set_xlabel("μ1"); ax.set_ylabel("μ2")
        ax.set_xlim(self.range["mu1_min"], self.range["mu1_max"])
        ax.set_ylim(self.range["mu2_min"], self.range["mu2_max"])
        self.param_canvas.canvas.draw_idle()

    def _configure_phase_axes(self):
        ax = self.phase_canvas.ax
        ax.clear()
        ax.set_title("Phase plane (x, y)")
        ax.set_xlabel("x"); ax.set_ylabel("y")
        ax.set_xlim(self.phase_range["x_min"], self.phase_range["x_max"])
        ax.set_ylim(self.phase_range["y_min"], self.phase_range["y_max"])
        self.phase_canvas.canvas.draw_idle()

    def _connect_events(self):
        self.param_canvas.canvas.mpl_connect("button_press_event", self._on_param_click)
        self.phase_canvas.canvas.mpl_connect("button_press_event", self._on_phase_click)

    def _on_spin_changed(self, _=None):
        μ1, μ2 = self.mu1_spin.value(), self.mu2_spin.value()
        self.current_mu = (μ1, μ2)
        self._clear_trajectories()
        self._toggle_separatrices(False)
        if self.param_marker:
            self.param_marker.remove()
        self.param_marker, = self.param_canvas.ax.plot(μ1, μ2, "xr", ms=8)
        self.param_canvas.canvas.draw_idle()
        self._draw_nullclines(μ1, μ2)
        if self.show_field:
            self._draw_vector_field()
        self.statusBar().showMessage(f"μ=({μ1:.3g}, {μ2:.3g})")

    def _on_param_click(self, e):
        if not (e.inaxes and e.dblclick):
            return
        μ1, μ2 = e.xdata, e.ydata
        self.current_mu = (μ1, μ2)
        self.mu1_spin.blockSignals(True); self.mu1_spin.setValue(μ1); self.mu1_spin.blockSignals(False)
        self.mu2_spin.blockSignals(True); self.mu2_spin.setValue(μ2); self.mu2_spin.blockSignals(False)
        self._clear_trajectories()
        self._toggle_separatrices(False)
        if self.param_marker:
            self.param_marker.remove()
        self.param_marker, = self.param_canvas.ax.plot(μ1, μ2, "xr", ms=8)
        self.param_canvas.canvas.draw_idle()
        self._draw_nullclines(μ1, μ2)
        if self.show_field:
            self._draw_vector_field()
        self.statusBar().showMessage(f"Selected μ=({μ1:.3g}, {μ2:.3g})")

    def _find_equilibria_numeric(self, μ1: float, μ2: float) -> list[tuple[float, float]]:
        sols: list[tuple[float, float]] = []
        tol_f, tol_xy = 1e-4, 1e-3
        # coarse grid 5×5
        guesses = [
            (x0, y0)
            for x0 in np.linspace(self.phase_range["x_min"], self.phase_range["x_max"], 5)
            for y0 in np.linspace(self.phase_range["y_min"], self.phase_range["y_max"], 5)
        ]
        def fg(v):
            return self.f_lam(v[0], v[1], μ1, μ2), self.g_lam(v[0], v[1], μ1, μ2)
        for x0, y0 in guesses:
            sol = root(
                lambda v: [np.tanh(fg(v)[0]), np.tanh(fg(v)[1])],
                [x0, y0], method="hybr",
                options={"maxfev": 200, "xtol": 1e-6}
            )
            if sol.success:
                xe, ye = sol.x
                if max(abs(fg((xe, ye))[0]), abs(fg((xe, ye))[1])) > tol_f:
                    continue
                if any(np.hypot(xe - xs, ye - ys) < tol_xy for xs, ys in sols):
                    continue
                sols.append((xe, ye))
        # refine on 50×50
        nx = np.linspace(self.phase_range["x_min"], self.phase_range["x_max"], 50)
        ny = np.linspace(self.phase_range["y_min"], self.phase_range["y_max"], 50)
        XX, YY = np.meshgrid(nx, ny)
        with np.errstate(over='ignore', invalid='ignore'):
            FF = self.f_lam(XX, YY, μ1, μ2)
            GG = self.g_lam(XX, YY, μ1, μ2)
        mask = (np.abs(FF) < 1e-2) & (np.abs(GG) < 1e-2)
        for i, j in np.argwhere(mask):
            x0, y0 = XX[i, j], YY[i, j]
            if any(np.hypot(x0 - xs, y0 - ys) < tol_xy for xs, ys in sols):
                continue
            sol = root(
                lambda v: [np.tanh(fg(v)[0]), np.tanh(fg(v)[1])],
                [x0, y0], method="hybr",
                options={"maxfev": 300, "xtol": 1e-8}
            )
            if sol.success:
                xe, ye = sol.x
                if max(abs(fg((xe, ye))[0]), abs(fg((xe, ye))[1])) > tol_f:
                    continue
                if any(np.hypot(xe - xs, ye - ys) < tol_xy for xs, ys in sols):
                    continue
                sols.append((xe, ye))
        return [(round(x, 6), round(y, 6)) for x, y in sols]

    def _draw_nullclines(self, μ1: float, μ2: float):
        for art in self.nullcline_art:
            if hasattr(art, "collections"):
                for c in art.collections:
                    c.remove()
            else:
                art.remove()
        self.nullcline_art.clear()
        for obj in self.nullcline_pts:
            obj.remove()
        self.nullcline_pts.clear()
        self.eq_table.setRowCount(0)
        self.equilibria.clear()

        ax = self.phase_canvas.ax
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        xx, yy = np.meshgrid(
            np.linspace(xmin, xmax, 300),
            np.linspace(ymin, ymax, 300)
        )
        with np.errstate(over='ignore', invalid='ignore'):
            F = self.f_lam(xx, yy, μ1, μ2)
            G = self.g_lam(xx, yy, μ1, μ2)
        F = np.nan_to_num(F, nan=0.0, posinf=np.nan, neginf=np.nan)
        G = np.nan_to_num(G, nan=0.0, posinf=np.nan, neginf=np.nan)

        cf = ax.contour(xx, yy, F, levels=[0], colors="blue",
                        linestyles="--", linewidths=1.2)
        cg = ax.contour(xx, yy, G, levels=[0], colors="green",
                        linestyles="-", linewidths=1.2)
        self.nullcline_art += [cf, cg]

        for xf, yf in self._find_equilibria_numeric(μ1, μ2):
            if not (xmin <= xf <= xmax and ymin <= yf <= ymax):
                continue
            a = self.J11(xf, yf, μ1, μ2)
            b = self.J12(xf, yf, μ1, μ2)
            c = self.J21(xf, yf, μ1, μ2)
            d = self.J22(xf, yf, μ1, μ2)
            Jmat = np.array([[a, b], [c, d]])
            ev, evec = np.linalg.eig(Jmat)
            re, im = np.real(ev), np.imag(ev)

            if abs(re[0]) < 1e-6 and abs(re[1]) < 1e-6 and np.any(im != 0):
                typ, color = "center", "green"
            elif np.any(im != 0):
                typ, color = (("stable focus", "purple") if np.all(re < 0)
                              else ("unstable focus", "magenta"))
            elif re[0] * re[1] < 0:
                typ, color = "saddle", "red"
            else:
                typ, color = (("stable node", "blue") if np.all(re < 0)
                              else ("unstable node", "cyan"))

            pt, = ax.plot(xf, yf, "o", color=color, ms=8)
            txt = ax.text(xf, yf, typ, color=color,
                          fontsize="small", va="bottom", ha="right")
            self.nullcline_pts += [pt, txt]
            self.equilibria.append({'x': xf, 'y': yf, 'type': typ,
                                    'eigvals': ev, 'eigvecs': evec})
            row = self.eq_table.rowCount()
            self.eq_table.insertRow(row)
            for col, val in enumerate([
                xf, yf, typ,
                f"{ev[0]:.3g}", f"{ev[1]:.3g}"
            ]):
                self.eq_table.setItem(row, col, QTableWidgetItem(str(val)))

        handles = [
            Line2D([], [], marker="o", color="red",     linestyle="", label="saddle"),
            Line2D([], [], marker="o", color="blue",    linestyle="", label="stable node"),
            Line2D([], [], marker="o", color="cyan",    linestyle="", label="unstable node"),
            Line2D([], [], marker="o", color="purple",  linestyle="", label="stable focus"),
            Line2D([], [], marker="o", color="magenta", linestyle="", label="unstable focus"),
            Line2D([], [], marker="o", color="green",   linestyle="", label="center")
        ]
        ax.legend(handles=handles, fontsize="small", loc="upper right")
        self.phase_canvas.canvas.draw_idle()

    def _toggle_vector_field(self, chk: bool):
        self.show_field = chk
        if chk:
            self._draw_vector_field()
        else:
            for art in self.field_art:
                art.remove()
            self.field_art.clear()
            self.phase_canvas.canvas.draw_idle()

    def _draw_vector_field(self):
        for art in self.field_art:
            art.remove()
        self.field_art.clear()
        if not self.current_mu:
            return
        μ1, μ2 = self.current_mu
        ax = self.phase_canvas.ax
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        XX, YY = np.meshgrid(
            np.linspace(xmin, xmax, 20),
            np.linspace(ymin, ymax, 20)
        )
        U = self.f_lam(XX, YY, μ1, μ2)
        V = self.g_lam(XX, YY, μ1, μ2)
        M = np.hypot(U, V)
        M[M == 0] = 1
        Q = ax.quiver(XX, YY, U / M, V / M,
                      angles="xy", pivot="mid", alpha=0.6)
        self.field_art.append(Q)
        self.phase_canvas.canvas.draw_idle()

    def _toggle_separatrices(self, chk: bool):
        if chk:
            self._draw_separatrices()
        else:
            for ln in self.sep_lines:
                ln.remove()
            self.sep_lines.clear()
            self.phase_canvas.canvas.draw_idle()

    def _draw_separatrices(self):
        for ln in self.sep_lines:
            ln.remove()
        self.sep_lines.clear()
        if not self.current_mu:
            return
        μ1, μ2 = self.current_mu
        ax = self.phase_canvas.ax
        for eq in self.equilibria:
            if eq['type'] != "saddle":
                continue
            x0, y0 = eq['x'], eq['y']
            ev, vec = eq['eigvals'], eq['eigvecs']
            for i in (0, 1):
                lam = float(np.real(ev[i]))
                if abs(lam) < 1e-4:
                    continue
                v = np.real_if_close(vec[:, i])
                v /= np.linalg.norm(v)
                for sgn in (+1, -1):
                    start = np.array([x0, y0]) + sgn * 5e-3 * v
                    t_span = (0, 6) if lam > 0 else (0, -6)
                    sol = solve_ivp(lambda t, y: self.rhs_func(t, y, μ1, μ2),
                                    t_span, start,
                                    max_step=0.2, rtol=1e-4, atol=1e-7)
                    ln, = ax.plot(sol.y[0], sol.y[1], 'k--', lw=1)
                    self.sep_lines.append(ln)
        self.phase_canvas.canvas.draw_idle()

    def _toggle_hopf(self, chk: bool):
        ax = self.hopf_canvas.ax
        ax.clear()

        if chk:
            # Переключаемся на вкладку Hopf
            self.tabs.setCurrentIndex(self.hopf_tab_index)

            # Быстро рисуем все предварительно сгенерированные ветви
            for phi_fun, xi_fun, yi_fun in self.hopf_branches:
                # Дискретизация по μ1
                m1_vals = np.linspace(self.range["mu1_min"],
                                      self.range["mu1_max"], 400)
                m2_vals = phi_fun(m1_vals)

                # Отсев по диапазону μ2
                mask_mu = np.isfinite(m2_vals) & \
                          (m2_vals >= self.range["mu2_min"]) & \
                          (m2_vals <= self.range["mu2_max"])
                m1_f, m2_f = m1_vals[mask_mu], m2_vals[mask_mu]

                # Проверка фазовых границ равновесий
                x_f = xi_fun(m1_f, m2_f)
                y_f = yi_fun(m1_f, m2_f)
                mask_xy = (x_f >= self.phase_range["x_min"]) & (x_f <= self.phase_range["x_max"]) & \
                          (y_f >= self.phase_range["y_min"]) & (y_f <= self.phase_range["y_max"])

                # Рисуем участки, попавшие в окно фазовой плоскости
                ax.plot(m1_f[mask_xy], m2_f[mask_xy],
                        color="white", linewidth=2, label="trJ = 0")

            ax.set_title("Hopf curve: trJ = 0")
            ax.set_xlabel("μ1")
            ax.set_ylabel("μ2")
            ax.legend(loc="upper right")

        else:
            # Возвращаемся на вкладку Plots
            self.tabs.setCurrentIndex(0)

        self.hopf_canvas.canvas.draw_idle()

    def _on_phase_click(self, e):
        ax = self.phase_canvas.ax
        # правый клик — удаляем близкую траекторию
        if e.button == 3 and e.inaxes == ax:
            self._delete_trajectory_at(e.xdata, e.ydata)
            return
        # левый двойной клик — добавляем траекторию
        if e.button == 1 and e.dblclick and self.current_mu:
            x0, y0 = e.xdata, e.ydata
            μ1, μ2 = self.current_mu
            method = self.integrator_cb.currentText()

            def rhs_sat(t, s):
                dx, dy = self.rhs_func(t, s, μ1, μ2)
                if not np.isfinite(dx) or not np.isfinite(dy):
                    return np.array([0.0, 0.0])
                V_max = 1e3
                v = np.hypot(dx, dy)
                if v > V_max:
                    dx, dy = dx * V_max / v, dy * V_max / v
                return np.array([dx, dy])

            cx = (self.phase_range["x_min"] + self.phase_range["x_max"]) / 2
            cy = (self.phase_range["y_min"] + self.phase_range["y_max"]) / 2
            rx = (self.phase_range["x_max"] - self.phase_range["x_min"]) / 2
            ry = (self.phase_range["y_max"] - self.phase_range["y_min"]) / 2

            def stop_out(t, y):
                return max(abs(y[0] - cx) - rx, abs(y[1] - cy) - ry)
            stop_out.terminal = True

            T, N = 15, 300
            sol_f = solve_ivp(
                rhs_sat, (0, T), [x0, y0],
                t_eval=np.linspace(0, T, N // 2),
                method=method, max_step=0.2,
                rtol=1e-4, atol=1e-7, events=stop_out
            )
            sol_b = solve_ivp(
                rhs_sat, (0, -T), [x0, y0],
                t_eval=np.linspace(0, -T, N // 2),
                method=method, max_step=0.2,
                rtol=1e-4, atol=1e-7, events=stop_out
            )

            xb = sol_b.y[0][::-1][:-1]
            yb = sol_b.y[1][::-1][:-1]
            xf, yf = sol_f.y
            xs = np.concatenate([xb, xf])
            ys = np.concatenate([yb, yf])

            ln, = ax.plot(xs, ys, color=next(self.color_cycle))
            self.traj_lines.append(ln)
            self.xt_data.append((np.concatenate([sol_b.t[::-1][:-1], sol_f.t]), xs))
            self.phase_canvas.canvas.draw_idle()
            self.act_xt.setEnabled(True)

    def _delete_trajectory_at(self, x_click: float, y_click: float):
        if not self.traj_lines:
            return
        ax = self.phase_canvas.ax
        xr = ax.get_xlim()[1] - ax.get_xlim()[0]
        yr = ax.get_ylim()[1] - ax.get_ylim()[0]
        thresh = 0.05 * max(xr, yr)
        for i, ln in enumerate(self.traj_lines):
            d = np.hypot(ln.get_xdata() - x_click, ln.get_ydata() - y_click)
            if d.min() < thresh:
                ln.remove()
                del self.traj_lines[i]
                del self.xt_data[i]
                self.phase_canvas.canvas.draw_idle()
                if not self.traj_lines:
                    self.act_xt.setEnabled(False)
                return

    def _plot_xt(self):
        if not self.xt_data:
            QMessageBox.information(self, "x(t)", "Нет данных. Сначала постройте траекторию.")
            return
        if self.xt_fig is None:
            self.xt_fig, self.xt_ax = plt.subplots()
            self.xt_ax.set_title("x(t) for all trajectories")
            self.xt_ax.set_xlabel("t")
            self.xt_ax.set_ylabel("x(t)")
        self.xt_ax.cla()
        self.xt_ax.set_title("x(t) for all trajectories")
        self.xt_ax.set_xlabel("t")
        self.xt_ax.set_ylabel("x(t)")
        for t_arr, x_arr in self.xt_data:
            self.xt_ax.plot(t_arr, x_arr)
        self.xt_fig.canvas.draw()
        self.xt_fig.show()

    def _clear_trajectories(self):
        for ln in self.traj_lines:
            ln.remove()
        self.traj_lines.clear()
        self.phase_canvas.canvas.draw_idle()
        self.xt_data.clear()
        if self.xt_ax is not None:
            self.xt_ax.cla()
            self.xt_ax.set_title("x(t) for all trajectories")
            self.xt_ax.set_xlabel("t")
            self.xt_ax.set_ylabel("x(t)")
            self.xt_fig.canvas.draw()
        self.act_xt.setEnabled(False)

    def _edit_system(self):
        dlg = SystemDialog(self.f_expr, self.g_expr, self)
        if dlg.exec():
            self.f_expr, self.g_expr = dlg.texts()
            try:
                self._compile_system()
                if self.current_mu:
                    self._draw_nullclines(*self.current_mu)
                if self.show_field:
                    self._draw_vector_field()
                if self.act_separatrices.isChecked():
                    self._draw_separatrices()
                if self.act_hopf.isChecked():
                    self._toggle_hopf(True)
                self.statusBar().showMessage("System updated")
            except Exception as e:
                QMessageBox.critical(self, "Error", str(e))

    def _edit_range(self):
        dlg = RangeDialog(self.range, self)
        if dlg.exec():
            self.range = dlg.values()
            self.mu1_spin.setRange(self.range["mu1_min"], self.range["mu1_max"])
            self.mu2_spin.setRange(self.range["mu2_min"], self.range["mu2_max"])
            self._configure_param_axes()
            self.statusBar().showMessage("Parameter range updated")

    def _edit_phase_range(self):
        dlg = PhaseRangeDialog(self.phase_range, self)
        if dlg.exec():
            self.phase_range = dlg.values()
            self._configure_phase_axes()
            self.statusBar().showMessage("Phase range updated")

def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
