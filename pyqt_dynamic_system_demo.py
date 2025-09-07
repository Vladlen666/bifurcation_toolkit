# -*- coding: utf-8 -*-

from __future__ import annotations
import sys, itertools, warnings
from collections.abc import Callable

import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp
from scipy.optimize import root

import matplotlib
# Для PyQt6 рекомендуемый бэкенд:
matplotlib.use("QtAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.lines import Line2D

from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QDialog,
    QVBoxLayout, QHBoxLayout, QSplitter,
    QPushButton, QFormLayout, QDialogButtonBox,
    QComboBox, QDoubleSpinBox, QLabel, QPlainTextEdit,
    QMessageBox, QTabWidget, QTableWidget, QTableWidgetItem, QSpinBox
)

# --------------------------------------------------------------------------- #
# 1. «Безопасная» экспонента                                                  #
# --------------------------------------------------------------------------- #
EXP_CLIP = 50.0
def safe_exp(z):
    return np.exp(np.clip(z, -EXP_CLIP, EXP_CLIP))
SAFE_MODULE = {'exp': safe_exp, 'safe_exp': safe_exp, 'np': np}

# --------------------------------------------------------------------------- #
# 2. Вспомогательные диалоги                                                  #
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
        box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok |
                               QDialogButtonBox.StandardButton.Cancel)
        box.accepted.connect(self.accept)
        box.rejected.connect(self.reject)
        form.addRow(box)

    def texts(self):
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
        box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok |
                               QDialogButtonBox.StandardButton.Cancel)
        box.accepted.connect(self.accept)
        box.rejected.connect(self.reject)
        form.addRow(box)

    def values(self):
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
        box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok |
                               QDialogButtonBox.StandardButton.Cancel)
        box.accepted.connect(self.accept)
        box.rejected.connect(self.reject)
        form.addRow(box)

    def values(self):
        return {k: w.value() for k, w in self.edits.items()}


# --------------------------------------------------------------------------- #
# 3. Matplotlib-канвас                                                        #
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
        s = 1.2 if e.button == "up" else 0.8
        xm, ym = e.xdata, e.ydata
        ax = e.inaxes
        ax.set_xlim(xm + (ax.get_xlim()[0] - xm) * s,
                    xm + (ax.get_xlim()[1] - xm) * s)
        ax.set_ylim(ym + (ax.get_ylim()[0] - ym) * s,
                    ym + (ax.get_ylim()[1] - ym) * s)
        self.canvas.draw_idle()

class PhaseXTDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Phase & x(t)")
        self.resize(900, 500)

        self.fig, (self.ax_phase, self.ax_xt) = plt.subplots(1, 2, figsize=(9, 4))
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.addWidget(self.canvas)
        layout.addWidget(self.toolbar)

# --------------------------------------------------------------------------- #
# 4. Главное окно                                                             #
# --------------------------------------------------------------------------- #
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Dynamic-System Sandbox — BT only")
        self.resize(1500, 650)

        # — диапазоны и формулы системы —
        self.range = dict(mu1_min=-15, mu1_max=15, mu2_min=-15, mu2_max=15)
        self.phase_range = dict(x_min=-5, x_max=25, y_min=-4, y_max=2)
        self.f_expr = "-2*exp(-x) + exp(-2*x) + y"
        self.g_expr = "(-x + mu2*y + mu1)*0.01"

        # подготовка под компиляцию
        self.eq_funcs: list[tuple[Callable, Callable]] = []
        self.hopf_branches: list[tuple[Callable, Callable|None, Callable|None]] = []
        self.bt_pts: list[tuple[float, float, float, float]] = []
        self._compile_system()

        # — GUI-состояние —
        self.current_mu: tuple[float, float] | None = None
        self.traj_lines = []
        self.xt_data     = []
        self.traj_lines_win = []

        self.nullcline_art = []
        self.nullcline_pts = []
        self.field_art   = []
        self.field_art_win = []
        self.sep_lines   = []
        self.sep_lines_win = []
        self.equilibria  = []
        self.show_field  = False
        self.param_marker = None
        self.color_cycle = itertools.cycle(
            plt.rcParams["axes.prop_cycle"].by_key()["color"]
        )

        # SDE/шум: максимальный внутренний шаг интеграции
        self.sde_max_step = 0.1

        # — основные холсты —
        self.param_canvas = MplCanvas()
        self.phase_canvas = MplCanvas()
        self.bt_canvas    = MplCanvas()
        self.xt_canvas    = MplCanvas()   # старый x(t)

        # — окно Phase & x(t) —
        self.phase_xt_window = PhaseXTDialog(self)

        # — тулбар —
        self.toolbar = self.addToolBar("Controls")
        self._build_toolbar()
        open_act = QAction("Phase+x(t) window", self)
        open_act.triggered.connect(self.phase_xt_window.show)
        self.toolbar.addAction(open_act)

        # — настройка осей —
        self._conf_param_axes()
        self._conf_phase_axes()

        # — табы —
        spl = QSplitter(Qt.Orientation.Horizontal)
        spl.addWidget(self.param_canvas)
        spl.addWidget(self.phase_canvas)

        self.tabs = QTabWidget()
        w_plots = QWidget()
        QVBoxLayout(w_plots).addWidget(spl)
        self.tabs.addTab(w_plots, "Plots")

        self.eq_table = QTableWidget(0, 5)
        self.eq_table.setHorizontalHeaderLabels(["x", "y", "type", "λ₁", "λ₂"])
        self.tabs.addTab(self.eq_table, "Equilibria")

        self.bt_table = QTableWidget(0, 6)
        self.bt_table.setHorizontalHeaderLabels(["x", "y", "μ₁", "μ₂", "λ₁", "λ₂"])
        bt_w = QWidget()
        bt_l = QVBoxLayout(bt_w)
        bt_l.addWidget(self.bt_canvas)
        bt_l.addWidget(self.bt_table)
        self.tabs.addTab(bt_w, "BT")
        self.bt_tab_index = self.tabs.indexOf(bt_w)

        self.setCentralWidget(self.tabs)
        self._connect_events()

        # первая отрисовка
        self.mu1_spin.setValue(self.mu1_spin.value())

    # -------------------- 4.1  Компиляция и BT-поиск ----------------------- #
    def _compile_system(self):
        """
        Компилирует текущую систему из self.f_expr и self.g_expr,
        формирует численные лямбда-функции, ветви Гопфа и точки БТ.
        """
        x, y, m1, m2 = sp.symbols("x y mu1 mu2")

        f_sym = sp.sympify(self.f_expr)
        g_sym = sp.sympify(self.g_expr)

        J = sp.Matrix([f_sym, g_sym]).jacobian([x, y])
        detJ = sp.simplify(J.det())
        trJ = sp.simplify(J.trace())

        # lambdify
        self.f_lam = sp.lambdify((x, y, m1, m2), f_sym, modules=[SAFE_MODULE, 'numpy'])
        self.g_lam = sp.lambdify((x, y, m1, m2), g_sym, modules=[SAFE_MODULE, 'numpy'])
        self.J11 = sp.lambdify((x, y, m1, m2), J[0, 0], modules=[SAFE_MODULE, 'numpy'])
        self.J12 = sp.lambdify((x, y, m1, m2), J[0, 1], modules=[SAFE_MODULE, 'numpy'])
        self.J21 = sp.lambdify((x, y, m1, m2), J[1, 0], modules=[SAFE_MODULE, 'numpy'])
        self.J22 = sp.lambdify((x, y, m1, m2), J[1, 1], modules=[SAFE_MODULE, 'numpy'])
        self.detJ_lam = sp.lambdify((x, y, m1, m2), detJ, modules=[SAFE_MODULE, 'numpy'])
        self.trJ_lam = sp.lambdify((x, y, m1, m2), trJ, modules=[SAFE_MODULE, 'numpy'])
        self.rhs_func = lambda t, s, μ1, μ2: np.array([
            self.f_lam(s[0], s[1], μ1, μ2),
            self.g_lam(s[0], s[1], μ1, μ2)
        ])

        # сброс
        self.eq_funcs.clear()
        self.hopf_branches.clear()
        self.bt_pts.clear()

        # 1) Аналитические равновесия и ветви Hopf (trJ=0)
        try:
            sols = sp.solve([f_sym, g_sym], [x, y], dict=True)
        except NotImplementedError:
            sols = []
        for sol in sols:
            xi, yi = sol[x], sol[y]
            xi_f = sp.lambdify((m1, m2), xi, modules=[SAFE_MODULE, 'numpy'])
            yi_f = sp.lambdify((m1, m2), yi, modules=[SAFE_MODULE, 'numpy'])
            self.eq_funcs.append((xi_f, yi_f))

            trJ_eq = sp.simplify(trJ.subs({x: xi, y: yi}))
            roots = []
            try:
                roots = sp.solve(trJ_eq, m2)
                if not isinstance(roots, (list, tuple)):
                    roots = [roots]
            except Exception:
                roots = []

            for phi in roots:
                try:
                    phi_f = sp.lambdify(m1, phi, modules=[SAFE_MODULE, 'numpy'])
                except Exception:
                    phi_f = None
                self.hopf_branches.append((phi_f, xi_f, yi_f))

        # 2) Численный поиск Bogdanov–Takens (f=0, g=0, trJ=0, detJ=0)
        def F(vars):
            xv, yv, m1v, m2v = vars
            return [
                self.f_lam(xv, yv, m1v, m2v),
                self.g_lam(xv, yv, m1v, m2v),
                self.trJ_lam(xv, yv, m1v, m2v),
                self.detJ_lam(xv, yv, m1v, m2v)
            ]

        guesses = [
            (0.0, 0.0, 1.0, 0.0),
            (0.5, 0.5, 1.0, -1.0),
            (-0.5, -0.5, 1.0, -1.0),
            (1.0, -1.0, 1.0, -1.0),
            (2.0, 0.0, 0.5, 0.5),
            (-2.0, 0.0, 0.5, -0.5),
        ]
        pts = []
        for guess in guesses:
            try:
                sol = root(F, guess, method='hybr', tol=1e-8)
            except Exception:
                continue
            if not sol.success:
                continue
            x0, y0, u, v = sol.x
            if not (self.range['mu1_min'] <= u <= self.range['mu1_max']
                    and self.range['mu2_min'] <= v <= self.range['mu2_max']):
                continue
            if any(abs(u - uu) < 1e-6 and abs(v - vv) < 1e-6 for _, _, uu, vv in pts):
                continue
            # проверим кратность нуля (ранг Якоби)
            Jnum = np.array([[self.J11(x0, y0, u, v), self.J12(x0, y0, u, v)],
                             [self.J21(x0, y0, u, v), self.J22(x0, y0, u, v)]], dtype=float)
            ev = np.linalg.eigvals(Jnum)
            if np.allclose(ev, 0, atol=1e-4) or np.count_nonzero(np.abs(ev) < 1e-4) >= 1:
                pts.append((float(x0), float(y0), float(u), float(v)))
        self.bt_pts = pts

    # ----------------------- 4.2  UI-элементы ----------------------------- #
    def _build_toolbar(self):
        bar = self.toolbar
        bar.setIconSize(QSize(16,16))
        for txt,slot in (("Edit system",self._dlg_system),
                         ("Set param range",self._dlg_range),
                         ("Set phase range",self._dlg_phase)):
            a=QAction(txt,self);a.triggered.connect(slot);bar.addAction(a)
        bar.addSeparator()

        bar.addWidget(QLabel("μ₁:"))
        self.mu1_spin=QDoubleSpinBox(decimals=6)
        self.mu1_spin.setRange(self.range["mu1_min"],self.range["mu1_max"])
        self.mu1_spin.valueChanged.connect(self._spin_changed)
        bar.addWidget(self.mu1_spin)

        bar.addWidget(QLabel("μ₂:"))
        self.mu2_spin=QDoubleSpinBox(decimals=6)
        self.mu2_spin.setRange(self.range["mu2_min"],self.range["mu2_max"])
        self.mu2_spin.valueChanged.connect(self._spin_changed)
        bar.addWidget(self.mu2_spin)

        bar.addSeparator()
        self.integrator_cb = QComboBox()
        self.integrator_cb.addItems(["RK45", "RK23", "DOP853", "LSODA", "Radau", "BDF"])
        bar.addWidget(QLabel("Integrator:"))
        bar.addWidget(self.integrator_cb)

        bar.addSeparator()
        bar.addWidget(QLabel("t₀:"))
        self.t0_spin = QDoubleSpinBox(decimals=1, minimum=-10000, maximum=10000)
        self.t0_spin.setValue(-10.0)
        bar.addWidget(self.t0_spin)

        bar.addWidget(QLabel("t₁:"))
        self.t1_spin = QDoubleSpinBox(decimals=1, minimum=-10000, maximum=10000)
        self.t1_spin.setValue(1000.0)
        bar.addWidget(self.t1_spin)

        bar.addSeparator()
        bar.addWidget(QLabel("rtol:"))
        self.rtol_spin = QDoubleSpinBox(decimals=8, minimum=1e-12, maximum=1.0)
        self.rtol_spin.setSingleStep(1e-4)
        self.rtol_spin.setValue(1e-4)
        bar.addWidget(self.rtol_spin)

        bar.addWidget(QLabel("atol:"))
        self.atol_spin = QDoubleSpinBox(decimals=8, minimum=1e-12, maximum=1.0)
        self.atol_spin.setSingleStep(1e-4)
        self.atol_spin.setValue(1e-7)
        bar.addWidget(self.atol_spin)

        # ======== ШУМ (новое) ======== #
        bar.addSeparator()
        bar.addWidget(QLabel("Noise:"))
        self.noise_cb = QComboBox()
        self.noise_cb.addItems(["None", "White", "OU"])
        self.noise_cb.currentTextChanged.connect(self._noise_changed)
        bar.addWidget(self.noise_cb)

        bar.addWidget(QLabel("σ:"))
        self.noise_sigma_spin = QDoubleSpinBox(decimals=4, minimum=0.0, maximum=10.0)
        self.noise_sigma_spin.setSingleStep(0.01)
        self.noise_sigma_spin.setValue(0.05)
        bar.addWidget(self.noise_sigma_spin)

        bar.addWidget(QLabel("τ:"))
        self.noise_tau_spin = QDoubleSpinBox(decimals=4, minimum=1e-6, maximum=1e6)
        self.noise_tau_spin.setSingleStep(0.01)
        self.noise_tau_spin.setValue(1.0)
        self.noise_tau_spin.setEnabled(False)
        bar.addWidget(self.noise_tau_spin)

        bar.addWidget(QLabel("Seed:"))
        self.noise_seed_spin = QSpinBox()
        self.noise_seed_spin.setRange(0, 2_147_483_647)
        self.noise_seed_spin.setValue(42)
        bar.addWidget(self.noise_seed_spin)
        # ============================== #

        bar.addSeparator()
        btn = QPushButton("Clear")
        btn.clicked.connect(self._clear_traj)
        bar.addWidget(btn)

        bar.addSeparator()
        self.act_vector = QAction("Vector field", self, checkable=True)
        self.act_vector.triggered.connect(self._toggle_vect)
        bar.addAction(self.act_vector)

        bar.addSeparator()
        self.act_sep = QAction("Separatrices", self, checkable=True)
        self.act_sep.triggered.connect(self._toggle_sep)
        bar.addAction(self.act_sep)

        bar.addSeparator()
        self.act_bt = QAction("BT", self, checkable=True)
        self.act_bt.triggered.connect(self._toggle_bt)
        bar.addAction(self.act_bt)

        bar.addSeparator()
        self.act_xt = QAction("Plot x(t)", self)
        self.act_xt.setEnabled(False)
        self.act_xt.triggered.connect(self._update_xt)
        bar.addAction(self.act_xt)

    def _noise_changed(self, *_):
        kind = self.noise_cb.currentText()
        self.noise_tau_spin.setEnabled(kind == "OU")

    def _conf_param_axes(self):
        ax=self.param_canvas.ax;ax.clear()
        ax.set_title("Parameter plane (μ1, μ2)")
        ax.set_xlabel("μ1");ax.set_ylabel("μ2")
        ax.set_xlim(self.range["mu1_min"],self.range["mu1_max"])
        ax.set_ylim(self.range["mu2_min"],self.range["mu2_max"])
        self._draw_param_features()  # Hopf/BT
        self.param_canvas.canvas.draw_idle()

    def _conf_phase_axes(self):
        ax=self.phase_canvas.ax;ax.clear()
        ax.set_title("Phase plane (x, y)")
        ax.set_xlabel("x", fontsize=15, fontweight="bold")
        ax.set_ylabel("y", fontsize=15, fontweight="bold")
        ax.tick_params(axis="both", labelsize=15)
        ax.set_xlim(self.phase_range["x_min"],self.phase_range["x_max"])
        ax.set_ylim(self.phase_range["y_min"],self.phase_range["y_max"])
        self.phase_canvas.canvas.draw_idle()

    def _connect_events(self):
        self.param_canvas.canvas.mpl_connect("button_press_event",self._param_click)
        self.phase_canvas.canvas.mpl_connect("button_press_event",self._phase_click)

    # -------------------- 4.3  Параметрическая плоскость ------------------- #
    def _draw_param_features(self):
        """Рисует на параметрической плоскости ветви Hopf и точки BT."""
        ax = self.param_canvas.ax
        # очистим старые артефакты, кроме маркера текущих μ
        for art in list(ax.lines):
            if art is self.param_marker:
                continue
            art.remove()
        for art in list(ax.collections):
            art.remove()

        μ1_grid = np.linspace(self.range["mu1_min"], self.range["mu1_max"], 800)

        # Ветви Hopf (если удалось получить phi(mu1))
        hopf_handles = []
        for phi_f, xi_f, yi_f in self.hopf_branches:
            if phi_f is None:
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                mu2_vals = phi_f(μ1_grid)
            mu2_vals = np.array(mu2_vals, dtype=float)
            mask = np.isfinite(mu2_vals)
            if not np.any(mask):
                continue
            (ln,) = ax.plot(μ1_grid[mask], mu2_vals[mask], '-', lw=1.8, alpha=0.9)
            hopf_handles.append(ln)

        # Точки Bogdanov–Takens
        if self.bt_pts:
            xs = [p[2] for p in self.bt_pts]
            ys = [p[3] for p in self.bt_pts]
            ax.plot(xs, ys, 'ks', ms=7, label="BT")

        # Легенда
        labels = []
        handles = []
        if hopf_handles:
            handles.append(Line2D([], [], color=hopf_handles[0].get_color(), lw=2))
            labels.append("Hopf (trJ=0 on eq)")
        if self.bt_pts:
            handles.append(Line2D([], [], marker='s', color='k', linestyle=''))
            labels.append("BT points")
        if handles:
            ax.legend(handles, labels, fontsize="small", loc="best")

    # -------------------- 4.4  BT-вкладка ---------------------------------- #
    def _toggle_bt(self, chk):
        ax = self.bt_canvas.ax
        ax.clear()
        self.bt_table.setRowCount(0)

        if chk:
            self.tabs.setCurrentIndex(self.bt_tab_index)
            ax.set_title("Bogdanov–Takens points (zoomed)")
            ax.set_xlabel("μ1"); ax.set_ylabel("μ2")

            if not self.bt_pts:
                ax.text(0.5, 0.5, "No BT points found", ha='center', va='center', transform=ax.transAxes)
            else:
                mu1_vals = [p[2] for p in self.bt_pts]
                mu2_vals = [p[3] for p in self.bt_pts]
                ax.plot(mu1_vals, mu2_vals, 'ks', ms=7)

                pad1 = max(1e-2, 0.05*(max(mu1_vals)-min(mu1_vals) if len(mu1_vals)>1 else 1.0))
                pad2 = max(1e-2, 0.05*(max(mu2_vals)-min(mu2_vals) if len(mu2_vals)>1 else 1.0))
                ax.set_xlim(min(mu1_vals)-pad1, max(mu1_vals)+pad1)
                ax.set_ylim(min(mu2_vals)-pad2, max(mu2_vals)+pad2)

                for (x0,y0,u,v) in self.bt_pts:
                    Jnum = np.array([[self.J11(x0, y0, u, v), self.J12(x0, y0, u, v)],
                                     [self.J21(x0, y0, u, v), self.J22(x0, y0, u, v)]], dtype=float)
                    ev = np.linalg.eigvals(Jnum)
                    row = self.bt_table.rowCount()
                    self.bt_table.insertRow(row)
                    for col, val in enumerate((x0,y0,u,v,ev[0],ev[1])):
                        try:
                            txt = f"{float(np.real(val)):.6g}" if abs(np.imag(val))<1e-12 else f"{val:.6g}"
                        except Exception:
                            txt = str(val)
                        self.bt_table.setItem(row, col, QTableWidgetItem(txt))
        else:
            self.tabs.setCurrentIndex(0)

        self.bt_canvas.canvas.draw_idle()

    # -------------------- 4.5  Спины/клики/обновления ---------------------- #
    def _spin_changed(self, _=None):
        μ1, μ2 = self.mu1_spin.value(), self.mu2_spin.value()
        self.current_mu = (μ1, μ2)
        self._clear_traj()
        self._toggle_sep(False)

        if self.param_marker:
            self.param_marker.remove()
        self._draw_param_features()
        self.param_marker, = self.param_canvas.ax.plot(μ1, μ2, "xr", ms=8)
        self.param_canvas.canvas.draw_idle()

        self._draw_nullclines(μ1, μ2)
        if self.show_field:
            self._draw_vect()

        self._draw_window_phase(μ1, μ2)
        if self.show_field:
            self._draw_window_field(μ1, μ2)

        self.statusBar().showMessage(f"μ=({μ1:.6g}, {μ2:.6g})")

    def _param_click(self,e):
        if not(e.inaxes and e.dblclick): return
        μ1,μ2=e.xdata,e.ydata
        self.mu1_spin.blockSignals(True);self.mu1_spin.setValue(μ1);self.mu1_spin.blockSignals(False)
        self.mu2_spin.blockSignals(True);self.mu2_spin.setValue(μ2);self.mu2_spin.blockSignals(False)
        self._spin_changed()

    # -------------------- 4.6  Равновесия/нулклайны ------------------------ #
    def _find_eq(self,μ1,μ2):
        sols=[];tol_f,tol_xy=1e-4,1e-3
        xmn,xmx=self.phase_range["x_min"],self.phase_range["x_max"]
        ymn,ymx=self.phase_range["y_min"],self.phase_range["y_max"]
        for x0 in np.linspace(xmn,xmx,10):
            for y0 in np.linspace(ymn,ymx,10):
                try:
                    sol=root(lambda v:[np.tanh(self.f_lam(v[0],v[1],μ1,μ2)),
                                        np.tanh(self.g_lam(v[0],v[1],μ1,μ2))],
                             [x0,y0],method="hybr",options={"maxfev":200,"xtol":1e-6})
                except Exception:
                    continue
                if sol.success:
                    xe,ye=sol.x
                    if max(abs(self.f_lam(xe,ye,μ1,μ2)),abs(self.g_lam(xe,ye,μ1,μ2)))>tol_f: continue
                    if any(np.hypot(xe-xs,ye-ys)<tol_xy for xs,ys in sols): continue
                    sols.append((round(xe,6),round(ye,6)))
        return sols

    @staticmethod
    def _classify_equilibrium(tr: float, det: float, discr: float) -> tuple[str, str]:
        eps_det = 1e-4
        eps_disc = 1e-6
        eps_tr = 1e-3

        if det < -eps_det:
            return "saddle", "red"
        if (abs(det) <= eps_det) or (abs(tr) <= eps_tr and abs(discr) <= eps_disc):
            return "degenerate", "orange"
        if discr < -eps_disc:
            if abs(tr) <= eps_tr:
                return "center", "green"
            return ("stable focus", "purple") if tr < 0 else ("unstable focus", "magenta")
        else:
            return ("stable node", "blue") if tr < 0 else ("unstable node", "cyan")

    def _draw_nullclines(self, μ1, μ2):
        ax = self.phase_canvas.ax
        for cs in self.nullcline_art: cs.remove()
        for txt in self.nullcline_pts: txt.remove()
        self.nullcline_art.clear()
        self.nullcline_pts.clear()
        self.eq_table.setRowCount(0)
        self.equilibria.clear()

        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        xx, yy = np.meshgrid(
            np.linspace(xmin, xmax, 300),
            np.linspace(ymin, ymax, 300)
        )

        with np.errstate(over='ignore', invalid='ignore'):
            F = np.nan_to_num(self.f_lam(xx, yy, μ1, μ2))
            G = np.nan_to_num(self.g_lam(xx, yy, μ1, μ2))

        cf = ax.contour(xx, yy, F, levels=[0], colors="blue", linestyles="--", linewidths=3.0)
        cg = ax.contour(xx, yy, G, levels=[0], colors="green", linestyles="--", linewidths=3.0)
        self.nullcline_art.extend([cf, cg])

        for xf, yf in self._find_eq(μ1, μ2):
            if not (xmin <= xf <= xmax and ymin <= yf <= ymax):
                continue

            J11 = float(self.J11(xf, yf, μ1, μ2))
            J12 = float(self.J12(xf, yf, μ1, μ2))
            J21 = float(self.J21(xf, yf, μ1, μ2))
            J22 = float(self.J22(xf, yf, μ1, μ2))
            tr = J11 + J22
            det = float(self.detJ_lam(xf, yf, μ1, μ2))
            discr = tr * tr - 4 * det

            typ, color = self._classify_equilibrium(tr, det, discr)

            (pt,) = ax.plot(xf, yf, "o", color=color, ms=8)
            label_txt = f"{typ}"
            txt = ax.text(xf, yf, label_txt, color=color, fontsize="small", va="bottom", ha="right")
            self.nullcline_pts.extend([pt, txt])

            row = self.eq_table.rowCount()
            self.eq_table.insertRow(row)
            try:
                ev = np.linalg.eigvals([[J11, J12], [J21, J22]])
            except Exception:
                ev = [np.nan, np.nan]
            vals = (xf, yf, typ) + tuple(ev)
            for col, v in enumerate(vals):
                if col <= 1:
                    text = f"{v:.6g}"
                elif col == 2:
                    text = str(v)
                else:
                    try:
                        text = f"{float(np.real(v)):.6g}" if abs(np.imag(v))<1e-12 else f"{v:.6g}"
                    except Exception:
                        text = str(v)
                self.eq_table.setItem(row, col, QTableWidgetItem(text))

            M = np.array([[J11, J12],[J21, J22]], dtype=float)
            evs, vec = np.linalg.eig(M)

            self.equilibria.append({
                'x': xf,
                'y': yf,
                'type': typ,
                'eigvals': evs,
                'eigvecs': vec
            })

        ax.legend(handles=[
            Line2D([], [], marker="o", color="red", linestyle="", label="saddle"),
            Line2D([], [], marker="o", color="blue", linestyle="", label="stable node"),
            Line2D([], [], marker="o", color="cyan", linestyle="", label="unstable node"),
            Line2D([], [], marker="o", color="purple", linestyle="", label="stable focus"),
            Line2D([], [], marker="o", color="magenta", linestyle="", label="unstable focus"),
            Line2D([], [], marker="o", color="green", linestyle="", label="center"),
            Line2D([], [], marker="o", color="orange", linestyle="", label="degenerate"),
        ], fontsize="small", loc="upper right")

        self.phase_canvas.canvas.draw_idle()

        self._draw_window_phase(μ1, μ2)
        if self.show_field:
            self._draw_window_field(μ1, μ2)

    # -------------------- 4.7  Окно Phase --------------------------------- #
    def _draw_window_phase(self, μ1, μ2):
        ax = self.phase_xt_window.ax_phase
        ax.clear()

        xmin, xmax = self.phase_range["x_min"], self.phase_range["x_max"]
        ymin, ymax = self.phase_range["y_min"], self.phase_range["y_max"]
        xx, yy = np.meshgrid(
            np.linspace(xmin, xmax, 300),
            np.linspace(ymin, ymax, 300)
        )

        with np.errstate(over='ignore', invalid='ignore'):
            F = np.nan_to_num(self.f_lam(xx, yy, μ1, μ2))
            G = np.nan_to_num(self.g_lam(xx, yy, μ1, μ2))

        ax.contour(xx, yy, F, levels=[0], colors="blue", linestyles="--", linewidths=3.5)
        ax.contour(xx, yy, G, levels=[0], colors="green", linestyles="--", linewidths=3.5)

        for xf, yf in self._find_eq(μ1, μ2):
            J11 = float(self.J11(xf, yf, μ1, μ2))
            J22 = float(self.J22(xf, yf, μ1, μ2))
            det = float(self.detJ_lam(xf, yf, μ1, μ2))
            tr = J11 + J22
            discr = tr*tr - 4*det
            typ, _ = self._classify_equilibrium(tr, det, discr)
            unstable = (det < 0) or (tr > 0) or typ in ("unstable node","unstable focus","degenerate")
            if typ == "saddle":
                ax.plot(xf, yf, 'o', markerfacecolor='white', markeredgecolor='black',
                        markeredgewidth=1.5, markersize=20, linestyle='None', zorder=5)
                ax.plot(xf, yf, 'x', color='black', markersize=15, mew=1.5, zorder=6)
            elif unstable:
                ax.plot(xf, yf, 'o', markerfacecolor='white', markeredgecolor='black',
                        markeredgewidth=1.5, markersize=20, linestyle='None')
            else:
                ax.plot(xf, yf, 'o', color='black', ms=20)

        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_xlabel("x", fontsize=20, fontweight="bold")
        ax.set_ylabel("y", fontsize=20, fontweight="bold")
        ax.tick_params(axis="both", labelsize=17)

        self.phase_xt_window.canvas.draw_idle()

    # -------------------- 4.8  Векторное поле ------------------------------ #
    def _draw_vect(self):
        if not self.current_mu:
            return
        μ1, μ2 = self.current_mu

        self._clear_vect()

        ax = self.phase_canvas.ax
        xlim, ylim = ax.get_xlim(), ax.get_ylim()
        XX, YY = np.meshgrid(
            np.linspace(*xlim, 20),
            np.linspace(*ylim, 20)
        )
        U = self.f_lam(XX, YY, μ1, μ2)
        V = self.g_lam(XX, YY, μ1, μ2)
        M = np.hypot(U, V); M[M == 0] = 1
        Q = ax.quiver(XX, YY, U / M, V / M, angles="xy", pivot="mid", alpha=0.6)
        self.field_art.append(Q)
        self.phase_canvas.canvas.draw_idle()

        self._draw_window_field(μ1, μ2)

    def _draw_window_field(self, μ1, μ2):
        ax = self.phase_xt_window.ax_phase
        for art in self.field_art_win:
            art.remove()
        self.field_art_win.clear()

        xlim, ylim = ax.get_xlim(), ax.get_ylim()
        XX, YY = np.meshgrid(
            np.linspace(*xlim, 30),
            np.linspace(*ylim, 30)
        )
        U = self.f_lam(XX, YY, μ1, μ2)
        V = self.g_lam(XX, YY, μ1, μ2)
        M = np.hypot(U, V); M[M == 0] = 1
        Qw = ax.quiver(XX, YY, U / M, V / M, angles="xy", pivot="mid", alpha=0.6)
        self.field_art_win.append(Qw)
        self.phase_xt_window.canvas.draw_idle()

    def _toggle_vect(self, chk):
        self.show_field = chk
        if chk:
            self._draw_vect()
        else:
            self._clear_vect()

    def _clear_vect(self):
        for art in self.field_art:
            art.remove()
        self.field_art.clear()
        for art in self.field_art_win:
            art.remove()
        self.field_art_win.clear()
        self.phase_canvas.canvas.draw_idle()
        self.phase_xt_window.canvas.draw_idle()

    # -------------------- 4.9  Сепаратрисы --------------------------------- #
    def _toggle_sep(self,chk):
        if chk: self._draw_sep()
        else: self._clear_sep()

    def _clear_sep(self):
        for ln in self.sep_lines:
            ln.remove()
        self.sep_lines.clear()
        for ln in self.sep_lines_win:
            ln.remove()
        self.sep_lines_win.clear()
        self.phase_canvas.canvas.draw_idle()
        self.phase_xt_window.canvas.draw_idle()

    def _draw_sep(self):
        self._clear_sep()
        if not self.current_mu:
            return

        μ1, μ2 = self.current_mu
        ax_main = self.phase_canvas.ax
        ax_win = self.phase_xt_window.ax_phase

        for eq in self.equilibria:
            if eq['type'] != "saddle":
                continue
            x0, y0 = eq['x'], eq['y']
            ev, vec = eq['eigvals'], eq['eigvecs']

            for i in (0, 1):
                lam = float(np.real(ev[i]))
                v = np.real_if_close(vec[:, i])
                if np.linalg.norm(v) < 1e-12 or abs(lam) < 1e-4:
                    continue
                v = v / np.linalg.norm(v)

                Tmax = 1000  # время интегрирования сепаратрис
                for sgn in (+1, -1):
                    start = np.array([x0, y0]) + sgn * 5e-3 * v
                    if lam > 0:  # неустойчивое направление → интегрируем вперёд
                        span = (0, Tmax)
                    else:  # устойчивое направление → интегрируем назад
                        span = (0, 0)

                    sol = solve_ivp(
                        lambda t, s: self.rhs_func(t, s, μ1, μ2),
                        span,
                        start,
                        max_step=0.2,
                        rtol=float(self.rtol_spin.value()),
                        atol=float(self.atol_spin.value())
                    )

                    (ln_main,) = ax_main.plot(sol.y[0], sol.y[1], 'r:', lw=6)
                    self.sep_lines.append(ln_main)

                    (ln_win,) = ax_win.plot(sol.y[0], sol.y[1], 'r:', lw=6)
                    self.sep_lines_win.append(ln_win)

        self.phase_canvas.canvas.draw_idle()
        self.phase_xt_window.canvas.draw_idle()

    # -------------------- 4.10  Интегрирование/траектории ------------------ #
    def _phase_click(self, e):
        ax = self.phase_canvas.ax
        if e.button == 3 and e.inaxes == ax:
            self._del_traj(e.xdata, e.ydata)
            return
        if e.button != 1 or not e.dblclick or not self.current_mu:
            return

        x0, y0 = e.xdata, e.ydata
        μ1, μ2 = self.current_mu

        def rhs_sat(t, s):
            dx, dy = self.rhs_func(t, s, μ1, μ2)
            vmax = 1e3
            v = np.hypot(dx, dy)
            return np.array([dx, dy]) * (vmax / v if v > vmax else 1)

        t0 = float(self.t0_spin.value())
        t1 = float(self.t1_spin.value())
        rtol = float(self.rtol_spin.value())
        atol = float(self.atol_spin.value())
        method = self.integrator_cb.currentText()

        N = 1000
        t_full = np.linspace(t0, t1, N)
        t_bwd = t_full[t_full <= 0]
        t_fwd = t_full[t_full >= 0]

        use_noise = (self.noise_cb.currentText() != "None") and (float(self.noise_sigma_spin.value()) > 0.0)

        if not use_noise:
            sol_bwd = solve_ivp(
                rhs_sat, (0, t0), [x0, y0],
                method=method, t_eval=t_bwd[::-1],
                rtol=rtol, atol=atol, max_step=0.1
            )
            sol_fwd = solve_ivp(
                rhs_sat, (0, t1), [x0, y0],
                method=method, t_eval=t_fwd,
                rtol=rtol, atol=atol, max_step=0.1
            )

            tb = sol_bwd.t[::-1]
            xb, yb = sol_bwd.y[0][::-1], sol_bwd.y[1][::-1]
            tf = sol_fwd.t
            xf, yf = sol_fwd.y
        else:
            seed = int(self.noise_seed_spin.value())
            rng_b = np.random.default_rng(seed)
            rng_f = np.random.default_rng(seed + 1)

            # обратная часть (0 -> t0), t_eval убывает от 0 к t0
            t_eval_bw = t_bwd[::-1]
            y_bw = self._sde_path(np.array([x0, y0], dtype=float), t_eval_bw, μ1, μ2, rng_b)
            tb = t_bwd
            xb, yb = y_bw[:,0][::-1], y_bw[:,1][::-1]

            # прямая часть (0 -> t1)
            t_eval_fw = t_fwd
            y_fw = self._sde_path(np.array([x0, y0], dtype=float), t_eval_fw, μ1, μ2, rng_f)
            tf = t_eval_fw
            xf, yf = y_fw[:,0], y_fw[:,1]

        t_vals = np.concatenate([tb, tf[1:]])
        xs = np.concatenate([xb, xf[1:]])
        ys = np.concatenate([yb, yf[1:]])

        color = next(self.color_cycle)

        (ln,) = ax.plot(xs, ys, color=color)
        self.traj_lines.append(ln)

        axw = self.phase_xt_window.ax_phase
        step = max(1, len(xs) // 20)
        (ln_win,) = axw.plot(xs, ys, color=color, lw=2.5)
        self.traj_lines_win.append(ln_win)

        self.phase_xt_window.canvas.draw_idle()

        self.xt_data.append((t_vals, xs, color))
        self._update_xt()

        self.phase_canvas.canvas.draw_idle()
        self.act_xt.setEnabled(True)

    def _sde_path(self, y0: np.ndarray, t_eval: np.ndarray, μ1: float, μ2: float, rng: np.random.Generator) -> np.ndarray:
        """
        Интеграция траектории с шумом по схеме Эйлера–Маруямы.
        Два вида шума:
          - "White":   x <- x + f dt + σ * sqrt(|dt|) * ξ
          - "OU":      x <- x + f dt + η dt,   dη = -(η/τ) dt + sqrt(2 σ^2 / τ) dW
        Шум добавляется к обоим уравнениям (x и y) аддитивно.
        """
        kind = self.noise_cb.currentText()
        sigma = float(self.noise_sigma_spin.value())
        tau = float(self.noise_tau_spin.value())

        y = y0.astype(float).copy()
        out = np.empty((len(t_eval), 2), dtype=float)
        out[0] = y
        eta = np.zeros(2, dtype=float)  # OU состояние

        for k in range(1, len(t_eval)):
            dt_tot = float(t_eval[k] - t_eval[k-1])
            nsteps = max(1, int(np.ceil(abs(dt_tot) / self.sde_max_step)))
            dt = dt_tot / nsteps
            sdt = np.sqrt(abs(dt))
            for _ in range(nsteps):
                # насыщенный детерминированный дрейф
                dx, dy = self.rhs_func(0.0, y, μ1, μ2)
                vmax = 1e3
                v = np.hypot(dx, dy)
                if v > vmax:
                    scale = vmax / v
                    dx *= scale; dy *= scale
                drift = np.array([dx, dy], dtype=float)

                if kind == "White":
                    dW = rng.normal(size=2) * sigma * sdt
                    y += drift * dt + dW
                elif kind == "OU":
                    # обновляем OU-процесс (стационарная дисперсия sigma^2)
                    eta += (-eta / tau) * dt + np.sqrt(2.0 * sigma * sigma / tau) * sdt * rng.normal(size=2)
                    y += drift * dt + eta * dt
                else:
                    y += drift * dt
            out[k] = y
        return out

    def _del_traj(self, x, y):
        if not self.traj_lines:
            return
        ax = self.phase_canvas.ax
        thresh = 0.01 * max(ax.get_xlim()[1] - ax.get_xlim()[0],
                            ax.get_ylim()[1] - ax.get_ylim()[0])
        for i, ln in enumerate(self.traj_lines):
            try:
                dist = np.min(np.hypot(ln.get_xdata() - x, ln.get_ydata() - y))
            except Exception:
                continue
            if dist < thresh:
                ln.remove()
                self.traj_lines.pop(i)
                ln_win = self.traj_lines_win.pop(i)
                ln_win.remove()
                self.xt_data.pop(i)
                self.phase_canvas.canvas.draw_idle()
                self.phase_xt_window.canvas.draw_idle()
                self._update_xt()
                break

    def _clear_traj(self):
        for ln in self.traj_lines:
            ln.remove()
        self.traj_lines.clear()
        for ln_win in self.traj_lines_win:
            ln_win.remove()
        self.traj_lines_win.clear()
        self.xt_data.clear()

        self.phase_canvas.canvas.draw_idle()
        self.phase_xt_window.canvas.draw_idle()
        self._update_xt()
        self.act_xt.setEnabled(False)

    def _update_xt(self):
        ax = self.xt_canvas.ax
        ax.clear()
        ax.set_title("x(t)")
        ax.set_xlabel("t", fontsize=15, fontweight="bold")
        ax.set_ylabel("x(t)", fontsize=15, fontweight="bold")
        ax.tick_params(axis="both", labelsize=15)
        for t_vals, x_vals, color in self.xt_data:
            ax.plot(t_vals, x_vals, color=color)
        self.xt_canvas.canvas.draw_idle()

        axw = self.phase_xt_window.ax_xt
        axw.clear()
        axw.set_xlabel("t", fontsize=20, fontweight="bold")
        axw.set_ylabel("x(t)", fontsize=20, fontweight="bold")
        axw.tick_params(axis="both", labelsize=17)
        for t_vals, x_vals, color in self.xt_data:
            axw.plot(t_vals, x_vals, color=color, lw=3)
        self.phase_xt_window.canvas.draw_idle()

    # -------------------- 4.11  Диалоги ------------------------------------ #
    def _dlg_system(self):
        dlg=SystemDialog(self.f_expr,self.g_expr,self)
        if dlg.exec():
            self.f_expr,self.g_expr=dlg.texts()
            try:
                self._compile_system()
                self._replot_all()
                self.statusBar().showMessage("System updated")
            except Exception as e:
                QMessageBox.critical(self,"Error",str(e))

    def _dlg_range(self):
        dlg=RangeDialog(self.range,self)
        if dlg.exec():
            self.range=dlg.values()
            self.mu1_spin.setRange(self.range["mu1_min"],self.range["mu1_max"])
            self.mu2_spin.setRange(self.range["mu2_min"],self.range["mu2_max"])
            self._compile_system()
            self._conf_param_axes()
            if self.current_mu:
                μ1, μ2 = self.current_mu
                if self.param_marker:
                    self.param_marker.remove()
                self.param_marker, = self.param_canvas.ax.plot(μ1, μ2, "xr", ms=8)
                self.param_canvas.canvas.draw_idle()

    def _dlg_phase(self):
        dlg=PhaseRangeDialog(self.phase_range,self)
        if dlg.exec():
            self.phase_range=dlg.values()
            self._conf_phase_axes()
            if self.current_mu:
                self._draw_nullclines(*self.current_mu)
                if self.show_field: self._draw_vect()
                self._draw_window_phase(*self.current_mu)
                if self.show_field: self._draw_window_field(*self.current_mu)

    def _replot_all(self):
        if self.current_mu:
            μ1, μ2 = self.current_mu
            self._conf_param_axes()
            if self.param_marker:
                self.param_marker.remove()
            self.param_marker, = self.param_canvas.ax.plot(μ1, μ2, "xr", ms=8)
            self.param_canvas.canvas.draw_idle()

            self._draw_nullclines(μ1, μ2)
            if self.show_field: self._draw_vect()
            self._draw_window_phase(μ1, μ2)
            if self.show_field: self._draw_window_field(μ1, μ2)

# --------------------------------------------------------------------------- #
# 5. main                                                                     #
# --------------------------------------------------------------------------- #
def main():
    app=QApplication(sys.argv)
    win=MainWindow()
    win.show()
    sys.exit(app.exec())

if __name__=="__main__":
    main()
