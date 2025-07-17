# -*- coding: utf-8 -*-

from __future__ import annotations
import sys, itertools
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
    QApplication, QMainWindow, QWidget, QDialog,
    QVBoxLayout, QHBoxLayout, QSplitter,
    QPushButton, QFormLayout, QDialogButtonBox,
    QComboBox, QDoubleSpinBox, QLabel, QPlainTextEdit,
    QMessageBox, QTabWidget, QTableWidget, QTableWidgetItem
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

        # создаём единый Figure с двумя осями
        self.fig, (self.ax_phase, self.ax_xt) = plt.subplots(1, 2, figsize=(9, 4))

        # холст и тулбар для сохранения/масштабирования
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)

        # собираем layout
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
        self.phase_range = dict(x_min=-5, x_max=15, y_min=-2, y_max=2)
        self.f_expr = "-2*exp(-x) + exp(-2*x) + y"
        self.g_expr = "(-x + mu2*y + mu1)*0.01"

        # подготовка списков для _compile_system
        self.eq_funcs = []
        self.hopf_branches = []
        self.bt_pts = []
        self._compile_system()

        # — GUI-состояние —
        self.current_mu = None
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

        # — основные холсты —
        self.param_canvas = MplCanvas()
        self.phase_canvas = MplCanvas()
        self.bt_canvas    = MplCanvas()
        self.xt_canvas    = MplCanvas()   # старый x(t)

        # — создаём единое окно Phase & x(t) —
        self.phase_xt_window = PhaseXTDialog(self)

        # — тулбар главного окна —
        self.toolbar = self.addToolBar("Controls")
        self._build_toolbar()
        open_act = QAction("Phase+x(t) window", self)
        open_act.triggered.connect(self.phase_xt_window.show)
        self.toolbar.addAction(open_act)

        # — настраиваем оси главных канвасов —
        self._conf_param_axes()
        self._conf_phase_axes()

        # — табы главного окна —
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

        # в конце — триггерим первую отрисовку
        self.mu1_spin.setValue(self.mu1_spin.value())
    # -------------------- 4.1  Компиляция и BT-поиск ----------------------- #
    def _compile_system(self):
        """
        Компилирует текущее определение системы из self.f_expr и self.g_expr,
        обновляет функции правых частей, факториалы Якобиана,
        а также ищет аналитические равновесия (если возможно) и ветви Гопфа.
        """
        # объявляем символические переменные
        x, y, m1, m2 = sp.symbols("x y mu1 mu2")

        # парсим выражения
        f_sym = sp.sympify(self.f_expr)
        g_sym = sp.sympify(self.g_expr)

        # матрица Якобиана и её детерминанта/трасса
        J = sp.Matrix([f_sym, g_sym]).jacobian([x, y])
        detJ = sp.simplify(J.det())
        trJ = sp.simplify(J.trace())

        # лямбда-функции для численных вычислений
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

        # сбрасываем предыдущие результаты
        self.eq_funcs.clear()
        self.hopf_branches.clear()
        self.bt_pts.clear()

        # 1) Аналитический поиск равновесий
        try:
            sols = sp.solve([f_sym, g_sym], [x, y], dict=True)
        except NotImplementedError:
            sols = []
        for sol in sols:
            xi, yi = sol[x], sol[y]
            # создаём функции xi(mu1,mu2), yi(mu1,mu2)
            xi_f = sp.lambdify((m1, m2), xi, modules=[SAFE_MODULE, 'numpy'])
            yi_f = sp.lambdify((m1, m2), yi, modules=[SAFE_MODULE, 'numpy'])
            self.eq_funcs.append((xi_f, yi_f))
            # ищем ветви Гопфа: решаем trJ=0 по m2
            trJ_eq = trJ.subs({x: xi, y: yi})
            roots = []
            try:
                roots = sp.solve(trJ_eq, m2)
            except (NotImplementedError, ValueError):
                roots = []
            if roots:
                phi = roots[0]
                phi_f = sp.lambdify(m1, phi, modules=[SAFE_MODULE, 'numpy'])
                self.hopf_branches.append((phi_f, xi_f, yi_f))

        # 2) Численный поиск Bogdanov–Takens-точек
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
            sol = root(F, guess, method='hybr', tol=1e-8)
            if not sol.success:
                continue
            x0, y0, u, v = sol.x
            # проверяем, что в пределах user-defined диапазонов
            if not (self.range['mu1_min'] <= u <= self.range['mu1_max']
                    and self.range['mu2_min'] <= v <= self.range['mu2_max']):
                continue
            # уникальность по параметрам
            if any(abs(u - uu) < 1e-6 and abs(v - vv) < 1e-6 for _, _, uu, vv in pts):
                continue
            pts.append((x0, y0, u, v))
        self.bt_pts = pts

    def _compute_bt_points(self):
        def F(vars):
            xv, yv, m1v, m2v = vars
            return [
                self.f_lam(xv, yv, m1v, m2v),
                self.g_lam(xv, yv, m1v, m2v),
                self.trJ_lam(xv, yv, m1v, m2v),
                self.detJ_lam(xv, yv, m1v, m2v)
            ]
        guesses = [(0,0,1,0), (0.5,0.5,1,-1), (-0.5,-0.5,1,-1)]
        pts = []
        for g in guesses:
            sol = root(F, g, method='hybr', tol=1e-8)
            if sol.success:
                x0,y0,u,v = sol.x
                if all(self.range[k] <= val <= self.range[k.replace('_min','_max')]
                       for k,val in zip(['mu1_min','mu2_min'], [u,v])):
                    if not any(abs(u-pp[2])<1e-6 and abs(v-pp[3])<1e-6 for pp in pts):
                        pts.append((x0,y0,u,v))
        self.bt_pts = pts

    def _detect_limit_cycle(self, sol):
        t, x, y = sol.t, sol.y[0], sol.y[1]
        crosses = []
        for i in range(len(t)-1):
            if y[i]<0<=y[i+1]:
                frac = -y[i]/(y[i+1]-y[i])
                crosses.append(t[i] + frac*(t[i+1]-t[i]))
        if len(crosses)<5: return None,None,None
        P = np.diff(crosses)[-3:]
        if np.std(P)/np.mean(P)<0.05:
            T = np.mean(P)
            t0,t1 = crosses[-2],crosses[-1]
            mask = (t>=t0)&(t<=t1)
            return x[mask], y[mask], T
        return None,None,None

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
        self.rtol_spin.setSingleStep(1e-4);
        self.rtol_spin.setValue(1e-4)
        bar.addWidget(self.rtol_spin)

        bar.addWidget(QLabel("atol:"))
        self.atol_spin = QDoubleSpinBox(decimals=8, minimum=1e-12, maximum=1.0)
        self.atol_spin.setSingleStep(1e-4);
        self.atol_spin.setValue(1e-7)
        bar.addWidget(self.atol_spin)

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

    def _conf_param_axes(self):
        ax=self.param_canvas.ax;ax.clear()
        ax.set_title("Parameter plane (μ1, μ2)")
        ax.set_xlabel("μ1");ax.set_ylabel("μ2")
        ax.set_xlim(self.range["mu1_min"],self.range["mu1_max"])
        ax.set_ylim(self.range["mu2_min"],self.range["mu2_max"])
        self.param_canvas.canvas.draw_idle()

    def _conf_phase_axes(self):
        ax=self.phase_canvas.ax;ax.clear()
        ax.set_title("Phase plane (x, y)")
        ax.set_xlabel("x");ax.set_ylabel("y")
        ax.set_xlim(self.phase_range["x_min"],self.phase_range["x_max"])
        ax.set_ylim(self.phase_range["y_min"],self.phase_range["y_max"])
        self.phase_canvas.canvas.draw_idle()

    def _connect_events(self):
        self.param_canvas.canvas.mpl_connect("button_press_event",self._param_click)
        self.phase_canvas.canvas.mpl_connect("button_press_event",self._phase_click)

    # -------------------- 4.3  BT-вкладка ------------------------------ #
    def _toggle_bt(self,chk):
        ax=self.bt_canvas.ax;ax.clear();self.bt_table.setRowCount(0)
        if chk:
            self.tabs.setCurrentIndex(self.bt_tab_index)
            for x0,y0,u,v in self.bt_pts:
                pass  # аналогично вашему коду
        else:
            self.tabs.setCurrentIndex(0)
        self.bt_canvas.canvas.draw_idle()

    def _spin_changed(self, _=None):
        μ1, μ2 = self.mu1_spin.value(), self.mu2_spin.value()
        self.current_mu = (μ1, μ2)
        self._clear_traj()
        self._toggle_sep(False)

        # обновляем маркер и основной портрет
        if self.param_marker:
            self.param_marker.remove()
        self.param_marker, = self.param_canvas.ax.plot(μ1, μ2, "xr", ms=8)
        self.param_canvas.canvas.draw_idle()
        self._draw_nullclines(μ1, μ2)
        if self.show_field:
            self._draw_vect()

        # **ЭТО** добавляет фазовый портрет в окно-диалог
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

    def _find_eq(self,μ1,μ2):
        sols=[];tol_f,tol_xy=1e-4,1e-3
        xmn,xmx=self.phase_range["x_min"],self.phase_range["x_max"]
        ymn,ymx=self.phase_range["y_min"],self.phase_range["y_max"]
        for x0 in np.linspace(xmn,xmx,10):
            for y0 in np.linspace(ymn,ymx,10):
                sol=root(lambda v:[np.tanh(self.f_lam(v[0],v[1],μ1,μ2)),
                                    np.tanh(self.g_lam(v[0],v[1],μ1,μ2))],
                         [x0,y0],method="hybr",options={"maxfev":200,"xtol":1e-6})
                if sol.success:
                    xe,ye=sol.x
                    if max(abs(self.f_lam(xe,ye,μ1,μ2)),abs(self.g_lam(xe,ye,μ1,μ2)))>tol_f: continue
                    if any(np.hypot(xe-xs,ye-ys)<tol_xy for xs,ys in sols): continue
                    sols.append((round(xe,6),round(ye,6)))
        return sols

    def _draw_nullclines(self, μ1, μ2):
        ax = self.phase_canvas.ax
        # Очистка предыдущих артефактов
        for cs in self.nullcline_art: cs.remove()
        for txt in self.nullcline_pts: txt.remove()
        self.nullcline_art.clear()
        self.nullcline_pts.clear()
        self.eq_table.setRowCount(0)
        self.equilibria.clear()

        # Сетка для контуров
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        xx, yy = np.meshgrid(
            np.linspace(xmin, xmax, 300),
            np.linspace(ymin, ymax, 300)
        )

        # Вычисляем F и G
        with np.errstate(over='ignore', invalid='ignore'):
            F = np.nan_to_num(self.f_lam(xx, yy, μ1, μ2))
            G = np.nan_to_num(self.g_lam(xx, yy, μ1, μ2))

        # Рисуем нулевые кривые (изоклины)
        cf = ax.contour(xx, yy, F, levels=[0], colors="blue", linestyles="--", linewidths=2.01)
        cg = ax.contour(xx, yy, G, levels=[0], colors="green", linestyles="--", linewidths=2)
        self.nullcline_art.extend([cf, cg])

        # Рисуем равновесия
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

            if det < 0:
                typ, color = "saddle", "red"
            else:
                if discr > 1e-6:
                    typ, color = ("stable node", "blue") if tr < 0 else ("unstable node", "cyan")
                else:
                    if abs(tr) < 1e-6:
                        typ, color = "center", "green"
                    elif tr < 0:
                        typ, color = "stable focus", "purple"
                    else:
                        typ, color = "unstable focus", "magenta"

            pt, = ax.plot(xf, yf, "o", color=color, ms=8)
            txt = ax.text(xf, yf, typ, color=color, fontsize="small", va="bottom", ha="right")
            self.nullcline_pts.extend([pt, txt])

            row = self.eq_table.rowCount()
            self.eq_table.insertRow(row)
            vals = (xf, yf, typ) + tuple(np.linalg.eigvals([[J11, J12], [J21, J22]]))
            for col, v in enumerate(vals):
                text = f"{v:.6g}" if isinstance(v, (int, float, np.floating)) else str(v)
                self.eq_table.setItem(row, col, QTableWidgetItem(text))

            M = np.array([[J11, J12],
                          [J21, J22]])
            ev, vec = np.linalg.eig(M)

            self.equilibria.append({
                'x': xf,
                'y': yf,
                'type': typ,
                'eigvals': ev,  # массив собственных значений
                'eigvecs': vec  # матрица (2×2) собственных векторов
            })

        ax.legend(handles=[
            Line2D([], [], marker="o", color="red", linestyle="", label="saddle"),
            Line2D([], [], marker="o", color="blue", linestyle="", label="stable node"),
            Line2D([], [], marker="o", color="cyan", linestyle="", label="unstable node"),
            Line2D([], [], marker="o", color="purple", linestyle="", label="stable focus"),
            Line2D([], [], marker="o", color="magenta", linestyle="", label="unstable focus"),
            Line2D([], [], marker="o", color="green", linestyle="", label="center")
        ], fontsize="small", loc="upper right")

        self.phase_canvas.canvas.draw_idle()

        # Обновляем отдельное окно
        self._draw_window_phase(μ1, μ2)
        if self.show_field:
            self._draw_window_field(μ1, μ2)

    def _draw_window_phase(self, μ1, μ2):
        ax = self.phase_xt_window.ax_phase
        ax.clear()

        # Сетка для контуров (используем phase_range)
        xmin, xmax = self.phase_range["x_min"], self.phase_range["x_max"]
        ymin, ymax = self.phase_range["y_min"], self.phase_range["y_max"]
        xx, yy = np.meshgrid(
            np.linspace(xmin, xmax, 1000),
            np.linspace(ymin, ymax, 1000)
        )

        # F и G
        with np.errstate(over='ignore', invalid='ignore'):
            F = np.nan_to_num(self.f_lam(xx, yy, μ1, μ2))
            G = np.nan_to_num(self.g_lam(xx, yy, μ1, μ2))

        # Рисуем нулевые кривые
        ax.contour(xx, yy, F, levels=[0], colors="blue", linestyles="--", linewidths=2)
        ax.contour(xx, yy, G, levels=[0], colors="green", linestyles="--", linewidths=2)

        # Рисуем равновесия
        for xf, yf in self._find_eq(μ1, μ2):
            # тип и цвет по тем же правилам
            J11 = float(self.J11(xf, yf, μ1, μ2))
            J22 = float(self.J22(xf, yf, μ1, μ2))
            tr = J11 + J22
            det = float(self.detJ_lam(xf, yf, μ1, μ2))
            discr = tr * tr - 4 * det
            # 2) определяем нестабильность
            unstable = (det < 0) or (tr > 0)
            # 3) рисуем
            if unstable:
                # неустойчивый узел, седло или спираль
                ax.plot(
                    xf, yf, 'o',
                    markerfacecolor='white',  # mfc – заливка
                    markeredgecolor='black',  # mec – обводка
                    markeredgewidth=1.5,  # mew – толщина обводки
                    markersize=8,
                    linestyle='None'
                )
            else:
                # устойчивый узел или спираль
                ax.plot(xf, yf, 'o', color='black', ms=8)

        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        # ax.set_title("Phase plane")
        ax.set_xlabel("x")
        ax.set_ylabel("y")

        self.phase_xt_window.canvas.draw_idle()

    def _draw_vect(self):
        """
        Рисует нормированное векторное поле на главном холсте и дублирует
        на оконном холсте.
        """
        if not self.current_mu:
            return
        μ1, μ2 = self.current_mu

        # сначала очистим старое поле
        self._clear_vect()

        # главный холст
        ax = self.phase_canvas.ax
        xlim, ylim = ax.get_xlim(), ax.get_ylim()
        XX, YY = np.meshgrid(
            np.linspace(*xlim, 20),
            np.linspace(*ylim, 20)
        )
        U = self.f_lam(XX, YY, μ1, μ2)
        V = self.g_lam(XX, YY, μ1, μ2)
        M = np.hypot(U, V)
        M[M == 0] = 1
        Q = ax.quiver(XX, YY, U / M, V / M, angles="xy", pivot="mid", alpha=0.6)
        self.field_art.append(Q)
        self.phase_canvas.canvas.draw_idle()

        # дублируем в окно-диалог
        self._draw_window_field(μ1, μ2)

    def _draw_window_field(self, μ1, μ2):
        """
        Рисует векторное поле точно так же, но в ax_phase отдельного окна.
        """
        ax = self.phase_xt_window.ax_phase
        # очистка старого
        for art in self.field_art_win:
            art.remove()
        self.field_art_win.clear()

        xlim, ylim = ax.get_xlim(), ax.get_ylim()
        XX, YY = np.meshgrid(
            np.linspace(*xlim, 20),
            np.linspace(*ylim, 20)
        )
        U = self.f_lam(XX, YY, μ1, μ2)
        V = self.g_lam(XX, YY, μ1, μ2)
        M = np.hypot(U, V)
        M[M == 0] = 1
        Qw = ax.quiver(XX, YY, U / M, V / M, angles="xy", pivot="mid", alpha=0.6)
        self.field_art_win.append(Qw)
        self.phase_xt_window.canvas.draw_idle()

    def _toggle_vect(self, chk):
        """
        Переключает показ векторного поля:
        если chk=True, рисует на обоих холстах;
        иначе — очищает.
        """
        self.show_field = chk
        if chk:
            self._draw_vect()
        else:
            self._clear_vect()

    def _clear_vect(self):
        """
        Убирает все векторы с главного и оконного холстов.
        """
        # главный холст
        for art in self.field_art:
            art.remove()
        self.field_art.clear()
        # оконный холст
        for art in self.field_art_win:
            art.remove()
        self.field_art_win.clear()
        # перерисовка
        self.phase_canvas.canvas.draw_idle()
        self.phase_xt_window.canvas.draw_idle()

    def _toggle_sep(self,chk):
        if chk: self._draw_sep()
        else: self._clear_sep()

    def _clear_sep(self):
        """
        Убирает все сепаратрисы с основного холста и из окна Phase & x(t).
        """
        # Основной холст
        for ln in self.sep_lines:
            ln.remove()
        self.sep_lines.clear()

        # Окно Phase & x(t)
        for ln in self.sep_lines_win:
            ln.remove()
        self.sep_lines_win.clear()

        # Перерисовать оба холста
        self.phase_canvas.canvas.draw_idle()
        self.phase_xt_window.canvas.draw_idle()

    def _draw_sep(self):
        """
        Строит сепаратрисы (по седловым равновесиям) на основном холсте
        и дублирует их в окно Phase & x(t).
        """
        # Сначала очищаем предыдущие сепаратрисы
        self._clear_sep()

        if not self.current_mu:
            return

        μ1, μ2 = self.current_mu
        ax_main = self.phase_canvas.ax
        ax_win = self.phase_xt_window.ax_phase

        # Проходим по всем равновесиям
        for eq in self.equilibria:
            if eq['type'] != "saddle":
                continue
            x0, y0 = eq['x'], eq['y']
            ev, vec = eq['eigvals'], eq['eigvecs']

            # Для каждого собственного направления
            for i in (0, 1):
                lam = float(np.real(ev[i]))
                v = np.real_if_close(vec[:, i])
                if abs(lam) < 1e-4:
                    continue
                v = v / np.linalg.norm(v)

                for sgn in (+1, -1):
                    start = np.array([x0, y0]) + sgn * 5e-3 * v
                    span = (0, 500) if lam > 0 else (0, -90)

                    sol = solve_ivp(
                        lambda t, s: self.rhs_func(t, s, μ1, μ2),
                        span,
                        start,
                        max_step=0.2,
                        rtol=float(self.rtol_spin.value()),
                        atol=float(self.atol_spin.value())
                    )

                    # На основном холсте
                    ln_main, = ax_main.plot(sol.y[0], sol.y[1], 'r:', lw=2)
                    self.sep_lines.append(ln_main)

                    # И в окне Phase & x(t)
                    ln_win, = ax_win.plot(sol.y[0], sol.y[1], 'r:', lw=2)
                    self.sep_lines_win.append(ln_win)

        # Перерисовать оба холста
        self.phase_canvas.canvas.draw_idle()
        self.phase_xt_window.canvas.draw_idle()

    def _phase_click(self, e):
        ax = self.phase_canvas.ax
        # правый клик — удаляем ближайшую траекторию
        if e.button == 3 and e.inaxes == ax:
            self._del_traj(e.xdata, e.ydata)
            return
        # двойной левый — интегрируем только если заданы параметры
        if e.button != 1 or not e.dblclick or not self.current_mu:
            return

        x0, y0 = e.xdata, e.ydata
        μ1, μ2 = self.current_mu

        # ограничивающая правая часть
        def rhs_sat(t, s):
            dx, dy = self.rhs_func(t, s, μ1, μ2)
            vmax = 1e3
            v = np.hypot(dx, dy)
            return np.array([dx, dy]) * (vmax / v if v > vmax else 1)

        # считываем спинбоксы
        t0 = float(self.t0_spin.value())
        t1 = float(self.t1_spin.value())
        rtol = float(self.rtol_spin.value())
        atol = float(self.atol_spin.value())
        method = self.integrator_cb.currentText()

        # формируем сетки для интегрирования
        N = 1000
        t_full = np.linspace(t0, t1, N)
        t_bwd = t_full[t_full <= 0]
        t_fwd = t_full[t_full >= 0]

        # интегрирование «назад»: 0 → t0
        sol_bwd = solve_ivp(
            rhs_sat,
            (0, t0),
            [x0, y0],
            method=method,
            t_eval=t_bwd[::-1],  # перевёрнутый массив
            rtol=rtol,
            atol=atol,
            max_step=0.1
        )

        # интегрирование «вперёд»: 0 → t1
        sol_fwd = solve_ivp(
            rhs_sat,
            (0, t1),
            [x0, y0],
            method=method,
            t_eval=t_fwd,
            rtol=rtol,
            atol=atol,
            max_step=0.1
        )

        # восстанавливаем порядок для «назад»
        tb = sol_bwd.t[::-1]
        xb, yb = sol_bwd.y[0][::-1], sol_bwd.y[1][::-1]

        # «вперёд» (пропускаем дублирующий t=0)
        tf = sol_fwd.t
        xf, yf = sol_fwd.y

        # склеиваем всю траекторию
        t_vals = np.concatenate([tb, tf[1:]])
        xs = np.concatenate([xb, xf[1:]])
        ys = np.concatenate([yb, yf[1:]])

        color = next(self.color_cycle)

        # рисуем на основном холсте
        ln, = ax.plot(xs, ys, color=color)
        self.traj_lines.append(ln)

        # рисуем в окне-диалоге
        axw = self.phase_xt_window.ax_phase
        ln_win, = axw.plot(xs, ys, color=color)
        self.traj_lines_win.append(ln_win)
        self.phase_xt_window.canvas.draw_idle()

        # сохраняем данные для x(t) и обновляем график x(t)
        self.xt_data.append((t_vals, xs, color))
        self._update_xt()

        self.phase_canvas.canvas.draw_idle()
        self.act_xt.setEnabled(True)

    def _del_traj(self, x, y):
        if not self.traj_lines:
            return
        ax = self.phase_canvas.ax
        thresh = 0.01 * max(ax.get_xlim()[1] - ax.get_xlim()[0],
                            ax.get_ylim()[1] - ax.get_ylim()[0])
        for i, ln in enumerate(self.traj_lines):
            if np.min(np.hypot(ln.get_xdata() - x, ln.get_ydata() - y)) < thresh:
                # удаляем с основного
                ln.remove()
                self.traj_lines.pop(i)
                # и с окна-диалога
                ln_win = self.traj_lines_win.pop(i)
                ln_win.remove()
                # удаляем данные x(t)
                self.xt_data.pop(i)
                # перерисуем оба холста
                self.phase_canvas.canvas.draw_idle()
                self.phase_xt_window.canvas.draw_idle()
                self._update_xt()
                break

    def _clear_traj(self):
        # на основном
        for ln in self.traj_lines:
            ln.remove()
        self.traj_lines.clear()
        # в окне-диалоге
        for ln_win in self.traj_lines_win:
            ln_win.remove()
        self.traj_lines_win.clear()
        # данные для x(t)
        self.xt_data.clear()

        self.phase_canvas.canvas.draw_idle()
        self.phase_xt_window.canvas.draw_idle()
        self._update_xt()
        self.act_xt.setEnabled(False)

    def _update_xt(self):
        # основной canvas
        ax = self.xt_canvas.ax
        ax.clear()
        ax.set_title("x(t)")
        ax.set_xlabel("t");
        ax.set_ylabel("x(t)")
        for t_vals, x_vals, color in self.xt_data:
            ax.plot(t_vals, x_vals, color=color)
        self.xt_canvas.canvas.draw_idle()

        # окно–диалог
        axw = self.phase_xt_window.ax_xt
        axw.clear()
        # axw.set_title("x(t) (window)")
        axw.set_xlabel("Время, t");
        axw.set_ylabel("x(t)")
        for t_vals, x_vals, color in self.xt_data:
            axw.plot(t_vals, x_vals, color=color)
        self.phase_xt_window.canvas.draw_idle()

    def _dlg_system(self):
        dlg=SystemDialog(self.f_expr,self.g_expr,self)
        if dlg.exec():
            self.f_expr,self.g_expr=dlg.texts()
            try:
                self._compile_system()
                if self.current_mu:
                    μ1,μ2=self.current_mu
                    self._draw_nullclines(μ1,μ2)
                    if self.show_field: self._draw_vect()
                    self._draw_window_phase(μ1,μ2)
                    if self.show_field: self._draw_window_field(μ1,μ2)
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

    def _dlg_phase(self):
        dlg=PhaseRangeDialog(self.phase_range,self)
        if dlg.exec():
            self.phase_range=dlg.values()
            self._conf_phase_axes()

def main():
    app=QApplication(sys.argv)
    win=MainWindow()
    win.show()
    sys.exit(app.exec())

if __name__=="__main__":
    main()
# -5,001558