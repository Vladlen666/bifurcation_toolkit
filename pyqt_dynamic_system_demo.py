# -*- coding: utf-8 -*-
"""Dynamic-System Sandbox — Bogdanov–Takens only (robust, 2025-07-06) — full version with BT table"""

from __future__ import annotations
import sys, itertools
from collections.abc import Callable

import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp
from scipy.optimize import root, root_scalar
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
# 1. «Безопасная» экспонента                                                  #
# --------------------------------------------------------------------------- #
EXP_CLIP = 50.0  # exp(±50) ≈ 5·10²¹

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

# --------------------------------------------------------------------------- #
# 4. Главное окно                                                             #
# --------------------------------------------------------------------------- #
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Dynamic-System Sandbox — BT only")
        self.resize(1100, 650)

        # диапазоны
        self.range = dict(mu1_min=-15, mu1_max=15, mu2_min=-15, mu2_max=15)
        self.phase_range = dict(x_min=-15, x_max=15, y_min=-15, y_max=15)
        # система
        self.f_expr = "-2*exp(-x) + exp(-2*x) + y"
        self.g_expr = "(-x + mu2*y + mu1)*0.01"

        # списки
        self.eq_funcs = []
        self.hopf_branches = []
        self.bt_pts = []
        # компиляция и BT
        self._compile_system()

        # GUI-состояние
        self.current_mu = None
        self.traj_lines = []
        self.xt_data = []
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

        # канвасы
        self.param_canvas = MplCanvas()
        self.phase_canvas = MplCanvas()
        self.bt_canvas = MplCanvas()
        spl = QSplitter(Qt.Orientation.Horizontal)
        spl.addWidget(self.param_canvas)
        spl.addWidget(self.phase_canvas)

        # вкладки
        tabs = QTabWidget()
        self.tabs = tabs
        w_plots = QWidget()
        QVBoxLayout(w_plots).addWidget(spl)
        tabs.addTab(w_plots, "Plots")
        # Equilibria
        self.eq_table = QTableWidget(0, 5)
        self.eq_table.setHorizontalHeaderLabels(["x", "y", "type", "λ₁", "λ₂"])
        tabs.addTab(self.eq_table, "Equilibria")
        # BT
        self.bt_table = QTableWidget(0, 6)
        self.bt_table.setHorizontalHeaderLabels(["x", "y", "μ₁", "μ₂", "λ₁", "λ₂"])
        bt_widget = QWidget()
        bt_layout = QVBoxLayout(bt_widget)
        bt_layout.setContentsMargins(2, 2, 2, 2)
        bt_layout.addWidget(self.bt_canvas)
        bt_layout.addWidget(self.bt_table)
        tabs.addTab(bt_widget, "BT")
        self.bt_tab_index = tabs.indexOf(bt_widget)

        self.setCentralWidget(tabs)

        # тулбар и события
        self._build_toolbar()
        self._conf_param_axes()
        self._conf_phase_axes()
        self._connect_events()

        self.xt_fig = None
        self.xt_ax = None

    # -------------------- 4.1  Компиляция и BT-поиск ----------------------- #
    def _compile_system(self):
        x, y, m1, m2 = sp.symbols("x y mu1 mu2")
        f_sym = sp.sympify(self.f_expr)
        g_sym = sp.sympify(self.g_expr)
        J = sp.Matrix([f_sym, g_sym]).jacobian([x, y])
        detJ, trJ = sp.simplify(J.det()), sp.simplify(J.trace())

        # лямбда-функции для RHS и Якобиана
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

        self.eq_funcs.clear()
        self.hopf_branches.clear()
        # поиск аналитических равновесий и ветвей Hopf
        try:
            for sol in sp.solve([f_sym, g_sym], [x, y], dict=True):
                xi, yi = sol[x], sol[y]
                xi_f = sp.lambdify((m1, m2), xi, modules=[SAFE_MODULE, 'numpy'])
                yi_f = sp.lambdify((m1, m2), yi, modules=[SAFE_MODULE, 'numpy'])
                self.eq_funcs.append((xi_f, yi_f))
                trJ_eq = trJ.subs({x: xi, y: yi})
                roots = sp.solve(trJ_eq, m2)
                if not roots:
                    continue
                phi = roots[0]
                phi_f = sp.lambdify(m1, phi, modules=[SAFE_MODULE, 'numpy'])
                self.hopf_branches.append((phi_f, xi_f, yi_f))
        except (sp.SympifyError, NotImplementedError):
            pass

        self._compute_bt_points()

    def _compute_bt_points(self):
        """Ищет BT-точки, конвертируя сначала в численные функции, а потом решая корни через SciPy."""
        # лямбда-функции из компиляции
        f_num = self.f_lam
        g_num = self.g_lam
        trJ_num = self.trJ_lam
        detJ_num = self.detJ_lam

        def F(vars):
            xv, yv, m1v, m2v = vars
            return [
                f_num(xv, yv, m1v, m2v),
                g_num(xv, yv, m1v, m2v),
                trJ_num(xv, yv, m1v, m2v),
                detJ_num(xv, yv, m1v, m2v)
            ]

        guesses = [
            ( 0.0,  0.0,  1.0,  0.0),
            ( 0.5,  0.5,  1.0, -1.0),
            (-0.5, -0.5,  1.0, -1.0),
            ( 1.0, -1.0,  1.0, -1.0),
            ( 2.0,  0.0,  0.5,  0.5),
            (-2.0,  0.0,  0.5, -0.5),
        ]

        pts: list[tuple[float, float, float, float]] = []
        for guess in guesses:
            sol = root(F, guess, method='hybr', tol=1e-8)
            if not sol.success:
                continue
            x_val, y_val, m1_val, m2_val = sol.x
            if not (self.range['mu1_min'] <= m1_val <= self.range['mu1_max'] and
                    self.range['mu2_min'] <= m2_val <= self.range['mu2_max']):
                continue
            if any(np.hypot(m1_val - u[2], m2_val - u[3]) < 1e-6 for u in pts):
                continue
            pts.append((x_val, y_val, m1_val, m2_val))

        self.bt_pts = pts
        if not pts:
            print("No BT points found in current parameter range.")
        else:
            print("\nBogdanov–Takens points (numeric):")
            for i, (x0, y0, u, v) in enumerate(pts, 1):
                print(f"{i:>2}:  x={x0:.6g}, y={y0:.6g}, μ1={u:.6g}, μ2={v:.6g}")
        print("-" * 50)

    # ----------------------- 4.2  UI-элементы ----------------------------- #
    def _build_toolbar(self):
        bar = self.addToolBar("Controls")
        bar.setIconSize(QSize(16, 16))
        for txt, slot in (("Edit system", self._dlg_system),
                          ("Set param range", self._dlg_range),
                          ("Set phase range", self._dlg_phase)):
            a = QAction(txt, self)
            a.triggered.connect(slot)
            bar.addAction(a)
        bar.addSeparator()
        bar.addWidget(QLabel("μ₁:"))
        self.mu1_spin = QDoubleSpinBox(decimals=3)
        self.mu1_spin.valueChanged.connect(self._spin_changed)
        bar.addWidget(self.mu1_spin)
        bar.addWidget(QLabel("μ₂:"))
        self.mu2_spin = QDoubleSpinBox(decimals=3)
        self.mu2_spin.valueChanged.connect(self._spin_changed)
        bar.addWidget(self.mu2_spin)
        bar.addSeparator()
        self.integrator_cb = QComboBox()
        self.integrator_cb.addItems(["RK45", "RK23", "DOP853", "LSODA", "Radau", "BDF"])
        bar.addWidget(QLabel("Integrator:"))
        bar.addWidget(self.integrator_cb)
        bar.addSeparator()
        btn = QPushButton("Clear")
        btn.clicked.connect(self._clear_traj)
        bar.addWidget(btn)
        bar.addSeparator()
        self.act_vector = QAction("Vector field", self, checkable=True)
        self.act_vector.triggered.connect(self._toggle_vect)
        bar.addAction(self.act_vector)
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
        ax = self.param_canvas.ax
        ax.clear()
        ax.set_title("Parameter plane (μ1, μ2)")
        ax.set_xlabel("μ1")
        ax.set_ylabel("μ2")
        ax.set_xlim(self.range["mu1_min"], self.range["mu1_max"])
        ax.set_ylim(self.range["mu2_min"], self.range["mu2_max"])
        self.param_canvas.canvas.draw_idle()

    def _conf_phase_axes(self):
        ax = self.phase_canvas.ax
        ax.clear()
        ax.set_title("Phase plane (x, y)")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_xlim(self.phase_range["x_min"], self.phase_range["x_max"])
        ax.set_ylim(self.phase_range["y_min"], self.phase_range["y_max"])
        self.phase_canvas.canvas.draw_idle()

    def _connect_events(self):
        self.param_canvas.canvas.mpl_connect("button_press_event", self._param_click)
        self.phase_canvas.canvas.mpl_connect("button_press_event", self._phase_click)

    # ----------------------- 4.3  BT-вкладка ------------------------------ #
    def _toggle_bt(self, chk):
        ax = self.bt_canvas.ax
        ax.clear()
        # очистка таблицы
        self.bt_table.setRowCount(0)
        if chk:
            self.tabs.setCurrentIndex(self.bt_tab_index)
            self._compute_bt_points()
            ax.set_title("Bogdanov–Takens points")
            ax.set_xlabel("μ1")
            ax.set_ylabel("μ2")
            ax.set_xlim(self.range["mu1_min"], self.range["mu1_max"])
            ax.set_ylim(self.range["mu2_min"], self.range["mu2_max"])
            if self.bt_pts:
                μ1s, μ2s = zip(*(pt[2:] for pt in self.bt_pts))
                ax.scatter(μ1s, μ2s,
                           marker="s", s=60,
                           facecolors="none",
                           edgecolors="red",
                           linewidths=1.4,
                           label="BT")
                ax.legend(loc="upper right", fontsize="small")
                # заполнение таблицы
                for x0, y0, m1, m2 in self.bt_pts:
                    # собственные числа Якобиана
                    J = np.array([
                        [self.J11(x0, y0, m1, m2), self.J12(x0, y0, m1, m2)],
                        [self.J21(x0, y0, m1, m2), self.J22(x0, y0, m1, m2)]
                    ], float)
                    λ1, λ2 = np.linalg.eigvals(J)
                    row = self.bt_table.rowCount()
                    self.bt_table.insertRow(row)
                    for col, val in enumerate((x0, y0, m1, m2, λ1, λ2)):
                        item = QTableWidgetItem(f"{val:.6g}")
                        self.bt_table.setItem(row, col, item)
        else:
            self.tabs.setCurrentIndex(0)
        self.bt_canvas.canvas.draw_idle()

    # ----------------------- 4.4  Спины/клики параметров ------------------ #
    def _spin_changed(self,_=None):
        μ1,μ2=self.mu1_spin.value(),self.mu2_spin.value()
        self.current_mu=(μ1,μ2)
        self._clear_traj(); self._toggle_sep(False)
        if self.param_marker: self.param_marker.remove()
        self.param_marker,=self.param_canvas.ax.plot(μ1,μ2,"xr",ms=8)
        self.param_canvas.canvas.draw_idle()
        self._draw_nullclines(μ1,μ2)
        if self.show_field: self._draw_vect()
        self.statusBar().showMessage(f"μ=({μ1:.3g}, {μ2:.3g})")

    def _param_click(self,e):
        if not(e.inaxes and e.dblclick): return
        μ1,μ2=e.xdata,e.ydata
        self.mu1_spin.blockSignals(True); self.mu1_spin.setValue(μ1); self.mu1_spin.blockSignals(False)
        self.mu2_spin.blockSignals(True); self.mu2_spin.setValue(μ2); self.mu2_spin.blockSignals(False)
        self._spin_changed()

    # ----------------------- 4.5  Изоклины и равновесия ------------------- #
    def _find_eq(self,μ1,μ2):
        x_min,x_max=self.phase_range["x_min"],self.phase_range["x_max"]
        y_min,y_max=self.phase_range["y_min"],self.phase_range["y_max"]
        sols,tol_f,tol_xy=[],1e-4,1e-3
        guesses=[(x0,y0) for x0 in np.linspace(x_min,x_max,10)
                          for y0 in np.linspace(y_min,y_max,10)]
        fg=lambda v:(self.f_lam(v[0],v[1],μ1,μ2),
                     self.g_lam(v[0],v[1],μ1,μ2))
        for x0,y0 in guesses:
            sol=root(lambda v:[np.tanh(fg(v)[0]),np.tanh(fg(v)[1])],
                     [x0,y0],method="hybr",options={"maxfev":200,"xtol":1e-6})
            if sol.success:
                xe,ye=sol.x
                if max(abs(fg((xe,ye))[0]),abs(fg((xe,ye))[1]))>tol_f: continue
                if any(np.hypot(xe-xs,ye-ys)<tol_xy for xs,ys in sols): continue
                sols.append((xe,ye))
        return [(round(x,6),round(y,6)) for x,y in sols]

    def _draw_nullclines(self,μ1,μ2):
        ax=self.phase_canvas.ax
        for cs in self.nullcline_art: cs.remove()
        for obj in self.nullcline_pts: obj.remove()
        self.nullcline_art.clear(); self.nullcline_pts.clear()
        self.eq_table.setRowCount(0); self.equilibria.clear()

        xmin,xmax=ax.get_xlim(); ymin,ymax=ax.get_ylim()
        xx,yy=np.meshgrid(np.linspace(xmin,xmax,300),
                          np.linspace(ymin,ymax,300))
        with np.errstate(over='ignore',invalid='ignore'):
            F=self.f_lam(xx,yy,μ1,μ2); G=self.g_lam(xx,yy,μ1,μ2)
        F=np.nan_to_num(F); G=np.nan_to_num(G)
        cf = ax.contour(xx, yy, F, levels=[0], colors="blue", linestyles="--", linewidths=1.2)
        cg = ax.contour(xx, yy, G, levels=[0], colors="green", linestyles="-", linewidths=1.2)

        self.nullcline_art+=[cf,cg]

        for xf,yf in self._find_eq(μ1,μ2):
            if not(xmin<=xf<=xmax and ymin<=yf<=ymax): continue
            J11=float(self.J11(xf,yf,μ1,μ2)); J12=float(self.J12(xf,yf,μ1,μ2))
            J21=float(self.J21(xf,yf,μ1,μ2)); J22=float(self.J22(xf,yf,μ1,μ2))
            Jmat=np.array([[J11,J12],[J21,J22]])
            tr,det=J11+J22,float(self.detJ_lam(xf,yf,μ1,μ2))
            discr=tr*tr-4*det; eigvals,eigvecs=np.linalg.eig(Jmat)
            tol=1e-6
            if det<0: typ,color="saddle","red"
            else:
                if discr>tol: typ,color=("stable node","blue") if tr<0 else ("unstable node","cyan")
                else:
                    if abs(tr)<tol: typ,color="center","green"
                    elif tr<0: typ,color="stable focus","purple"
                    else: typ,color="unstable focus","magenta"
            pt,=ax.plot(xf,yf,"o",color=color,ms=8)
            txt=ax.text(xf,yf,typ,color=color,fontsize="small",va="bottom",ha="right")
            self.nullcline_pts+=[pt,txt]

            row=self.eq_table.rowCount(); self.eq_table.insertRow(row)
            for col,val in enumerate((xf,yf,typ,*eigvals)):
                self.eq_table.setItem(row,col,QTableWidgetItem(f"{val:.6g}" if isinstance(val,float) else str(val)))

            self.equilibria.append({'x':xf,'y':yf,'type':typ,'eigvals':eigvals,'eigvecs':eigvecs})

        ax.legend(handles=[
            Line2D([],[],marker="o",color="red",linestyle="",label="saddle"),
            Line2D([],[],marker="o",color="blue",linestyle="",label="stable node"),
            Line2D([],[],marker="o",color="cyan",linestyle="",label="unstable node"),
            Line2D([],[],marker="o",color="purple",linestyle="",label="stable focus"),
            Line2D([],[],marker="o",color="magenta",linestyle="",label="unstable focus"),
            Line2D([],[],marker="o",color="green",linestyle="",label="center")
        ],fontsize="small",loc="upper right")
        self.phase_canvas.canvas.draw_idle()

    # ----------------------- 4.6  Векторное поле/сепаратрисы -------------- #
    def _toggle_vect(self,chk):
        self.show_field=chk
        (self._draw_vect() if chk else self._clear_vect())

    def _clear_vect(self):
        for art in self.field_art: art.remove()
        self.field_art.clear(); self.phase_canvas.canvas.draw_idle()

    def _draw_vect(self):
        self._clear_vect()
        if not self.current_mu:return
        μ1,μ2=self.current_mu; ax=self.phase_canvas.ax
        XX,YY=np.meshgrid(np.linspace(*ax.get_xlim(),20),
                          np.linspace(*ax.get_ylim(),20))
        U=self.f_lam(XX,YY,μ1,μ2); V=self.g_lam(XX,YY,μ1,μ2)
        M=np.hypot(U,V); M[M==0]=1
        Q=ax.quiver(XX,YY,U/M,V/M,angles="xy",pivot="mid",alpha=0.6)
        self.field_art.append(Q); self.phase_canvas.canvas.draw_idle()

    def _toggle_sep(self,chk):
        (self._draw_sep() if chk else self._clear_sep())

    def _clear_sep(self):
        for ln in self.sep_lines: ln.remove()
        self.sep_lines.clear(); self.phase_canvas.canvas.draw_idle()

    def _draw_sep(self):
        self._clear_sep()
        if not self.current_mu:return
        μ1,μ2=self.current_mu; ax=self.phase_canvas.ax
        for eq in self.equilibria:
            if eq['type']!="saddle": continue
            x0,y0=eq['x'],eq['y']; ev,vec=eq['eigvals'],eq['eigvecs']
            for i in (0,1):
                lam=float(np.real(ev[i])); v=np.real_if_close(vec[:,i])
                if abs(lam)<1e-4: continue
                v/=np.linalg.norm(v)
                for sgn in (+1,-1):
                    start=np.array([x0,y0])+sgn*5e-3*v
                    span=(0,6) if lam>0 else (0,-6)
                    sol=solve_ivp(lambda t,y:self.rhs_func(t,y,μ1,μ2),
                                  span,start,max_step=0.2,rtol=1e-4,atol=1e-7)
                    ln,=ax.plot(sol.y[0],sol.y[1],'k--',lw=1)
                    self.sep_lines.append(ln)
        self.phase_canvas.canvas.draw_idle()

    # ----------------------- 4.7  Клики по фазовой плоскости -------------- #
    def _phase_click(self,e):
        ax=self.phase_canvas.ax
        if e.button==3 and e.inaxes==ax: self._del_traj(e.xdata,e.ydata); return
        if e.button!=1 or not e.dblclick or not self.current_mu: return
        x0,y0=e.xdata,e.ydata; μ1,μ2=self.current_mu
        method=self.integrator_cb.currentText()

        def rhs_sat(t,s):
            dx,dy=self.rhs_func(t,s,μ1,μ2)
            if not np.isfinite(dx) or not np.isfinite(dy): return np.array([0,0])
            vmax=1e3; v=np.hypot(dx,dy)
            return np.array([dx,dy])*(vmax/v if v>vmax else 1)

        cx=(self.phase_range["x_min"]+self.phase_range["x_max"])/2
        cy=(self.phase_range["y_min"]+self.phase_range["y_max"])/2
        rx=(self.phase_range["x_max"]-self.phase_range["x_min"])/2
        ry=(self.phase_range["y_max"]-self.phase_range["y_min"])/2
        def stop(t,s): return max(abs(s[0]-cx)-rx,abs(s[1]-cy)-ry)
        stop.terminal=True

        T,N=15,300
        sol_f=solve_ivp(rhs_sat,(0,T),[x0,y0],t_eval=np.linspace(0,T,N//2),
                        method=method,max_step=0.2,rtol=1e-4,atol=1e-7,events=stop)
        sol_b=solve_ivp(rhs_sat,(0,-T),[x0,y0],t_eval=np.linspace(0,-T,N//2),
                        method=method,max_step=0.2,rtol=1e-4,atol=1e-7,events=stop)
        xs=np.concatenate([sol_b.y[0][::-1][:-1],sol_f.y[0]])
        ys=np.concatenate([sol_b.y[1][::-1][:-1],sol_f.y[1]])
        color=next(self.color_cycle); ln,=ax.plot(xs,ys,color=color)
        self.traj_lines.append(ln)
        t_full=np.concatenate([sol_b.t[::-1][:-1],sol_f.t])
        self.xt_data.append((t_full,xs,color))
        self.phase_canvas.canvas.draw_idle()
        self.act_xt.setEnabled(True)
        if self.xt_fig: self._update_xt()

    # ----------------------- 4.8  Управление траекториями ----------------- #
    def _del_traj(self,x,y):
        if not self.traj_lines:return
        ax=self.phase_canvas.ax
        thresh=0.05*max(ax.get_xlim()[1]-ax.get_xlim()[0],
                        ax.get_ylim()[1]-ax.get_ylim()[0])
        for i,ln in enumerate(self.traj_lines):
            if np.min(np.hypot(ln.get_xdata()-x,ln.get_ydata()-y))<thresh:
                ln.remove(); self.traj_lines.pop(i); self.xt_data.pop(i)
                self.phase_canvas.canvas.draw_idle()
                if self.xt_fig: self._update_xt() if self.traj_lines else self._clear_xt()
                break

    def _clear_traj(self):
        for ln in self.traj_lines: ln.remove()
        self.traj_lines.clear(); self.xt_data.clear()
        self.phase_canvas.canvas.draw_idle(); self._clear_xt()
        self.act_xt.setEnabled(False)

    # ----------------------- 4.9  x(t) окно -------------------------------- #
    def _update_xt(self):
        if not self.xt_data:return
        if not self.xt_fig:
            self.xt_fig,self.xt_ax=plt.subplots()
            self.xt_ax.set_xlabel("t"); self.xt_ax.set_ylabel("x(t)")
        self.xt_ax.cla(); self.xt_ax.set_title("x(t) for all trajectories")
        for t,x,color in self.xt_data: self.xt_ax.plot(t,x,color=color)
        self.xt_fig.canvas.draw_idle(); self.xt_fig.show()

    def _clear_xt(self):
        if self.xt_fig:
            self.xt_ax.cla(); self.xt_ax.set_title("x(t) for all trajectories")
            self.xt_ax.set_xlabel("t"); self.xt_ax.set_ylabel("x(t)")
            self.xt_fig.canvas.draw_idle()

    # ----------------------- 4.10  Диалоги -------------------------------- #
    def _dlg_system(self):
        dlg=SystemDialog(self.f_expr,self.g_expr,self)
        if dlg.exec():
            self.f_expr,self.g_expr=dlg.texts()
            try:
                self._compile_system()
                if self.current_mu: self._draw_nullclines(*self.current_mu)
                if self.show_field: self._draw_vect()
                if self.act_sep.isChecked(): self._draw_sep()
                if self.act_bt.isChecked(): self._toggle_bt(True)
                self.statusBar().showMessage("System updated")
            except Exception as e: QMessageBox.critical(self,"Error",str(e))

    def _dlg_range(self):
        dlg=RangeDialog(self.range,self)
        if dlg.exec():
            self.range=dlg.values()
            self.mu1_spin.setRange(self.range["mu1_min"],self.range["mu1_max"])
            self.mu2_spin.setRange(self.range["mu2_min"],self.range["mu2_max"])
            self._compile_system(); self._conf_param_axes()
            if self.act_bt.isChecked(): self._toggle_bt(True)

    def _dlg_phase(self):
        dlg=PhaseRangeDialog(self.phase_range,self)
        if dlg.exec():
            self.phase_range=dlg.values(); self._conf_phase_axes()
            if self.current_mu: self._draw_nullclines(*self.current_mu)

# --------------------------------------------------------------------------- #
# main                                                                        #
# --------------------------------------------------------------------------- #
def main():
    app=QApplication(sys.argv)
    win=MainWindow(); win.show()
    sys.exit(app.exec())

if __name__=="__main__":
    main()
