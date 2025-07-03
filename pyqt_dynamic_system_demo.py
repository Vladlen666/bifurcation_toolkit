# -*- coding: utf-8 -*-
"""pyqt_dynamic_system_demo.py — Двух-параметрическая песочница с бифуркациями Hopf
=================================================================
• Выбор μ₁,μ₂ через спинбоксы или двойной клик; синхронизация.
• Отрисовка изоклин, классификация СР (saddle, node, focus, center).
• Vector field, Separatrices для седел.
• Hopf: условие trJ=0, detJ>0, d(trJ)/dμ₁≠0 → кривая на плоскости параметров.
• Два режима: Plots и Equilibria (таблица равновесий с λ₁, λ₂).
• Настройка диапазонов, выбор интегратора, Clear, Max Param/Phase.
"""

from __future__ import annotations
import sys, itertools
from collections.abc import Callable

import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp

import matplotlib; matplotlib.use("QtAgg")
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


class SystemDialog(QDialog):
    def __init__(self, f_text: str, g_text: str, parent: QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle("Edit system")
        self.f_edit = QPlainTextEdit(f_text)
        self.g_edit = QPlainTextEdit(g_text)
        form = QFormLayout(self)
        form.addRow("f(x, y; μ1, μ2) =", self.f_edit)
        form.addRow("g(x, y; μ1, μ2) =", self.g_edit)
        box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok |
                               QDialogButtonBox.StandardButton.Cancel)
        box.accepted.connect(self.accept)
        box.rejected.connect(self.reject)
        form.addRow(box)
    def texts(self) -> tuple[str, str]:
        return self.f_edit.toPlainText(), self.g_edit.toPlainText()


class RangeDialog(QDialog):
    def __init__(self, rng: dict[str, float], parent: QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle("Parameter range")
        self.edits: dict[str, QDoubleSpinBox] = {}
        form = QFormLayout(self)
        for key in ("mu1_min", "mu1_max", "mu2_min", "mu2_max"):
            spin = QDoubleSpinBox(minimum=-1e6, maximum=1e6, decimals=4)
            spin.setValue(rng[key])
            form.addRow(key, spin)
            self.edits[key] = spin
        box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok |
                               QDialogButtonBox.StandardButton.Cancel)
        box.accepted.connect(self.accept)
        box.rejected.connect(self.reject)
        form.addRow(box)
    def values(self) -> dict[str, float]:
        return {k: w.value() for k, w in self.edits.items()}


class PhaseRangeDialog(QDialog):
    def __init__(self, pr: dict[str, float], parent: QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle("Phase range")
        self.edits: dict[str, QDoubleSpinBox] = {}
        form = QFormLayout(self)
        for key, label in (("x_min", "x min"), ("x_max", "x max"),
                           ("y_min", "y min"), ("y_max", "y max")):
            spin = QDoubleSpinBox(minimum=-1e6, maximum=1e6, decimals=4)
            spin.setValue(pr[key])
            form.addRow(label, spin)
            self.edits[key] = spin
        box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok |
                               QDialogButtonBox.StandardButton.Cancel)
        box.accepted.connect(self.accept)
        box.rejected.connect(self.reject)
        form.addRow(box)
    def values(self) -> dict[str, float]:
        return {k: w.value() for k, w in self.edits.items()}


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

    def _on_scroll(self, event):
        if event.inaxes is None: return
        scale = 1.2 if event.button=="up" else 0.8
        ax = event.inaxes
        xm, ym = event.xdata, event.ydata
        xmin,xmax = ax.get_xlim(); ymin,ymax = ax.get_ylim()
        ax.set_xlim(xm + (xmin-xm)*scale, xm + (xmax-xm)*scale)
        ax.set_ylim(ym + (ymin-ym)*scale, ym + (ymax-ym)*scale)
        self.canvas.draw_idle()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Dynamic System Sandbox — бифуркации Hopf")
        self.resize(1100,650)

        # default system
        self.f_expr = "y+mu2"
        self.g_expr = "mu1*(1 - x**2)*y - x"
        self._compile_system()

        # state
        self.range = dict(mu1_min=-3,mu1_max=3,mu2_min=-3,mu2_max=3)
        self.phase_range = dict(x_min=-3,x_max=3,y_min=-3,y_max=3)
        self.current_mu = None
        self.traj_lines = []
        self.nullcline_artists = []
        self.nullcline_points = []
        self.field_artists = []
        self.sep_lines = []
        self.equilibria = []
        self.hopf_line = None
        self.show_field = False
        self.param_marker = None
        self.color_cycle = itertools.cycle(
            plt.rcParams["axes.prop_cycle"].by_key()["color"]
        )

        # canvases
        self.param_canvas = MplCanvas()
        self.phase_canvas = MplCanvas()
        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        self.splitter.addWidget(self.param_canvas)
        self.splitter.addWidget(self.phase_canvas)
        self.splitter.setSizes([1,1])

        # tabs
        tabs = QTabWidget()
        page1 = QWidget()
        lay1 = QVBoxLayout(page1)
        lay1.addWidget(self.splitter)
        tabs.addTab(page1,"Plots")
        self.eq_table = QTableWidget(0,5,self)
        self.eq_table.setHorizontalHeaderLabels(["x","y","type","λ₁","λ₂"])
        tabs.addTab(self.eq_table,"Equilibria")
        self.setCentralWidget(tabs)

        # toolbar
        self._build_toolbar()
        self._configure_param_axes()
        self._configure_phase_axes()
        self._connect_events()

    def _build_toolbar(self):
        bar = self.addToolBar("Controls"); bar.setIconSize(QSize(16,16))
        for txt,slot in [("Edit system",self._edit_system),
                         ("Set param range",self._edit_range),
                         ("Set phase range",self._edit_phase_range)]:
            act = QAction(txt,self); act.triggered.connect(slot); bar.addAction(act)

        bar.addSeparator()
        # μ spinboxes
        bar.addWidget(QLabel("μ₁:"))
        self.mu1_spin = QDoubleSpinBox(); self.mu1_spin.setRange(self.range["mu1_min"],self.range["mu1_max"])
        self.mu1_spin.setDecimals(3); self.mu1_spin.valueChanged.connect(self._on_spin_changed)
        bar.addWidget(self.mu1_spin)
        bar.addWidget(QLabel("μ₂:"))
        self.mu2_spin = QDoubleSpinBox(); self.mu2_spin.setRange(self.range["mu2_min"],self.range["mu2_max"])
        self.mu2_spin.setDecimals(3); self.mu2_spin.valueChanged.connect(self._on_spin_changed)
        bar.addWidget(self.mu2_spin)

        bar.addSeparator()
        self.integrator_cb = QComboBox(); self.integrator_cb.addItems(["RK45","RK23","DOP853","LSODA"])
        bar.addWidget(QLabel("Integrator:")); bar.addWidget(self.integrator_cb)

        btn_clear = QPushButton("Clear"); btn_clear.clicked.connect(self._clear_trajectories)
        bar.addWidget(btn_clear)

        bar.addSeparator()
        self.act_vector = QAction("Vector field",self); self.act_vector.setCheckable(True)
        self.act_vector.triggered.connect(self._toggle_vector_field); bar.addAction(self.act_vector)

        self.act_separatrices = QAction("Separatrices",self); self.act_separatrices.setCheckable(True)
        self.act_separatrices.triggered.connect(self._toggle_separatrices); bar.addAction(self.act_separatrices)

        bar.addSeparator()
        self.act_hopf = QAction("Hopf",self); self.act_hopf.setCheckable(True)
        self.act_hopf.triggered.connect(self._toggle_hopf); bar.addAction(self.act_hopf)

        bar.addSeparator()
        self.act_max_param = QAction("Max Param",self); self.act_max_param.setCheckable(True)
        self.act_max_param.triggered.connect(self._toggle_max_param); bar.addAction(self.act_max_param)
        self.act_max_phase = QAction("Max Phase",self); self.act_max_phase.setCheckable(True)
        self.act_max_phase.triggered.connect(self._toggle_max_phase); bar.addAction(self.act_max_phase)

    def _compile_system(self):
        x,y,m1,m2 = sp.symbols("x y mu1 mu2")
        f_sym = sp.sympify(self.f_expr); g_sym = sp.sympify(self.g_expr)
        self.f_lam = sp.lambdify((x,y,m1,m2),f_sym,"numpy")
        self.g_lam = sp.lambdify((x,y,m1,m2),g_sym,"numpy")
        # Jacobian, trace, det, derivative trace wrt mu1
        J = sp.Matrix([f_sym,g_sym]).jacobian([x,y])
        trJ = sp.simplify(J.trace()); detJ = sp.simplify(J.det())
        dtr_dmu1 = sp.diff(trJ,m1)
        self._trJ_sym = trJ
        self.detJ_lam = sp.lambdify((x,y,m1,m2),detJ,"numpy")
        self.dtr_dmu1_lam = sp.lambdify((x,y,m1,m2),dtr_dmu1,"numpy")
        self.J11 = sp.lambdify((x,y,m1,m2),J[0,0],"numpy")
        self.J12 = sp.lambdify((x,y,m1,m2),J[0,1],"numpy")
        self.J21 = sp.lambdify((x,y,m1,m2),J[1,0],"numpy")
        self.J22 = sp.lambdify((x,y,m1,m2),J[1,1],"numpy")
        self.rhs_func: Callable = lambda t,s,μ1,μ2: np.array([
            self.f_lam(s[0],s[1],μ1,μ2), self.g_lam(s[0],s[1],μ1,μ2)
        ])

    def _configure_param_axes(self):
        ax = self.param_canvas.ax; ax.clear()
        ax.set_title("Parameter plane (μ1, μ2)"); ax.set_xlabel("μ1"); ax.set_ylabel("μ2")
        ax.set_xlim(self.range["mu1_min"],self.range["mu1_max"])
        ax.set_ylim(self.range["mu2_min"],self.range["mu2_max"])
        self.param_canvas.canvas.draw_idle()

    def _configure_phase_axes(self):
        ax = self.phase_canvas.ax; ax.clear()
        ax.set_title("Phase plane (x, y)"); ax.set_xlabel("x"); ax.set_ylabel("y")
        ax.set_xlim(self.phase_range["x_min"],self.phase_range["x_max"])
        ax.set_ylim(self.phase_range["y_min"],self.phase_range["y_max"])
        self.phase_canvas.canvas.draw_idle()

    def _connect_events(self):
        self.param_canvas.canvas.mpl_connect("button_press_event",self._on_param_click)
        self.phase_canvas.canvas.mpl_connect("button_press_event",self._on_phase_click)

    def _on_spin_changed(self,_=None):
        μ1,μ2 = self.mu1_spin.value(), self.mu2_spin.value()
        self.current_mu=(μ1,μ2); self._clear_trajectories()
        if self.param_marker: self.param_marker.remove()
        self.param_marker,=self.param_canvas.ax.plot(μ1,μ2,"xr",ms=8)
        self.param_canvas.canvas.draw_idle()
        self._draw_nullclines(μ1,μ2)
        if self.show_field: self._draw_vector_field()
        self.statusBar().showMessage(f"μ=({μ1:.3g},{μ2:.3g}) (input)")

    def _on_param_click(self,event):
        if not(event.inaxes and event.dblclick): return
        self._clear_trajectories()
        μ1,μ2 = event.xdata,event.ydata; self.current_mu=(μ1,μ2)
        self.mu1_spin.blockSignals(True); self.mu1_spin.setValue(μ1); self.mu1_spin.blockSignals(False)
        self.mu2_spin.blockSignals(True); self.mu2_spin.setValue(μ2); self.mu2_spin.blockSignals(False)
        if self.param_marker: self.param_marker.remove()
        self.param_marker,=self.param_canvas.ax.plot(μ1,μ2,"xr",ms=8)
        self.param_canvas.canvas.draw_idle()
        self._draw_nullclines(μ1,μ2)
        if self.show_field: self._draw_vector_field()
        self.statusBar().showMessage(f"Selected μ=({μ1:.3g},{μ2:.3g})")

    def _draw_nullclines(self,μ1,μ2):
        # clear old
        for art in self.nullcline_artists:
            if hasattr(art,"collections"):
                for c in art.collections: c.remove()
            else:
                try: art.remove()
                except: pass
        self.nullcline_artists.clear()
        for pt in self.nullcline_points:
            try: pt.remove()
            except NotImplementedError: pt.set_visible(False)
        self.nullcline_points.clear()
        self.eq_table.setRowCount(0); self.equilibria.clear()

        ax=self.phase_canvas.ax
        xmin,xmax=ax.get_xlim(); ymin,ymax=ax.get_ylim()
        xx,yy=np.meshgrid(np.linspace(xmin,xmax,300),np.linspace(ymin,ymax,300))
        cf=ax.contour(xx,yy,self.f_lam(xx,yy,μ1,μ2),levels=[0],colors="blue",linestyles="--",linewidths=1.2)
        cg=ax.contour(xx,yy,self.g_lam(xx,yy,μ1,μ2),levels=[0],colors="green",linestyles="-",linewidths=1.2)
        self.nullcline_artists+=[cf,cg]

        # equilibria
        x,y,m1,m2=sp.symbols("x y mu1 mu2")
        f0=sp.sympify(self.f_expr).subs({m1:μ1,m2:μ2})
        g0=sp.sympify(self.g_expr).subs({m1:μ1,m2:μ2})
        sols=sp.solve([f0,g0],[x,y],dict=True)
        for sol in sols:
            xr,yr=sol[x],sol[y]
            if not(xr.is_real and yr.is_real): continue
            xf,yf=float(xr),float(yr)
            if not(xmin<=xf<=xmax and ymin<=yf<=ymax): continue
            a,b = self.J11(xf,yf,μ1,μ2), self.J12(xf,yf,μ1,μ2)
            c,d = self.J21(xf,yf,μ1,μ2), self.J22(xf,yf,μ1,μ2)
            Jmat=np.array([[a,b],[c,d]])
            ev,evec=np.linalg.eig(Jmat)
            re,im=np.real(ev),np.imag(ev)
            if abs(re[0])<1e-6 and abs(re[1])<1e-6 and np.any(im!=0):
                typ,color="center","green"
            elif np.any(im!=0):
                typ,color=("stable focus","purple") if np.all(re<0) else ("unstable focus","magenta")
            elif re[0]*re[1]<0:
                typ,color="saddle","red"
            else:
                typ,color=("stable node","blue") if np.all(re<0) else ("unstable node","cyan")
            pt,=ax.plot(xf,yf,"o",color=color,ms=8)
            txt=ax.text(xf,yf,typ,color=color,fontsize="small",va="bottom",ha="right")
            self.nullcline_points+=[pt,txt]
            self.equilibria.append({'x':xf,'y':yf,'type':typ,'eigvals':ev,'eigvecs':evec})
            row=self.eq_table.rowCount(); self.eq_table.insertRow(row)
            for col,val in enumerate([xf,yf,typ,f"{ev[0]:.3g}",f"{ev[1]:.3g}"]):
                self.eq_table.setItem(row,col,QTableWidgetItem(str(val)))

        handles=[
            Line2D([],[],marker="o",color="red",   linestyle="",label="saddle"),
            Line2D([],[],marker="o",color="blue",  linestyle="",label="stable node"),
            Line2D([],[],marker="o",color="cyan",  linestyle="",label="unstable node"),
            Line2D([],[],marker="o",color="purple",linestyle="",label="stable focus"),
            Line2D([],[],marker="o",color="magenta",linestyle="",label="unstable focus"),
            Line2D([],[],marker="o",color="green", linestyle="",label="center"),
        ]
        ax.legend(handles=handles,loc="upper right",fontsize="small")
        self.phase_canvas.canvas.draw_idle()

    def _toggle_vector_field(self,checked):
        self.show_field=checked
        if checked: self._draw_vector_field()
        else:
            for art in self.field_artists: art.remove()
            self.field_artists.clear(); self.phase_canvas.canvas.draw_idle()

    def _draw_vector_field(self):
        for art in self.field_artists: art.remove()
        self.field_artists.clear()
        if not self.current_mu: return
        μ1,μ2=self.current_mu; ax=self.phase_canvas.ax
        xmin,xmax=ax.get_xlim(); ymin,ymax=ax.get_ylim()
        XX,YY=np.meshgrid(np.linspace(xmin,xmax,20),np.linspace(ymin,ymax,20))
        U=self.f_lam(XX,YY,μ1,μ2); V=self.g_lam(XX,YY,μ1,μ2)
        M=np.hypot(U,V); M[M==0]=1; U2,V2=U/M,V/M
        Q=ax.quiver(XX,YY,U2,V2,angles="xy",pivot="mid",alpha=0.6)
        self.field_artists.append(Q); self.phase_canvas.canvas.draw_idle()

    def _toggle_separatrices(self,checked):
        if checked: self._draw_separatrices()
        else:
            for ln in self.sep_lines: ln.remove()
            self.sep_lines.clear(); self.phase_canvas.canvas.draw_idle()

    def _draw_separatrices(self):
        for ln in self.sep_lines: ln.remove()
        self.sep_lines.clear()
        if not self.current_mu: return
        ax=self.phase_canvas.ax
        for eq in self.equilibria:
            if eq['type']!="saddle": continue
            x0,y0=eq['x'],eq['y']; ev,vec=eq['eigvals'],eq['eigvecs']
            for i in (0,1):
                v=vec[:,i]; lam=ev[i]
                for sign in (1,-1):
                    start=np.array([x0,y0])+sign*1e-3*v
                    t_span=(0,10) if lam>0 else (0,-10)
                    sol=solve_ivp(lambda t,y: self.rhs_func(t,y,*self.current_mu),
                                  t_span,start,max_step=0.05)
                    ln,=ax.plot(sol.y[0],sol.y[1],'k--',lw=1)
                    self.sep_lines.append(ln)
        self.phase_canvas.canvas.draw_idle()

    def _toggle_hopf(self,checked):
        if checked: self._draw_hopf()
        else:
            if self.hopf_line: self.hopf_line.remove(); self.hopf_line=None
            self.param_canvas.canvas.draw_idle()

    def _draw_hopf(self):
        """Строит кривую Андронова–Гопфа: trJ=0 на равновесиях f=0,g=0."""
        # 1) Очистка предыдущей линии
        if self.hopf_line:
            self.hopf_line.remove()
            self.hopf_line = None

        # 2) Символьные переменные и выражения
        x, y, m1, m2 = sp.symbols("x y mu1 mu2")
        f_sym = sp.sympify(self.f_expr)
        g_sym = sp.sympify(self.g_expr)

        # Якобиан, trace и det
        J = sp.Matrix([f_sym, g_sym]).jacobian([x, y])
        trJ = sp.simplify(J.trace())
        detJ = sp.simplify(J.det())

        # 3) Найти символически равновесия (x,y) как функции параметров
        sols = sp.solve([f_sym, g_sym], [x, y], dict=True)
        if not sols:
            QMessageBox.warning(self, "Hopf", "Не удалось найти равновесия f=0,g=0")
            self.act_hopf.setChecked(False)
            return

        # Будем собирать подходящие точки (μ1,μ2)
        pts = []

        # Параметр, по которому будем сканировать (берём сетку на μ1×μ2)
        m1_min, m1_max = self.range["mu1_min"], self.range["mu1_max"]
        m2_min, m2_max = self.range["mu2_min"], self.range["mu2_max"]
        M1 = np.linspace(m1_min, m1_max, 300)
        M2 = np.linspace(m2_min, m2_max, 300)

        # 4) Проходим по сетке параметров и ищем те (μ1,μ2),
        #    для которых существует равновесие с trJ=0, detJ>0 и d(trJ)/dμ≠0
        trJ_lam = sp.lambdify((x, y, m1, m2), trJ, "numpy")
        detJ_lam = sp.lambdify((x, y, m1, m2), detJ, "numpy")
        # производная trace по μ1 (можно по μ2, зависит от случая)
        dtr_dmu1 = sp.diff(trJ, m1)
        dtr_dmu1_lam = sp.lambdify((x, y, m1, m2), dtr_dmu1, "numpy")

        for μ1 in M1:
            for μ2 in M2:
                # для каждой пары параметров проверяем все равновесия
                for sol in sols:
                    xr = sol[x].subs({m1: μ1, m2: μ2})
                    yr = sol[y].subs({m1: μ1, m2: μ2})
                    # интересуют только реальные решения
                    if not (xr.is_real and yr.is_real):
                        continue
                    x0, y0 = float(xr), float(yr)

                    # 4.a) trJ=0?
                    if abs(trJ_lam(x0, y0, μ1, μ2)) > 1e-2:
                        continue
                    # 4.b) detJ>0?
                    if detJ_lam(x0, y0, μ1, μ2) <= 0:
                        continue
                    # 4.c) транверсальность d(trJ)/dμ1 != 0?
                    if abs(dtr_dmu1_lam(x0, y0, μ1, μ2)) < 1e-3:
                        continue

                    pts.append((μ1, μ2))
                    break  # для данной пары (μ1,μ2) достаточно одного равновесия

        if not pts:
            QMessageBox.warning(self, "Hopf", "Нет точек, удовлетворяющих условиям Hopf")
            self.act_hopf.setChecked(False)
            return

        # 5) Разбиваем список точек в массивы для рисования
        M1_good, M2_good = zip(*pts)

        # 6) Рисуем на плоскости параметров
        ax = self.param_canvas.ax
        self.hopf_line, = ax.plot(M1_good, M2_good, color="magenta", lw=2, label="Hopf")
        ax.legend(loc="best", fontsize="small")
        self.param_canvas.canvas.draw_idle()

        # 7) Показываем общее уравнение (только trJ), чтобы пользователь понимал
        QMessageBox.information(
            self, "Hopf bifurcation",
            "Условие Hopf определяется из trJ=0 на равновесиях.\n"
            f"trJ = {sp.pretty(trJ)}\n"
            "и накладывает detJ>0, d(trJ)/dμ1≠0"
        )

    def _on_phase_click(self,event):
        if not(event.inaxes and event.dblclick) or not self.current_mu: return
        x0,y0=event.xdata,event.ydata; μ1,μ2=self.current_mu; method=self.integrator_cb.currentText()
        T,N=20,400
        sol_f=solve_ivp(lambda t,y: self.rhs_func(t,y,μ1,μ2),(0,T),[x0,y0],
                        t_eval=np.linspace(0,T,N//2),method=method)
        sol_b=solve_ivp(lambda t,y: self.rhs_func(t,y,μ1,μ2),(0,-T),[x0,y0],
                        t_eval=np.linspace(0,-T,N//2),method=method)
        xb,yb=sol_b.y[0][::-1][:-1],sol_b.y[1][::-1][:-1]
        xf,yf=sol_f.y; xs=np.concatenate([xb,xf]); ys=np.concatenate([yb,yf])
        line,=self.phase_canvas.ax.plot(xs,ys,color=next(self.color_cycle))
        self.traj_lines.append(line); self.phase_canvas.canvas.draw_idle()

    def _clear_trajectories(self):
        for ln in self.traj_lines: ln.remove()
        self.traj_lines.clear(); self.phase_canvas.canvas.draw_idle()

    def _edit_system(self):
        dlg=SystemDialog(self.f_expr,self.g_expr,self)
        if dlg.exec():
            self.f_expr,self.g_expr=dlg.texts()
            try:
                self._compile_system()
                if self.current_mu:
                    self._draw_nullclines(*self.current_mu)
                    if self.show_field: self._draw_vector_field()
                    if self.act_separatrices.isChecked(): self._draw_separatrices()
                    if self.act_hopf.isChecked(): self._draw_hopf()
                self.statusBar().showMessage("System updated")
            except Exception as e:
                QMessageBox.critical(self,"Error",str(e))

    def _edit_range(self):
        dlg=RangeDialog(self.range,self)
        if dlg.exec():
            self.range=dlg.values(); self._configure_param_axes(); self.statusBar().showMessage("Parameter range updated")

    def _edit_phase_range(self):
        dlg=PhaseRangeDialog(self.phase_range,self)
        if dlg.exec():
            self.phase_range=dlg.values(); self._configure_phase_axes(); self.statusBar().showMessage("Phase range updated")

    def _toggle_max_param(self,checked):
        if checked:
            self.phase_canvas.hide(); self.splitter.setStretchFactor(0,1); self.splitter.setStretchFactor(1,0)
            self.act_max_phase.setChecked(False)
        else:
            self.phase_canvas.show(); self.splitter.setStretchFactor(0,1); self.splitter.setStretchFactor(1,1)

    def _toggle_max_phase(self,checked):
        if checked:
            self.param_canvas.hide(); self.splitter.setStretchFactor(0,0); self.splitter.setStretchFactor(1,1)
            self.act_max_param.setChecked(False)
        else:
            self.param_canvas.show(); self.splitter.setStretchFactor(0,1); self.splitter.setStretchFactor(1,1)


def main() -> None:
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
