# -*- coding: utf-8 -*-
"""Dynamic System Sandbox — Hopf bifurcations (robust version)."""

from __future__ import annotations
import sys, itertools
from collections.abc import Callable

import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp
from scipy.optimize import root
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

# ---------------------------------------------------------------------------
# 1. safe_exp — экспонента с клипованным аргументом
EXP_CLIP = 50.0                      # exp(±50) ~ 5e21
def safe_exp(z):
    return np.exp(np.clip(z, -EXP_CLIP, EXP_CLIP))

# 2. модуль для lambdify: exp → safe_exp, всё остальное — NumPy
SAFE_MODULE = {'exp': safe_exp, 'safe_exp': safe_exp, 'np': np}

# --------------------------------------------------------------------------- диалоги

class SystemDialog(QDialog):
    def __init__(self, f_txt, g_txt, parent=None):
        super().__init__(parent); self.setWindowTitle("Edit system")
        self.f_edit, self.g_edit = QPlainTextEdit(f_txt), QPlainTextEdit(g_txt)
        form = QFormLayout(self)
        form.addRow("f(x, y; μ1, μ2) =", self.f_edit)
        form.addRow("g(x, y; μ1, μ2) =", self.g_edit)
        box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok |
                               QDialogButtonBox.StandardButton.Cancel)
        box.accepted.connect(self.accept); box.rejected.connect(self.reject)
        form.addRow(box)
    def texts(self): return self.f_edit.toPlainText(), self.g_edit.toPlainText()


class RangeDialog(QDialog):
    def __init__(self, rng, parent=None):
        super().__init__(parent); self.setWindowTitle("Parameter range")
        self.edits={}
        form=QFormLayout(self)
        for k in ("mu1_min","mu1_max","mu2_min","mu2_max"):
            spb=QDoubleSpinBox(minimum=-1e6, maximum=1e6, decimals=4)
            spb.setValue(rng[k]); form.addRow(k, spb); self.edits[k]=spb
        box=QDialogButtonBox(QDialogButtonBox.StandardButton.Ok |
                             QDialogButtonBox.StandardButton.Cancel)
        box.accepted.connect(self.accept); box.rejected.connect(self.reject)
        form.addRow(box)
    def values(self): return {k:w.value() for k,w in self.edits.items()}


class PhaseRangeDialog(QDialog):
    def __init__(self, pr, parent=None):
        super().__init__(parent); self.setWindowTitle("Phase range")
        self.edits={}
        form=QFormLayout(self)
        for k,l in (("x_min","x min"),("x_max","x max"),
                    ("y_min","y min"),("y_max","y max")):
            spb=QDoubleSpinBox(minimum=-1e6, maximum=1e6, decimals=4)
            spb.setValue(pr[k]); form.addRow(l, spb); self.edits[k]=spb
        box=QDialogButtonBox(QDialogButtonBox.StandardButton.Ok |
                             QDialogButtonBox.StandardButton.Cancel)
        box.accepted.connect(self.accept); box.rejected.connect(self.reject)
        form.addRow(box)
    def values(self): return {k:w.value() for k,w in self.edits.items()}

# --------------------------------------------------------------------------- холст

class MplCanvas(QWidget):
    def __init__(self):
        super().__init__()
        self.fig, self.ax = plt.subplots()
        self.canvas = FigureCanvasQTAgg(self.fig)
        lay = QVBoxLayout(self); lay.setContentsMargins(0,0,0,0)
        lay.addWidget(self.canvas); lay.addWidget(NavigationToolbar2QT(self.canvas,self))
        self.canvas.mpl_connect("scroll_event", self._on_scroll)
    def _on_scroll(self, e):
        if e.inaxes is None: return
        scale=1.2 if e.button=="up" else 0.8
        ax=e.inaxes; xm,ym=e.xdata,e.ydata
        xmin,xmax=ax.get_xlim(); ymin,ymax=ax.get_ylim()
        ax.set_xlim(xm+(xmin-xm)*scale, xm+(xmax-xm)*scale)
        ax.set_ylim(ym+(ymin-ym)*scale, ym+(ymax-ym)*scale)
        self.canvas.draw_idle()

# --------------------------------------------------------------------------- главное окно

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__(); self.setWindowTitle("Dynamic System Sandbox — Hopf")
        self.resize(1100,650)

        # ----------- система по умолчанию -----------------------------------
        self.f_expr = "-2*exp(-x) + exp(-2*x) + y"
        self.g_expr = "-x + mu1*y + mu2"
        self._compile_system()

        # ----------- состояние ----------------------------------------------
        self.range=dict(mu1_min=-3,mu1_max=3,mu2_min=-3,mu2_max=3)
        self.phase_range=dict(x_min=-3,x_max=3,y_min=-3,y_max=3)
        self.current_mu=None
        self.traj_lines=[]; self.nullcline_artists=[]; self.nullcline_points=[]
        self.field_artists=[]; self.sep_lines=[]; self.equilibria=[]
        self.hopf_line=None; self.show_field=False; self.param_marker=None
        self.color_cycle=itertools.cycle(plt.rcParams["axes.prop_cycle"].by_key()["color"])

        # ----------- интерфейс ----------------------------------------------
        self.param_canvas,self.phase_canvas=MplCanvas(),MplCanvas()
        self.splitter=QSplitter(Qt.Orientation.Horizontal)
        self.splitter.addWidget(self.param_canvas); self.splitter.addWidget(self.phase_canvas)

        tabs=QTabWidget(); pg=QWidget(); QVBoxLayout(pg).addWidget(self.splitter)
        tabs.addTab(pg,"Plots")
        self.eq_table=QTableWidget(0,5); self.eq_table.setHorizontalHeaderLabels(["x","y","type","λ₁","λ₂"])
        tabs.addTab(self.eq_table,"Equilibria"); self.setCentralWidget(tabs)

        self._build_toolbar(); self._configure_param_axes(); self._configure_phase_axes(); self._connect_events()

    # ------------------ компиляция системы (все lambdify через SAFE_MODULE) -
    def _compile_system(self):
        x,y,m1,m2 = sp.symbols("x y mu1 mu2")
        f_sym,g_sym = sp.sympify(self.f_expr), sp.sympify(self.g_expr)
        self.f_lam = sp.lambdify((x,y,m1,m2), f_sym, modules=[SAFE_MODULE,'numpy'])
        self.g_lam = sp.lambdify((x,y,m1,m2), g_sym, modules=[SAFE_MODULE,'numpy'])
        J = sp.Matrix([f_sym,g_sym]).jacobian([x,y])
        trJ, detJ = sp.simplify(J.trace()), sp.simplify(J.det())
        dtr_dmu1 = sp.diff(trJ, m1)

        self._trJ_sym = trJ
        self.detJ_lam  = sp.lambdify((x,y,m1,m2), detJ, modules=[SAFE_MODULE,'numpy'])
        self.dtr_dmu1_lam = sp.lambdify((x,y,m1,m2), dtr_dmu1, modules=[SAFE_MODULE,'numpy'])
        self.J11 = sp.lambdify((x,y,m1,m2), J[0,0], modules=[SAFE_MODULE,'numpy'])
        self.J12 = sp.lambdify((x,y,m1,m2), J[0,1], modules=[SAFE_MODULE,'numpy'])
        self.J21 = sp.lambdify((x,y,m1,m2), J[1,0], modules=[SAFE_MODULE,'numpy'])
        self.J22 = sp.lambdify((x,y,m1,m2), J[1,1], modules=[SAFE_MODULE,'numpy'])
        self.rhs_func: Callable = lambda t,s,μ1,μ2: np.array([
            self.f_lam(s[0],s[1],μ1,μ2),
            self.g_lam(s[0],s[1],μ1,μ2)])

    # ------------------ тулбар ----------------------------------------------
    def _build_toolbar(self):
        bar=self.addToolBar("Controls"); bar.setIconSize(QSize(16,16))
        for txt,slot in [("Edit system",self._edit_system),
                         ("Set param range",self._edit_range),
                         ("Set phase range",self._edit_phase_range)]:
            act=QAction(txt,self); act.triggered.connect(slot); bar.addAction(act)
        bar.addSeparator(); bar.addWidget(QLabel("μ₁:"))
        self.mu1_spin=QDoubleSpinBox(); self.mu1_spin.setDecimals(3)
        self.mu1_spin.valueChanged.connect(self._on_spin_changed); bar.addWidget(self.mu1_spin)
        bar.addWidget(QLabel("μ₂:"))
        self.mu2_spin=QDoubleSpinBox(); self.mu2_spin.setDecimals(3)
        self.mu2_spin.valueChanged.connect(self._on_spin_changed); bar.addWidget(self.mu2_spin)
        bar.addSeparator(); self.integrator_cb=QComboBox()
        self.integrator_cb.addItems(["RK45","RK23","DOP853","LSODA","Radau","BDF"])
        bar.addWidget(QLabel("Integrator:")); bar.addWidget(self.integrator_cb)
        btn_clear=QPushButton("Clear"); btn_clear.clicked.connect(self._clear_trajectories); bar.addWidget(btn_clear)
        bar.addSeparator(); self.act_vector=QAction("Vector field",self,checkable=True)
        self.act_vector.triggered.connect(self._toggle_vector_field); bar.addAction(self.act_vector)
        self.act_separatrices=QAction("Separatrices",self,checkable=True)
        self.act_separatrices.triggered.connect(self._toggle_separatrices); bar.addAction(self.act_separatrices)
        bar.addSeparator(); self.act_hopf=QAction("Hopf",self,checkable=True)
        self.act_hopf.triggered.connect(self._toggle_hopf); bar.addAction(self.act_hopf)
        bar.addSeparator(); self.act_max_param=QAction("Max Param",self,checkable=True)
        self.act_max_param.triggered.connect(self._toggle_max_param); bar.addAction(self.act_max_param)
        self.act_max_phase=QAction("Max Phase",self,checkable=True)
        self.act_max_phase.triggered.connect(self._toggle_max_phase); bar.addAction(self.act_max_phase)

    # ---------------- оси, события, спинбоксы --------------------------------
    def _configure_param_axes(self):
        ax=self.param_canvas.ax; ax.clear(); ax.set_title("Parameter plane (μ1, μ2)")
        ax.set_xlabel("μ1"); ax.set_ylabel("μ2")
        ax.set_xlim(self.range["mu1_min"],self.range["mu1_max"])
        ax.set_ylim(self.range["mu2_min"],self.range["mu2_max"])
        self.param_canvas.canvas.draw_idle()
    def _configure_phase_axes(self):
        ax=self.phase_canvas.ax; ax.clear(); ax.set_title("Phase plane (x, y)")
        ax.set_xlabel("x"); ax.set_ylabel("y")
        ax.set_xlim(self.phase_range["x_min"],self.phase_range["x_max"])
        ax.set_ylim(self.phase_range["y_min"],self.phase_range["y_max"])
        self.phase_canvas.canvas.draw_idle()
    def _connect_events(self):
        self.param_canvas.canvas.mpl_connect("button_press_event", self._on_param_click)
        self.phase_canvas.canvas.mpl_connect("button_press_event", self._on_phase_click)

    # ---------------------- спинбоксы / выбор μ ------------------------------
    def _on_spin_changed(self,_=None):
        μ1,μ2=self.mu1_spin.value(),self.mu2_spin.value()
        self.current_mu=(μ1,μ2); self._clear_trajectories()
        self._toggle_separatrices(False)
        if self.param_marker: self.param_marker.remove()
        self.param_marker,=self.param_canvas.ax.plot(μ1,μ2,"xr",ms=8)
        self.param_canvas.canvas.draw_idle()
        self._draw_nullclines(μ1,μ2)
        if self.show_field: self._draw_vector_field()
        self.statusBar().showMessage(f"μ=({μ1:.3g}, {μ2:.3g})")

    def _on_param_click(self,e):
        if not(e.inaxes and e.dblclick): return
        μ1,μ2=e.xdata,e.ydata; self.current_mu=(μ1,μ2)
        self.mu1_spin.blockSignals(True); self.mu1_spin.setValue(μ1); self.mu1_spin.blockSignals(False)
        self.mu2_spin.blockSignals(True); self.mu2_spin.setValue(μ2); self.mu2_spin.blockSignals(False)
        self._clear_trajectories()
        self._toggle_separatrices(False)
        if self.param_marker: self.param_marker.remove()
        self.param_marker,=self.param_canvas.ax.plot(μ1,μ2,"xr",ms=8)
        self.param_canvas.canvas.draw_idle()
        self._draw_nullclines(μ1,μ2);  self._draw_vector_field() if self.show_field else None
        self.statusBar().showMessage(f"Selected μ=({μ1:.3g}, {μ2:.3g})")

    # ----------------------- поиск равновесий (быстрый) ----------------------
    def _find_equilibria_numeric(self, μ1, μ2):
        guesses = [(x0, y0)
                   for x0 in np.linspace(self.phase_range["x_min"],
                                         self.phase_range["x_max"], 5)
                   for y0 in np.linspace(self.phase_range["y_min"],
                                         self.phase_range["y_max"], 5)]
        sols = []
        tol = 1e-6  # и по функции, и по координате
        fvec = lambda v: [self.f_lam(v[0], v[1], μ1, μ2),
                          self.g_lam(v[0], v[1], μ1, μ2)]

        for x0, y0 in guesses:
            sol = root(lambda v: [np.tanh(fvec(v)[0]), np.tanh(fvec(v)[1])],
                       [x0, y0], method="hybr",
                       options={"maxfev": 200, "xtol": tol})
            if sol.success:
                x_eq, y_eq = sol.x
                # 1) Проверяем функцию в точке — это действительно пересечение?
                if (abs(fvec([x_eq, y_eq])[0]) > tol or
                        abs(fvec([x_eq, y_eq])[1]) > tol):
                    continue
                # 2) Сравниваем с уже найденными по расстоянию
                if not any(np.hypot(x_eq - xs, y_eq - ys) < 1e-3 for xs, ys in sols):
                    sols.append((x_eq, y_eq))
        return [(round(x, 6), round(y, 6)) for x, y in sols]

    # ------------------- nullclines + классификация --------------------------
    def _draw_nullclines(self, μ1, μ2):
        for art in self.nullcline_artists:
            if hasattr(art,"collections"):
                for c in art.collections: c.remove()
            else: art.remove()
        self.nullcline_artists.clear()
        for pt in self.nullcline_points: pt.remove()
        self.nullcline_points.clear()
        self.eq_table.setRowCount(0); self.equilibria.clear()

        ax=self.phase_canvas.ax; xmin,xmax=ax.get_xlim(); ymin,ymax=ax.get_ylim()
        xx,yy=np.meshgrid(np.linspace(xmin,xmax,300),
                          np.linspace(ymin,ymax,300))
        with np.errstate(over='ignore',invalid='ignore'):
            F=self.f_lam(xx,yy,μ1,μ2); G=self.g_lam(xx,yy,μ1,μ2)
        F=np.nan_to_num(F,nan=0.0,posinf=np.nan,neginf=np.nan)
        G=np.nan_to_num(G,nan=0.0,posinf=np.nan,neginf=np.nan)
        cf=ax.contour(xx,yy,F,levels=[0],colors="blue",linestyles="--",linewidths=1.2)
        cg=ax.contour(xx,yy,G,levels=[0],colors="green",linestyles="-", linewidths=1.2)
        self.nullcline_artists+= [cf,cg]

        # равновесия
        for xf,yf in self._find_equilibria_numeric(μ1,μ2):
            if not(xmin<=xf<=xmax and ymin<=yf<=ymax): continue
            a,b=self.J11(xf,yf,μ1,μ2),self.J12(xf,yf,μ1,μ2)
            c,d=self.J21(xf,yf,μ1,μ2),self.J22(xf,yf,μ1,μ2)
            Jmat=np.array([[a,b],[c,d]])
            ev,evec=np.linalg.eig(Jmat)
            re,im=np.real(ev),np.imag(ev)
            if abs(re[0])<1e-6 and abs(re[1])<1e-6 and np.any(im!=0):
                typ,color="center","green"
            elif np.any(im!=0):
                typ,color=(("stable focus","purple") if np.all(re<0)
                           else ("unstable focus","magenta"))
            elif re[0]*re[1]<0: typ,color="saddle","red"
            else: typ,color=(("stable node","blue") if np.all(re<0)
                             else ("unstable node","cyan"))
            pt,=ax.plot(xf,yf,"o",color=color,ms=8)
            txt=ax.text(xf,yf,typ,color=color,fontsize="small",va="bottom",ha="right")
            self.nullcline_points+= [pt,txt]
            self.equilibria.append({'x':xf,'y':yf,'type':typ,
                                    'eigvals':ev,'eigvecs':evec})
            row=self.eq_table.rowCount(); self.eq_table.insertRow(row)
            for col,val in enumerate([xf,yf,typ,f"{ev[0]:.3g}",f"{ev[1]:.3g}"]):
                self.eq_table.setItem(row,col,QTableWidgetItem(str(val)))

        handles=[Line2D([],[],marker="o",color="red",linestyle="",label="saddle"),
                 Line2D([],[],marker="o",color="blue",linestyle="",label="stable node"),
                 Line2D([],[],marker="o",color="cyan",linestyle="",label="unstable node"),
                 Line2D([],[],marker="o",color="purple",linestyle="",label="stable focus"),
                 Line2D([],[],marker="o",color="magenta",linestyle="",label="unstable focus"),
                 Line2D([],[],marker="o",color="green",linestyle="",label="center")]
        ax.legend(handles=handles,loc="upper right",fontsize="small")
        self.phase_canvas.canvas.draw_idle()

    # ------------------ векторное поле --------------------------------------
    def _toggle_vector_field(self,chk):
        self.show_field=chk
        if chk: self._draw_vector_field()
        else:
            for art in self.field_artists: art.remove()
            self.field_artists.clear(); self.phase_canvas.canvas.draw_idle()

    def _draw_vector_field(self):
        for art in self.field_artists: art.remove()
        self.field_artists.clear()
        if not self.current_mu: return
        μ1,μ2=self.current_mu; ax=self.phase_canvas.ax
        xmin,xmax=ax.get_xlim(); ymin,ymax=ax.get_ylim()
        XX,YY=np.meshgrid(np.linspace(xmin,xmax,20),
                          np.linspace(ymin,ymax,20))
        U=self.f_lam(XX,YY,μ1,μ2); V=self.g_lam(XX,YY,μ1,μ2)
        M=np.hypot(U,V); M[M==0]=1
        Q=ax.quiver(XX,YY,U/M,V/M,angles="xy",pivot="mid",alpha=0.6)
        self.field_artists.append(Q); self.phase_canvas.canvas.draw_idle()

    # ------------------ сепаратрисы седла -----------------------------------
    def _toggle_separatrices(self,chk):
        if chk: self._draw_separatrices()
        else:
            for ln in self.sep_lines: ln.remove()
            self.sep_lines.clear(); self.phase_canvas.canvas.draw_idle()

    def _draw_separatrices(self):
        for ln in self.sep_lines: ln.remove()
        self.sep_lines.clear()
        if not self.current_mu: return
        ax=self.phase_canvas.ax; μ1,μ2=self.current_mu
        for eq in self.equilibria:
            if eq['type']!="saddle": continue
            x0,y0=eq['x'],eq['y']; ev,vec=eq['eigvals'],eq['eigvecs']
            for i in (0,1):
                lam=np.real(ev[i])
                if abs(lam)<1e-4: continue
                v=np.real_if_close(vec[:,i]); v/=np.linalg.norm(v)
                for sgn in (+1,-1):
                    start=np.array([x0,y0])+sgn*5e-3*v
                    t_span=(0,6) if lam>0 else (0,-6)
                    sol=solve_ivp(lambda t,y:self.rhs_func(t,y,μ1,μ2),
                                  t_span,start,max_step=0.2,
                                  rtol=1e-4,atol=1e-7)
                    ln,=ax.plot(sol.y[0],sol.y[1],'k--',lw=1)
                    self.sep_lines.append(ln)
        self.phase_canvas.canvas.draw_idle()

    # ------------------ Hopf curve (как раньше) -----------------------------
    def _toggle_hopf(self,chk):
        if chk: self._draw_hopf()
        else:
            if self.hopf_line: self.hopf_line.remove(); self.hopf_line=None
            self.param_canvas.canvas.draw_idle()

    def _draw_hopf(self):
        x,y,μ1,μ2=sp.symbols("x y mu1 mu2")
        f_sym,g_sym=sp.sympify(self.f_expr),sp.sympify(self.g_expr)
        eqs=sp.solve([f_sym,g_sym],[x,y],dict=True)
        if not eqs:
            QMessageBox.warning(self,"Hopf","Не удалось найти равновесия")
            self.act_hopf.setChecked(False); return
        J=sp.Matrix([f_sym,g_sym]).jacobian([x,y])
        trJ_sym=sp.simplify(J.trace())
        detJ_lam=sp.lambdify((x,y,μ1,μ2),sp.simplify(J.det()),modules=[SAFE_MODULE,'numpy'])
        dtr_dmu1_lam=sp.lambdify((x,y,μ1,μ2),sp.diff(trJ_sym,μ1),modules=[SAFE_MODULE,'numpy'])
        dtr_dmu2_lam=sp.lambdify((x,y,μ1,μ2),sp.diff(trJ_sym,μ2),modules=[SAFE_MODULE,'numpy'])
        curves=[]
        for sol in eqs:
            xi,yi = sol[x],sol[y]
            trJ_i=sp.simplify(trJ_sym.subs({x:xi,y:yi}))
            sol_mu2=sp.solve(trJ_i,μ2)
            if sol_mu2:
                phi=sol_mu2[0]; var_in,var_out=μ1,μ2
                inv_fun=sp.lambdify(μ1,phi,modules=[SAFE_MODULE,'numpy'])
                dtrans=dtr_dmu2_lam
            else:
                sol_mu1=sp.solve(trJ_i,μ1)
                if not sol_mu1: continue
                psi=sol_mu1[0]; var_in,var_out=μ2,μ1
                inv_fun=sp.lambdify(μ2,psi,modules=[SAFE_MODULE,'numpy'])
                dtrans=dtr_dmu1_lam
            rng_in=(self.range["mu1_min"],self.range["mu1_max"]) if var_in is μ1 \
                   else (self.range["mu2_min"],self.range["mu2_max"])
            vals=np.linspace(*rng_in,400)
            try: out=inv_fun(vals)
            except: continue
            mask=np.isfinite(out)
            if var_out is μ1:
                mask &= (out>=self.range["mu1_min"])&(out<=self.range["mu1_max"])
            else:
                mask &= (out>=self.range["mu2_min"])&(out<=self.range["mu2_max"])
            if not np.any(mask): continue
            a_in,a_out=vals[mask],out[mask]
            x_fun=sp.lambdify((μ1,μ2),xi,modules=[SAFE_MODULE,'numpy'])
            y_fun=sp.lambdify((μ1,μ2),yi,modules=[SAFE_MODULE,'numpy'])
            pts=[]
            for u,v in zip(a_in,a_out):
                mu1v,mu2v=(u,v) if var_in is μ1 else (v,u)
                x0=float(x_fun(mu1v,mu2v)); y0=float(y_fun(mu1v,mu2v))
                if detJ_lam(x0,y0,mu1v,mu2v)<=0: continue
                if abs(dtrans(x0,y0,mu1v,mu2v))<1e-3: continue
                pts.append((mu1v,mu2v))
            if pts: curves.append(tuple(zip(*pts)))
        if not curves:
            QMessageBox.warning(self,"Hopf","Нет участков кривой Hopf")
            self.act_hopf.setChecked(False); return
        ax=self.param_canvas.ax
        for xs,ys in curves:
            self.hopf_line,=ax.plot(xs,ys,color="magenta",lw=2,label="Hopf")
        ax.legend(loc="best",fontsize="small")
        self.param_canvas.canvas.draw_idle()
        QMessageBox.information(self,"Hopf",
            "Условие Hopf: trJ=0, detJ>0, d(trJ)/dμ ≠ 0")

    # ------------------ фазовые траектории ----------------------------------
    def _on_phase_click(self,e):
        if not(e.inaxes and e.dblclick) or not self.current_mu: return
        x0,y0=e.xdata,e.ydata; μ1,μ2=self.current_mu
        method=self.integrator_cb.currentText()

        # безопасный RHS
        def rhs_sat(t,s):
            dx,dy=self.rhs_func(t,s,μ1,μ2)
            if not np.isfinite(dx) or not np.isfinite(dy): return np.array([0.0,0.0])
            V_max=1e3; v=np.hypot(dx,dy)
            if v>V_max: dx,dy=dx*V_max/v,dy*V_max/v
            return np.array([dx,dy])
        # событие-стоп
        cx=(self.phase_range["x_min"]+self.phase_range["x_max"])/2
        cy=(self.phase_range["y_min"]+self.phase_range["y_max"])/2
        rx=(self.phase_range["x_max"]-self.phase_range["x_min"])/2
        ry=(self.phase_range["y_max"]-self.phase_range["y_min"])/2
        def stop_out(t,y): return max(abs(y[0]-cx)-rx, abs(y[1]-cy)-ry)
        stop_out.terminal=True

        T,N=15,300
        sol_f=solve_ivp(rhs_sat,(0,T),[x0,y0],t_eval=np.linspace(0,T,N//2),
                        method=method,max_step=0.2,rtol=1e-4,atol=1e-7,events=stop_out)
        sol_b=solve_ivp(rhs_sat,(0,-T),[x0,y0],t_eval=np.linspace(0,-T,N//2),
                        method=method,max_step=0.2,rtol=1e-4,atol=1e-7,events=stop_out)
        xb,yb=sol_b.y[0][::-1][:-1],sol_b.y[1][::-1][:-1]
        xf,yf=sol_f.y
        xs=np.concatenate([xb,xf]); ys=np.concatenate([yb,yf])
        line,=self.phase_canvas.ax.plot(xs,ys,color=next(self.color_cycle))
        self.traj_lines.append(line); self.phase_canvas.canvas.draw_idle()

    # ------------------ прочие мелочи ---------------------------------------
    def _clear_trajectories(self):
        for ln in self.traj_lines: ln.remove()
        self.traj_lines.clear(); self.phase_canvas.canvas.draw_idle()

    def _edit_system(self):
        dlg=SystemDialog(self.f_expr,self.g_expr,self)
        if dlg.exec():
            self.f_expr,self.g_expr=dlg.texts()
            try:
                self._compile_system()
                if self.current_mu: self._draw_nullclines(*self.current_mu)
                if self.show_field: self._draw_vector_field()
                if self.act_separatrices.isChecked(): self._draw_separatrices()
                if self.act_hopf.isChecked(): self._draw_hopf()
                self.statusBar().showMessage("System updated")
            except Exception as e:
                QMessageBox.critical(self,"Error",str(e))

    def _edit_range(self):
        dlg=RangeDialog(self.range,self)
        if dlg.exec():
            self.range=dlg.values()
            self.mu1_spin.setRange(self.range["mu1_min"],self.range["mu1_max"])
            self.mu2_spin.setRange(self.range["mu2_min"],self.range["mu2_max"])
            self._configure_param_axes()
            self.statusBar().showMessage("Parameter range updated")

    def _edit_phase_range(self):
        dlg=PhaseRangeDialog(self.phase_range,self)
        if dlg.exec():
            self.phase_range=dlg.values(); self._configure_phase_axes()
            self.statusBar().showMessage("Phase range updated")

    def _toggle_max_param(self,chk):
        if chk:
            self.phase_canvas.hide(); self.splitter.setStretchFactor(0,1); self.splitter.setStretchFactor(1,0)
            self.act_max_phase.setChecked(False)
        else:
            self.phase_canvas.show(); self.splitter.setStretchFactor(0,1); self.splitter.setStretchFactor(1,1)

    def _toggle_max_phase(self,chk):
        if chk:
            self.param_canvas.hide(); self.splitter.setStretchFactor(0,0); self.splitter.setStretchFactor(1,1)
            self.act_max_param.setChecked(False)
        else:
            self.param_canvas.show(); self.splitter.setStretchFactor(0,1); self.splitter.setStretchFactor(1,1)

# --------------------------------------------------------------------------- main

def main():
    app=QApplication(sys.argv); win=MainWindow(); win.show(); sys.exit(app.exec())

if __name__=="__main__": main()
