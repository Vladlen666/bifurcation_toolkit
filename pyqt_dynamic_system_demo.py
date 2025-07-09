# -*- coding: utf-8 -*-
"""
Dynamic-System Sandbox — Bogdanov–Takens only (robust, 2025-07-09)

Полная версия с:
 • BT-таблицей
 • обнаружением предельного цикла
 • настраиваемым временем траекторий
 • полноэкранным фазовым портретом (⇄ Esc / F11 или кнопка)
 • гибким сохранением рисунков (задаёте размер, DPI; опция «сохранить x(t)»)
 • вкладкой Combined (phase + x(t) в одном изображении)
"""

from __future__ import annotations
import sys, itertools, os
from collections.abc import Callable

import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp
from scipy.optimize import root
import matplotlib
matplotlib.use("QtAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import (
    FigureCanvasQTAgg, NavigationToolbar2QT
)
from matplotlib.lines import Line2D

from PyQt6.QtCore   import Qt, QSize
from PyQt6.QtGui    import QAction
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QSplitter,
    QPushButton, QFormLayout, QDialog, QDialogButtonBox,
    QComboBox, QDoubleSpinBox, QLabel, QPlainTextEdit, QMessageBox,
    QTabWidget, QTableWidget, QTableWidgetItem, QProgressBar,
    QCheckBox, QFileDialog
)

# --------------------------------------------------------------------------- #
# 1. «Безопасная» экспонента                                                  #
# --------------------------------------------------------------------------- #
EXP_CLIP = 50.0                       # exp(±50) ≈ 5·10²¹
def safe_exp(z):                      # защита от переполнения
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
        box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        box.accepted.connect(self.accept); box.rejected.connect(self.reject)
        form.addRow(box)

    def texts(self):  # -> tuple[str,str]
        return self.f_edit.toPlainText(), self.g_edit.toPlainText()


class RangeDialog(QDialog):
    def __init__(self, rng: dict[str, float], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Parameter range")
        self.edits: dict[str, QDoubleSpinBox] = {}
        form = QFormLayout(self)
        for k in ("mu1_min", "mu1_max", "mu2_min", "mu2_max"):
            spb = QDoubleSpinBox(minimum=-1e6, maximum=1e6, decimals=4)
            spb.setValue(rng[k]); form.addRow(k, spb); self.edits[k] = spb
        box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        box.accepted.connect(self.accept); box.rejected.connect(self.reject)
        form.addRow(box)

    def values(self):
        return {k: w.value() for k, w in self.edits.items()}


class PhaseRangeDialog(QDialog):
    def __init__(self, pr: dict[str, float], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Phase range")
        self.edits: dict[str, QDoubleSpinBox] = {}
        form = QFormLayout(self)
        for k, l in (("x_min","x min"),("x_max","x max"),
                     ("y_min","y min"),("y_max","y max")):
            spb = QDoubleSpinBox(minimum=-1e6, maximum=1e6, decimals=4)
            spb.setValue(pr[k]); form.addRow(l, spb); self.edits[k] = spb
        box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        box.accepted.connect(self.accept); box.rejected.connect(self.reject)
        form.addRow(box)

    def values(self):
        return {k: w.value() for k, w in self.edits.items()}


class SaveFigDialog(QDialog):
    """Диалог задания размера и DPI при сохранении."""
    def __init__(self, xt_available: bool, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Save figure…")
        self.resize(260, 140)
        form = QFormLayout(self)

        self.w_spin = QDoubleSpinBox(decimals=1, minimum=1, maximum=30)
        self.h_spin = QDoubleSpinBox(decimals=1, minimum=1, maximum=30)
        self.dpi_spin = QDoubleSpinBox(decimals=0, minimum=50, maximum=600)

        self.w_spin.setValue(6.0); self.h_spin.setValue(4.0); self.dpi_spin.setValue(300)

        form.addRow("Width (in)", self.w_spin)
        form.addRow("Height (in)", self.h_spin)
        form.addRow("DPI",        self.dpi_spin)

        self.chk_xt = QCheckBox("Also save x(t)")
        self.chk_xt.setChecked(True); self.chk_xt.setEnabled(xt_available)
        form.addRow(self.chk_xt)

        box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        box.accepted.connect(self.accept); box.rejected.connect(self.reject)
        form.addRow(box)

    def values(self):
        return (self.w_spin.value(),
                self.h_spin.value(),
                int(self.dpi_spin.value()),
                self.chk_xt.isChecked())

# --------------------------------------------------------------------------- #
# 3. Matplotlib-канвасы                                                       #
# --------------------------------------------------------------------------- #
class MplCanvas(QWidget):
    def __init__(self):
        super().__init__()
        self.fig, self.ax = plt.subplots()
        self.canvas = FigureCanvasQTAgg(self.fig)
        lay = QVBoxLayout(self); lay.setContentsMargins(0,0,0,0)
        lay.addWidget(self.canvas)
        lay.addWidget(NavigationToolbar2QT(self.canvas, self))
        self.canvas.mpl_connect("scroll_event", self._on_scroll)

    def _on_scroll(self, e):
        if e.inaxes is None: return
        s = 1.2 if e.button == "up" else 0.8
        xm, ym = e.xdata, e.ydata; ax = e.inaxes
        ax.set_xlim(xm+(ax.get_xlim()[0]-xm)*s, xm+(ax.get_xlim()[1]-xm)*s)
        ax.set_ylim(ym+(ax.get_ylim()[0]-ym)*s, ym+(ax.get_ylim()[1]-ym)*s)
        self.canvas.draw_idle()

class CombinedCanvas(QWidget):
    """Два под-графика — фазовый портрет и x(t) — РЯДОМ друг с другом."""
    def __init__(self):
        super().__init__()

        # ⬇️  Широкая, невысокая фигура  – удобнее для горизонтального деления
        self.fig = plt.figure(figsize=(9, 4))      # ← здесь изначальный размер

        # ▸ вместо (2, 1, 1) и (2, 1, 2) используем сетку 1×2
        gs = self.fig.add_gridspec(1, 2, wspace=0.30)   # wspace — зазор между осями
        self.ax_phase = self.fig.add_subplot(gs[0, 0])
        self.ax_xt    = self.fig.add_subplot(gs[0, 1])

        self.canvas = FigureCanvasQTAgg(self.fig)
        lay = QVBoxLayout(self); lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.canvas)
        lay.addWidget(NavigationToolbar2QT(self.canvas, self))

        # колесо мыши — масштаб внутри конкретной оси
        self.canvas.mpl_connect("scroll_event", self._on_scroll)

    def _on_scroll(self, e):
        if e.inaxes is None: return
        s = 1.2 if e.button == "up" else 0.8
        xm, ym = e.xdata, e.ydata; ax = e.inaxes
        ax.set_xlim(xm+(ax.get_xlim()[0]-xm)*s, xm+(ax.get_xlim()[1]-xm)*s)
        ax.set_ylim(ym+(ax.get_ylim()[0]-ym)*s, ym+(ax.get_ylim()[1]-ym)*s)
        self.canvas.draw_idle()

# --------------------------------------------------------------------------- #
# 4. Полноэкранное окно фазового портрета                                     #
# --------------------------------------------------------------------------- #
class FullScreenPhaseWindow(QMainWindow):
    """Показывает текущий Figure на весь экран, а при закрытии
       возвращает канвас обратно в MainWindow."""
    def __init__(self, phase_canvas: MplCanvas, parent: "MainWindow"):
        super().__init__(parent)
        self._main = parent
        self.setWindowTitle("Phase plane — Full screen")

        # «вырываем» канвас из главного окна
        self.canvas = phase_canvas
        self.canvas.setParent(None)

        central = QWidget()
        lay = QVBoxLayout(central); lay.setContentsMargins(0,0,0,0)
        lay.addWidget(self.canvas)
        tb = NavigationToolbar2QT(self.canvas.canvas, self)
        exit_act = QAction("Exit [F11 / Esc]", self)
        exit_act.triggered.connect(self.close)
        tb.addAction(exit_act)
        lay.addWidget(tb)
        self.setCentralWidget(central)

        self.showFullScreen()

    # ------- клавиши выхода -------
    def keyPressEvent(self, ev):
        if ev.key() in (Qt.Key_Escape, Qt.Key_F11):
            self.close()
        else:
            super().keyPressEvent(ev)

    # ------- возвращаем канвас обратно -------
    def closeEvent(self, ev):
        splitter: QSplitter = self._main.tabs.widget(0).findChild(QSplitter)
        splitter.addWidget(self.canvas)            # возвращаем в правую панель
        self._main.phase_fs_win = None
        super().closeEvent(ev)

# --------------------------------------------------------------------------- #
# 5. Главное окно                                                             #
# --------------------------------------------------------------------------- #
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Dynamic-System Sandbox — BT only")
        self.resize(1100, 700)

        # диапазоны и система
        self.range = dict(mu1_min=-15, mu1_max=15, mu2_min=-15, mu2_max=15)
        self.phase_range = dict(x_min=-3, x_max=25, y_min=-2, y_max=2)
        self.f_expr = "-2*exp(-x) + exp(-2*x) + y"
        self.g_expr = "(-x + mu2*y + mu1)*0.01"

        # списки и компиляция
        self.eq_funcs, self.hopf_branches, self.bt_pts = [], [], []
        self._compile_system()

        # GUI-состояние
        self.current_mu = None
        self.traj_lines, self.xt_data = [], []
        self.nullcline_art, self.nullcline_pts = [], []
        self.field_art, self.sep_lines = [], []
        self.equilibria = []; self.show_field = False
        self.param_marker = None; self.phase_fs_win = None
        self.limit_cycle_line = None
        self.color_cycle = itertools.cycle(
            plt.rcParams["axes.prop_cycle"].by_key()["color"]
        )

        # --- канвасы и вкладки ---
        self.param_canvas = MplCanvas()
        self.phase_canvas = MplCanvas()
        self.bt_canvas    = MplCanvas()
        self.combined_canvas = CombinedCanvas()

        spl = QSplitter(Qt.Orientation.Horizontal)
        spl.addWidget(self.param_canvas); spl.addWidget(self.phase_canvas)

        tabs = QTabWidget(); self.tabs = tabs
        w_plots = QWidget(); QVBoxLayout(w_plots).addWidget(spl)
        tabs.addTab(w_plots, "Plots")

        self.eq_table = QTableWidget(0,5)
        self.eq_table.setHorizontalHeaderLabels(["x","y","type","λ₁","λ₂"])
        tabs.addTab(self.eq_table, "Equilibria")

        self.bt_table = QTableWidget(0,6)
        self.bt_table.setHorizontalHeaderLabels(["x","y","μ₁","μ₂","λ₁","λ₂"])
        bt_widget = QWidget(); bt_layout = QVBoxLayout(bt_widget); bt_layout.setContentsMargins(2,2,2,2)
        bt_layout.addWidget(self.bt_canvas); bt_layout.addWidget(self.bt_table)
        tabs.addTab(bt_widget, "BT"); self.bt_tab_index = tabs.indexOf(bt_widget)

        combined_widget = QWidget(); QVBoxLayout(combined_widget).addWidget(self.combined_canvas)
        tabs.addTab(combined_widget, "Combined")
        self.combined_widget = combined_widget

        self.setCentralWidget(tabs)

        # тулбар, графики, события
        self._build_toolbar()
        self._conf_param_axes(); self._conf_phase_axes()
        self._connect_events()
        self.xt_fig = None; self.xt_ax = None
        self._refresh_combined()   # начальная пустая вкладка

    # -------------------- 5.1  Компиляция системы + BT-поиск --------------- #
    def _compile_system(self):
        x,y,m1,m2 = sp.symbols("x y mu1 mu2")
        f_sym, g_sym = sp.sympify(self.f_expr), sp.sympify(self.g_expr)
        J = sp.Matrix([f_sym,g_sym]).jacobian([x,y])
        detJ, trJ = sp.simplify(J.det()), sp.simplify(J.trace())

        self.f_lam  = sp.lambdify((x,y,m1,m2), f_sym, modules=[SAFE_MODULE,'numpy'])
        self.g_lam  = sp.lambdify((x,y,m1,m2), g_sym, modules=[SAFE_MODULE,'numpy'])
        self.J11 = sp.lambdify((x,y,m1,m2), J[0,0], modules=[SAFE_MODULE,'numpy'])
        self.J12 = sp.lambdify((x,y,m1,m2), J[0,1], modules=[SAFE_MODULE,'numpy'])
        self.J21 = sp.lambdify((x,y,m1,m2), J[1,0], modules=[SAFE_MODULE,'numpy'])
        self.J22 = sp.lambdify((x,y,m1,m2), J[1,1], modules=[SAFE_MODULE,'numpy'])
        self.detJ_lam = sp.lambdify((x,y,m1,m2), detJ, modules=[SAFE_MODULE,'numpy'])
        self.trJ_lam  = sp.lambdify((x,y,m1,m2), trJ , modules=[SAFE_MODULE,'numpy'])
        self.rhs_func = lambda t,s,μ1,μ2: np.array([
            self.f_lam(s[0],s[1],μ1,μ2),
            self.g_lam(s[0],s[1],μ1,μ2)
        ])

        self.eq_funcs.clear(); self.hopf_branches.clear()
        try:
            for sol in sp.solve([f_sym,g_sym],[x,y],dict=True):
                xi, yi = sol[x], sol[y]
                fx = sp.lambdify((m1,m2), xi, modules=[SAFE_MODULE,'numpy'])
                fy = sp.lambdify((m1,m2), yi, modules=[SAFE_MODULE,'numpy'])
                self.eq_funcs.append((fx,fy))
                tr_eq = trJ.subs({x:xi, y:yi})
                roots = sp.solve(tr_eq, m2)
                if roots:
                    phi = roots[0]; phi_f = sp.lambdify(m1, phi, modules=[SAFE_MODULE,'numpy'])
                    self.hopf_branches.append((phi_f, fx, fy))
        except (sp.SympifyError, NotImplementedError):
            pass
        self._compute_bt_points()

    def _compute_bt_points(self):
        f,g = self.f_lam, self.g_lam
        trJ, detJ = self.trJ_lam, self.detJ_lam
        def F(v): x,y,u,v2 = v; return [f(x,y,u,v2), g(x,y,u,v2), trJ(x,y,u,v2), detJ(x,y,u,v2)]
        guesses = [(0,0,1,0),( .5,.5,1,-1),(-.5,-.5,1,-1),(1,-1,1,-1),(2,0,.5,.5),(-2,0,.5,-.5)]
        pts=[]
        for g0 in guesses:
            sol=root(F,g0,method='hybr',tol=1e-8)
            if not sol.success: continue
            x0,y0,u,v2 = sol.x
            if not(self.range['mu1_min']<=u<=self.range['mu1_max'] and
                   self.range['mu2_min']<=v2<=self.range['mu2_max']): continue
            if any(np.hypot(u-a[2],v2-a[3])<1e-6 for a in pts): continue
            pts.append((x0,y0,u,v2))
        self.bt_pts=pts

    # ---------- 5.2  UI-элементы: тулбар, новые действия ------------------ #
    def _build_toolbar(self):
        bar = self.addToolBar("Controls"); bar.setIconSize(QSize(16,16))

        # --- диалоги системы/диапазонов
        for txt,slot in (("Edit system",self._dlg_system),
                         ("Set param range",self._dlg_range),
                         ("Set phase range",self._dlg_phase)):
            act = QAction(txt,self); act.triggered.connect(slot); bar.addAction(act)
        bar.addSeparator()

        # μ-спины
        bar.addWidget(QLabel("μ₁:")); self.mu1_spin = QDoubleSpinBox(decimals=3)
        self.mu1_spin.setRange(self.range["mu1_min"],self.range["mu1_max"])
        self.mu1_spin.valueChanged.connect(self._spin_changed); bar.addWidget(self.mu1_spin)
        bar.addWidget(QLabel("μ₂:")); self.mu2_spin = QDoubleSpinBox(decimals=3)
        self.mu2_spin.setRange(self.range["mu2_min"],self.range["mu2_max"])
        self.mu2_spin.valueChanged.connect(self._spin_changed); bar.addWidget(self.mu2_spin)

        bar.addSeparator()
        # интегратор + время
        bar.addWidget(QLabel("Integrator:")); self.integrator_cb = QComboBox()
        self.integrator_cb.addItems(["RK23","RK45","DOP853","LSODA","Radau","BDF"])
        bar.addWidget(self.integrator_cb)
        bar.addSeparator()
        bar.addWidget(QLabel("T⁺:")); self.time_fwd_spin = QDoubleSpinBox(decimals=1); self.time_fwd_spin.setRange(0.1,1000); self.time_fwd_spin.setValue(15); bar.addWidget(self.time_fwd_spin)
        bar.addWidget(QLabel("T⁻:")); self.time_bwd_spin = QDoubleSpinBox(decimals=1); self.time_bwd_spin.setRange(0.1,1000); self.time_bwd_spin.setValue(15); bar.addWidget(self.time_bwd_spin)

        bar.addSeparator()
        bar.addWidget(QLabel("rtol:")); self.rtol_spin = QDoubleSpinBox(decimals=6); self.rtol_spin.setRange(1e-12,1); self.rtol_spin.setSingleStep(1e-6); self.rtol_spin.setValue(1e-4); bar.addWidget(self.rtol_spin)
        bar.addWidget(QLabel("atol:")); self.atol_spin = QDoubleSpinBox(decimals=6); self.atol_spin.setRange(1e-15,1); self.atol_spin.setSingleStep(1e-10); self.atol_spin.setValue(1e-4); bar.addWidget(self.atol_spin)

        bar.addSeparator()
        clear_btn = QPushButton("Clear"); clear_btn.clicked.connect(self._clear_traj); bar.addWidget(clear_btn)

        bar.addSeparator()
        self.act_vector = QAction("Vector field", self, checkable=True); self.act_vector.triggered.connect(self._toggle_vect); bar.addAction(self.act_vector)
        self.act_sep = QAction("Separatrices", self, checkable=True); self.act_sep.triggered.connect(self._toggle_sep); bar.addAction(self.act_sep)
        self.act_bt  = QAction("BT", self, checkable=True); self.act_bt.triggered.connect(self._toggle_bt); bar.addAction(self.act_bt)

        bar.addSeparator()
        self.act_xt = QAction("Plot x(t)", self); self.act_xt.setEnabled(False); self.act_xt.triggered.connect(self._update_xt); bar.addAction(self.act_xt)

        bar.addSeparator()
        self.act_full = QAction("Phase fullscreen", self); self.act_full.triggered.connect(self._open_phase_fullscreen); bar.addAction(self.act_full)

        bar.addSeparator()
        self.act_save = QAction("Save figure…", self); self.act_save.triggered.connect(self._save_figure); bar.addAction(self.act_save)

        # индикатор прогресса
        self.progress = QProgressBar(); self.progress.setRange(0,0); self.progress.setVisible(False)
        self.statusBar().addPermanentWidget(self.progress)

    # ---------- 5.3  Полноэкранный режим -------- #
    def _open_phase_fullscreen(self):
        if self.phase_fs_win and self.phase_fs_win.isVisible():
            self.phase_fs_win.close(); self.phase_fs_win = None
        else:
            self.phase_fs_win = FullScreenPhaseWindow(self.phase_canvas, parent=self)
            self.phase_fs_win.destroyed.connect(lambda *_: setattr(self,'phase_fs_win',None))

    # ---------- 5.4  Сохранение рисунков --------- #
    def _save_figure(self):
        # какая канва активна?
        active_canvas = self.combined_canvas if self.tabs.currentWidget() is self.combined_widget else self.phase_canvas
        dlg = SaveFigDialog(self.xt_fig is not None and active_canvas is self.phase_canvas, self)
        if not dlg.exec():
            return
        w, h, dpi, save_xt = dlg.values()

        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save figure", "",
            "PNG (*.png);;PDF (*.pdf);;SVG (*.svg)"
        )
        if not file_path:
            return
        if not os.path.splitext(file_path)[1]:
            file_path += ".png"  # расширение по-умолчанию

        try:
            from matplotlib.backends.backend_agg import FigureCanvasAgg

            # ---------- активный рисунок ----------
            fig_src = active_canvas.fig
            fig_tmp = plt.figure(figsize=(w, h), dpi=dpi)
            tmp_canvas = FigureCanvasAgg(fig_tmp)

            ax_src = fig_src.axes[0] if len(fig_src.axes)==1 else None
            if ax_src:                       # одиночная ось
                ax_tmp = fig_tmp.add_subplot(111)
                for art in ax_src.get_children():
                    art.remove()
                    ax_tmp.add_artist(art)
            else:                            # несколько осей – просто копируем whole figure
                fig_src.canvas.draw()
                fig_src.savefig(file_path, dpi=dpi, bbox_inches="tight")
                plt.close(fig_tmp)
                self.statusBar().showMessage("Figure saved")
                return

            tmp_canvas.draw()
            fig_tmp.savefig(file_path, dpi=dpi, bbox_inches="tight")
            plt.close(fig_tmp)

            # вернуть Artists на место
            for art in list(ax_tmp.get_children()):
                art.remove()
                ax_src.add_artist(art)
            active_canvas.canvas.draw_idle()

            # ---------- x(t) ----------
            if save_xt and self.xt_fig:
                base, ext = os.path.splitext(file_path)
                xt_file = f"{base}_xt{ext}"

                fig_xt_src = self.xt_fig
                fig_xt_tmp = plt.figure(figsize=(w, h), dpi=dpi)
                xt_canvas = FigureCanvasAgg(fig_xt_tmp)

                ax_xt_src = fig_xt_src.axes[0]
                ax_xt_tmp = fig_xt_tmp.add_subplot(111)
                for art in ax_xt_src.get_children():
                    art.remove()
                    ax_xt_tmp.add_artist(art)

                xt_canvas.draw()
                fig_xt_tmp.savefig(xt_file, dpi=dpi, bbox_inches="tight")
                plt.close(fig_xt_tmp)

                for art in list(ax_xt_tmp.get_children()):
                    art.remove()
                    ax_xt_src.add_artist(art)
                fig_xt_src.canvas.draw_idle()

            self.statusBar().showMessage("Figure saved")

        except Exception as err:
            QMessageBox.critical(self, "Save error", str(err))


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
                for x0, y0, m1, m2 in self.bt_pts:
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
    def _spin_changed(self, _=None):
        if self.limit_cycle_line is not None:
            self.limit_cycle_line.remove()
            self.limit_cycle_line = None
            self.statusBar().clearMessage()

        μ1, μ2 = self.mu1_spin.value(), self.mu2_spin.value()
        self.current_mu = (μ1, μ2)
        self._clear_traj()
        self._toggle_sep(False)
        if self.param_marker:
            self.param_marker.remove()
        self.param_marker, = self.param_canvas.ax.plot(μ1, μ2, "xr", ms=8)
        self.param_canvas.canvas.draw_idle()
        self._draw_nullclines(μ1, μ2)
        if self.show_field:
            self._draw_vect()
        self.statusBar().showMessage(f"μ=({μ1:.3g}, {μ2:.3g})")
        self._refresh_combined()

    def _param_click(self, e):
        if not (e.inaxes and e.dblclick):
            return
        μ1, μ2 = e.xdata, e.ydata
        self.mu1_spin.blockSignals(True)
        self.mu1_spin.setValue(μ1)
        self.mu1_spin.blockSignals(False)
        self.mu2_spin.blockSignals(True)
        self.mu2_spin.setValue(μ2)
        self.mu2_spin.blockSignals(False)
        self._spin_changed()

    # ----------------------- 4.5  Изоклины и равновесие ------------------- #
    def _find_eq(self, μ1, μ2):
        x_min, x_max = self.phase_range["x_min"], self.phase_range["x_max"]
        y_min, y_max = self.phase_range["y_min"], self.phase_range["y_max"]
        sols, tol_f, tol_xy = [], 1e-4, 1e-3
        guesses = [(x0, y0) for x0 in np.linspace(x_min, x_max, 10)
                            for y0 in np.linspace(y_min, y_max, 10)]
        fg = lambda v: (self.f_lam(v[0], v[1], μ1, μ2),
                        self.g_lam(v[0], v[1], μ1, μ2))
        for x0, y0 in guesses:
            sol = root(lambda v: [np.tanh(fg(v)[0]), np.tanh(fg(v)[1])],
                       [x0, y0], method="hybr",
                       options={"maxfev": 200, "xtol": 1e-6})
            if sol.success:
                xe, ye = sol.x
                if max(abs(fg((xe, ye))[0]), abs(fg((xe, ye))[1])) > tol_f:
                    continue
                if any(np.hypot(xe - xs, ye - ys) < tol_xy for xs, ys in sols):
                    continue
                sols.append((xe, ye))
        return [(round(x,6), round(y,6)) for x,y in sols]

    def _draw_nullclines(self, μ1, μ2):
        ax = self.phase_canvas.ax
        for cs in self.nullcline_art: cs.remove()
        for obj in self.nullcline_pts: obj.remove()
        self.nullcline_art.clear(); self.nullcline_pts.clear()
        self.eq_table.setRowCount(0); self.equilibria.clear()

        xmin, xmax = ax.get_xlim(); ymin, ymax = ax.get_ylim()
        xx, yy = np.meshgrid(np.linspace(xmin, xmax, 300),
                             np.linspace(ymin, ymax, 300))
        with np.errstate(over='ignore', invalid='ignore'):
            F = self.f_lam(xx, yy, μ1, μ2)
            G = self.g_lam(xx, yy, μ1, μ2)
        F = np.nan_to_num(F); G = np.nan_to_num(G)
        cf = ax.contour(xx, yy, F, levels=[0],
                        colors="blue", linestyles="--", linewidths=1.2)
        cg = ax.contour(xx, yy, G, levels=[0],
                        colors="green", linestyles="-", linewidths=1.2)
        self.nullcline_art += [cf, cg]

        for xf, yf in self._find_eq(μ1, μ2):
            if not(xmin <= xf <= xmax and ymin <= yf <= ymax):
                continue
            J11 = float(self.J11(xf, yf, μ1, μ2))
            J12 = float(self.J12(xf, yf, μ1, μ2))
            J21 = float(self.J21(xf, yf, μ1, μ2))
            J22 = float(self.J22(xf, yf, μ1, μ2))
            Jmat = np.array([[J11, J12], [J21, J22]])
            tr, det = J11 + J22, float(self.detJ_lam(xf, yf, μ1, μ2))
            discr = tr*tr - 4*det
            eigvals, eigvecs = np.linalg.eig(Jmat)
            tol = 1e-6
            if det < 0:
                typ, color = "saddle", "red"
            else:
                if discr > tol:
                    typ, color = ("stable node", "blue") if tr < 0 else ("unstable node", "cyan")
                else:
                    if abs(tr) < tol:
                        typ, color = "center", "green"
                    elif tr < 0:
                        typ, color = "stable focus", "purple"
                    else:
                        typ, color = "unstable focus", "magenta"
            pt, = ax.plot(xf, yf, "o", color=color, ms=8)
            txt = ax.text(xf, yf, typ, color=color,
                          fontsize="small", va="bottom", ha="right")
            self.nullcline_pts += [pt, txt]

            row = self.eq_table.rowCount()
            self.eq_table.insertRow(row)
            for col, val in enumerate((xf, yf, typ, *eigvals)):
                item = QTableWidgetItem(f"{val:.6g}" if isinstance(val, float) else str(val))
                self.eq_table.setItem(row, col, item)

            self.equilibria.append({
                'x': xf, 'y': yf, 'type': typ,
                'eigvals': eigvals, 'eigvecs': eigvecs
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
        self._refresh_combined()

    # -------------------- 4.6  Векторное поле/сепаратрисы --------------- #
    def _toggle_vect(self, chk):
        self.show_field = chk
        (self._draw_vect() if chk else self._clear_vect())
        self._refresh_combined()

    def _clear_vect(self):
        for art in self.field_art: art.remove()
        self.field_art.clear()
        self.phase_canvas.canvas.draw_idle()

    def _draw_vect(self):
        self._clear_vect()
        if not self.current_mu:
            return
        μ1, μ2 = self.current_mu
        ax = self.phase_canvas.ax
        XX, YY = np.meshgrid(np.linspace(*ax.get_xlim(), 20),
                             np.linspace(*ax.get_ylim(), 20))
        U = self.f_lam(XX, YY, μ1, μ2)
        V = self.g_lam(XX, YY, μ1, μ2)
        M = np.hypot(U, V)
        M[M == 0] = 1
        Q = ax.quiver(XX, YY, U/M, V/M, angles="xy", pivot="mid", alpha=0.6)
        self.field_art.append(Q)
        self.phase_canvas.canvas.draw_idle()

    # -------------------- 4.7  Сепаратрисы ------------------------------- #
    def _toggle_sep(self, chk):
        (self._draw_sep() if chk else self._clear_sep())
        self._refresh_combined()

    def _clear_sep(self):
        for ln in self.sep_lines:
            ln.remove()
        self.sep_lines.clear()
        self.phase_canvas.canvas.draw_idle()

    def _draw_sep(self):
        self._clear_sep()
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
                v = np.real_if_close(vec[:, i])
                if abs(lam) < 1e-4:
                    continue
                v /= np.linalg.norm(v)
                for sgn in (+1, -1):
                    start = np.array([x0, y0]) + sgn * 5e-3 * v
                    span = (0, 60) if lam > 0 else (0, -60)
                    sol = solve_ivp(lambda t, y: self.rhs_func(t, y, μ1, μ2),
                                    span, start,
                                    max_step=0.2, rtol=1e-4, atol=1e-7)
                    ln, = ax.plot(sol.y[0], sol.y[1], 'k--', lw=1)
                    self.sep_lines.append(ln)
        self.phase_canvas.canvas.draw_idle()

    def _detect_limit_cycle(self, sol):
        t = sol.t
        x = sol.y[0]
        y = sol.y[1]

        crossings = [t[i] + (-y[i]) / (y[i + 1] - y[i]) * (t[i + 1] - t[i])
                     for i in range(len(t) - 1) if y[i] < 0 <= y[i + 1]]

        if len(crossings) < 5:
            return None, None, None

        periods = np.diff(crossings)
        last3 = periods[-3:]
        if np.std(last3) / np.mean(last3) < 0.05:
            T = np.mean(last3)
            t0, t1 = crossings[-2], crossings[-1]
            msk = (t >= t0) & (t <= t1)
            return x[msk], y[msk], T
        return None, None, None

    # -------------------- 4.8  Клики по фазовой плоскости -------------- #
    def _phase_click(self, e):
        ax = self.phase_canvas.ax
        if e.button == 3 and e.inaxes == ax:
            self._del_traj(e.xdata, e.ydata)
            return
        if e.button != 1 or not e.dblclick or not self.current_mu:
            return

        x0, y0 = e.xdata, e.ydata
        μ1, μ2 = self.current_mu
        method = self.integrator_cb.currentText()

        T_fwd = float(self.time_fwd_spin.value())
        T_bwd = float(self.time_bwd_spin.value())
        max_step = max(T_fwd, T_bwd) / 100.0

        rtol = float(self.rtol_spin.value())
        atol = float(self.atol_spin.value())

        def rhs_sat(t, s):
            dx, dy = self.rhs_func(t, s, μ1, μ2)
            if not np.isfinite(dx + dy):
                return np.zeros(2)
            v = np.hypot(dx, dy)
            return np.array([dx, dy]) if v < 1e3 else np.array([dx * 1e3 / v, dy * 1e3 / v])

        self.progress.setVisible(True)

        sol_f = solve_ivp(rhs_sat, (0, T_fwd), [x0, y0],
                          method=method, max_step=max_step,
                          rtol=rtol, atol=atol)
        sol_b = solve_ivp(rhs_sat, (0, -T_bwd), [x0, y0],
                          method=method, max_step=max_step,
                          rtol=rtol, atol=atol)

        self.progress.setVisible(False)

        if self.limit_cycle_line:
            self.limit_cycle_line.remove()
            self.limit_cycle_line = None

        xs_c, ys_c, period = self._detect_limit_cycle(sol_f)
        if xs_c is not None:
            line, = ax.plot(xs_c, ys_c, linewidth=2.5, label="Limit cycle")
            self.limit_cycle_line = line
            self.statusBar().showMessage(f"Обнаружен предельный цикл, T ≈ {period:.3g}")
            ax.legend(fontsize="small")

        xs = np.concatenate([sol_b.y[0][::-1], sol_f.y[0]])
        ys = np.concatenate([sol_b.y[1][::-1], sol_f.y[1]])
        color = next(self.color_cycle)
        ln, = ax.plot(xs, ys, color=color)
        self.traj_lines.append(ln)
        t_full = np.concatenate([sol_b.t[::-1], sol_f.t])
        self.xt_data.append((t_full, xs, color))

        self.phase_canvas.canvas.draw_idle()
        self.act_xt.setEnabled(True)
        if self.xt_fig:
            self._update_xt()
        self._refresh_combined()

    # -------------------- 4.9  Управление траекториями ----------------- #
    def _del_traj(self, x, y):
        if not self.traj_lines:
            return
        ax = self.phase_canvas.ax
        thresh = 0.05 * max(
            ax.get_xlim()[1] - ax.get_xlim()[0],
            ax.get_ylim()[1] - ax.get_ylim()[0]
        )
        for i, ln in enumerate(self.traj_lines):
            if np.min(np.hypot(ln.get_xdata() - x,
                                ln.get_ydata() - y)) < thresh:
                ln.remove()
                self.traj_lines.pop(i)
                self.xt_data.pop(i)
                self.phase_canvas.canvas.draw_idle()
                if self.xt_fig:
                    if self.traj_lines:
                        self._update_xt()
                    else:
                        self._clear_xt()
                break
        self._refresh_combined()

    def _clear_traj(self):
        for ln in self.traj_lines:
            ln.remove()
        self.traj_lines.clear()
        self.xt_data.clear()
        if self.limit_cycle_line is not None:
            self.limit_cycle_line.remove()
            self.limit_cycle_line = None
        self.phase_canvas.canvas.draw_idle()
        self._clear_xt()
        self.act_xt.setEnabled(False)
        self.statusBar().clearMessage()
        self._refresh_combined()

    # -------------------- 4.10  x(t) окно ------------------------------- #
    def _update_xt(self):
        if not self.xt_data:
            return
        if not self.xt_fig:
            self.xt_fig, self.xt_ax = plt.subplots()
            self.xt_ax.set_xlabel("t")
            self.xt_ax.set_ylabel("x(t)")
        self.xt_ax.cla()
        self.xt_ax.set_title("x(t) for all trajectories")
        for t, x, color in self.xt_data:
            self.xt_ax.plot(t, x, color=color)
        self.xt_fig.canvas.draw_idle()
        self.xt_fig.show()
        self._refresh_combined()

    def _clear_xt(self):
        if self.xt_fig:
            self.xt_ax.cla()
            self.xt_ax.set_title("x(t) for all trajectories")
            self.xt_ax.set_xlabel("t")
            self.xt_ax.set_ylabel("x(t)")
            self.xt_fig.canvas.draw_idle()

    # -------------------- 4.11  Combined вкладка ------------------------ #
    def _refresh_combined(self):
        fig = self.combined_canvas.fig
        fig.clf()
        gs = fig.add_gridspec(1, 2, wspace=0.30)  # ⬅️ сетка 1×2
        ax_phase = fig.add_subplot(gs[0, 0])
        ax_xt = fig.add_subplot(gs[0, 1])

        # ----- фазовый портрет -----
        ax_phase.set_title("Phase plane (x, y)")
        ax_phase.set_xlabel("x"); ax_phase.set_ylabel("y")
        ax_phase.set_xlim(self.phase_range["x_min"], self.phase_range["x_max"])
        ax_phase.set_ylim(self.phase_range["y_min"], self.phase_range["y_max"])

        if self.current_mu:
            μ1, μ2 = self.current_mu
            xmin, xmax = self.phase_range["x_min"], self.phase_range["x_max"]
            ymin, ymax = self.phase_range["y_min"], self.phase_range["y_max"]
            xx, yy = np.meshgrid(np.linspace(xmin, xmax, 300),
                                 np.linspace(ymin, ymax, 300))
            with np.errstate(over='ignore', invalid='ignore'):
                F = self.f_lam(xx, yy, μ1, μ2)
                G = self.g_lam(xx, yy, μ1, μ2)
            F = np.nan_to_num(F); G = np.nan_to_num(G)
            ax_phase.contour(xx, yy, F, levels=[0],
                             colors="blue", linestyles="--", linewidths=1.2)
            ax_phase.contour(xx, yy, G, levels=[0],
                             colors="green", linestyles="-", linewidths=1.2)

            color_map = {"saddle":"red","stable node":"blue","unstable node":"cyan",
                         "stable focus":"purple","unstable focus":"magenta","center":"green"}
            for eq in self.equilibria:
                color = color_map.get(eq['type'], 'black')
                ax_phase.plot(eq['x'], eq['y'], 'o', color=color, ms=7)

            if self.show_field:
                XX, YY = np.meshgrid(np.linspace(*ax_phase.get_xlim(), 20),
                                     np.linspace(*ax_phase.get_ylim(), 20))
                U = self.f_lam(XX, YY, μ1, μ2)
                V = self.g_lam(XX, YY, μ1, μ2)
                M = np.hypot(U, V); M[M==0]=1
                ax_phase.quiver(XX, YY, U/M, V/M, angles="xy", pivot="mid", alpha=0.6)

        for ln in self.traj_lines:
            ax_phase.plot(ln.get_xdata(), ln.get_ydata(), color=ln.get_color())
        if self.limit_cycle_line:
            ax_phase.plot(self.limit_cycle_line.get_xdata(),
                          self.limit_cycle_line.get_ydata(),
                          linewidth=2.5, label="Limit cycle")
            ax_phase.legend(fontsize="small", loc="upper right")

        # ----- x(t) -----
        ax_xt.set_title("x(t) for all trajectories")
        ax_xt.set_xlabel("t")
        ax_xt.set_ylabel("x(t)")
        for t, x, color in self.xt_data:
            ax_xt.plot(t, x, color=color)

        self.combined_canvas.canvas.draw_idle()

    # -------------------- 4.12  Диалоги ------------------------------- #
    def _dlg_system(self):
        dlg = SystemDialog(self.f_expr, self.g_expr, self)
        if dlg.exec():
            self.f_expr, self.g_expr = dlg.texts()
            try:
                self._compile_system()
                if self.current_mu:
                    self._draw_nullclines(*self.current_mu)
                if self.show_field:
                    self._draw_vect()
                if self.act_sep.isChecked():
                    self._draw_sep()
                if self.act_bt.isChecked():
                    self._toggle_bt(True)
                self._refresh_combined()
                self.statusBar().showMessage("System updated")
            except Exception as e:
                QMessageBox.critical(self, "Error", str(e))

    def _dlg_range(self):
        dlg = RangeDialog(self.range, self)
        if dlg.exec():
            self.range = dlg.values()
            self.mu1_spin.setRange(self.range["mu1_min"],
                                   self.range["mu1_max"])
            self.mu2_spin.setRange(self.range["mu2_min"],
                                   self.range["mu2_max"])
            self._compile_system()
            self._conf_param_axes()
            if self.act_bt.isChecked():
                self._toggle_bt(True)
            self._refresh_combined()

    def _dlg_phase(self):
        dlg = PhaseRangeDialog(self.phase_range, self)
        if dlg.exec():
            self.phase_range = dlg.values()
            self._conf_phase_axes()
            if self.current_mu:
                self._draw_nullclines(*self.current_mu)
            self._refresh_combined()

# --------------------------------------------------------------------------- #
# main                                                                        #
# --------------------------------------------------------------------------- #
def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
