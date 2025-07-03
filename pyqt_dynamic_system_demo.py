# -*- coding: utf-8 -*-
"""pyqt_dynamic_system_demo.py — «Двух-параметрическая песочница» с векторным полем
=================================================================
• Двойной клик на плоскости μ → перестроить изоклины и показать СР
  с классификацией (saddle, node, focus, center).
• Векторное поле (toggle “Vector field”).
• Двойной клик на фазовой плоскости → добавить траекторию t∈[−T,T].
• Настройка диапазонов, выбор интегратора, Clear, Zoom/Pan, Max Param/Phase.
• Вкладки: “Plots” и “Equilibria” (таблица СР).
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
        if event.inaxes is None:
            return
        scale = 1.2 if event.button == "up" else 0.8
        ax = event.inaxes
        xm, ym = event.xdata, event.ydata
        xmin, xmax = ax.get_xlim(); ymin, ymax = ax.get_ylim()
        ax.set_xlim(xm + (xmin - xm) * scale, xm + (xmax - xm) * scale)
        ax.set_ylim(ym + (ymin - ym) * scale, ym + (ymax - ym) * scale)
        self.canvas.draw_idle()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Dynamic System Sandbox — equilibria & field")
        self.resize(1100, 650)

        # default system
        self.f_expr = "y"
        self.g_expr = "mu1*(1 - x**2)*y - x"
        self._compile_system()

        # state
        self.range = dict(mu1_min=-3, mu1_max=3, mu2_min=-3, mu2_max=3)
        self.phase_range = dict(x_min=-3, x_max=3, y_min=-3, y_max=3)
        self.current_mu: tuple[float, float] | None = None
        self.traj_lines: list[Line2D] = []
        self.nullcline_artists = []
        self.nullcline_points: list[Line2D] = []
        self.field_artists = []
        self.show_field = False
        self.param_marker: Line2D | None = None
        self.color_cycle = itertools.cycle(
            plt.rcParams["axes.prop_cycle"].by_key()["color"]
        )

        # canvases in splitter
        self.param_canvas = MplCanvas()
        self.phase_canvas = MplCanvas()
        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        self.splitter.addWidget(self.param_canvas)
        self.splitter.addWidget(self.phase_canvas)
        self.splitter.setSizes([1, 1])

        # tabs: Plots and Equilibria
        tabs = QTabWidget()
        page1 = QWidget()
        lay1 = QVBoxLayout(page1)
        lay1.addWidget(self.splitter)
        tabs.addTab(page1, "Plots")

        self.eq_table = QTableWidget(0, 5, self)
        self.eq_table.setHorizontalHeaderLabels(["x", "y", "type", "λ₁", "λ₂"])
        tabs.addTab(self.eq_table, "Equilibria")

        self.setCentralWidget(tabs)

        # toolbar and config
        self._build_toolbar()
        self._configure_param_axes()
        self._configure_phase_axes()
        self._connect_events()

    def _build_toolbar(self):
        bar = self.addToolBar("Controls")
        bar.setIconSize(QSize(16, 16))
        for txt, slot in [
            ("Edit system", self._edit_system),
            ("Set param range", self._edit_range),
            ("Set phase range", self._edit_phase_range),
        ]:
            act = QAction(txt, self)
            act.triggered.connect(slot)
            bar.addAction(act)

        bar.addSeparator()
        self.integrator_cb = QComboBox()
        self.integrator_cb.addItems(["RK45", "RK23", "DOP853", "LSODA"])
        bar.addWidget(QLabel("Integrator:"))
        bar.addWidget(self.integrator_cb)

        btn_clear = QPushButton("Clear")
        btn_clear.clicked.connect(self._clear_trajectories)
        bar.addWidget(btn_clear)

        bar.addSeparator()
        self.act_vector = QAction("Vector field", self)
        self.act_vector.setCheckable(True)
        self.act_vector.triggered.connect(self._toggle_vector_field)
        bar.addAction(self.act_vector)

        bar.addSeparator()
        self.act_max_param = QAction("Max Param", self)
        self.act_max_param.setCheckable(True)
        self.act_max_param.triggered.connect(self._toggle_max_param)
        bar.addAction(self.act_max_param)

        self.act_max_phase = QAction("Max Phase", self)
        self.act_max_phase.setCheckable(True)
        self.act_max_phase.triggered.connect(self._toggle_max_phase)
        bar.addAction(self.act_max_phase)

    def _compile_system(self):
        x, y, m1, m2 = sp.symbols("x y mu1 mu2")
        f_sym = sp.sympify(self.f_expr)
        g_sym = sp.sympify(self.g_expr)
        self.f_lam = sp.lambdify((x, y, m1, m2), f_sym, "numpy")
        self.g_lam = sp.lambdify((x, y, m1, m2), g_sym, "numpy")
        J = sp.Matrix([f_sym, g_sym]).jacobian([x, y])
        self.J11 = sp.lambdify((x, y, m1, m2), J[0, 0], "numpy")
        self.J12 = sp.lambdify((x, y, m1, m2), J[0, 1], "numpy")
        self.J21 = sp.lambdify((x, y, m1, m2), J[1, 0], "numpy")
        self.J22 = sp.lambdify((x, y, m1, m2), J[1, 1], "numpy")
        self.rhs_func: Callable = lambda t, s, mu1, mu2: np.array([
            self.f_lam(s[0], s[1], mu1, mu2),
            self.g_lam(s[0], s[1], mu1, mu2),
        ])

    def _configure_param_axes(self):
        ax = self.param_canvas.ax; ax.clear()
        ax.set_title("Parameter plane (μ1, μ2)")
        ax.set_xlabel("μ1"); ax.set_ylabel("μ2")
        ax.set_xlim(self.range["mu1_min"], self.range["mu1_max"])
        ax.set_ylim(self.range["mu2_min"], self.range["mu2_max"])
        self.param_canvas.canvas.draw_idle()

    def _configure_phase_axes(self):
        ax = self.phase_canvas.ax; ax.clear()
        ax.set_title("Phase plane (x, y)")
        ax.set_xlabel("x"); ax.set_ylabel("y")
        ax.set_xlim(self.phase_range["x_min"], self.phase_range["x_max"])
        ax.set_ylim(self.phase_range["y_min"], self.phase_range["y_max"])
        self.phase_canvas.canvas.draw_idle()

    def _connect_events(self):
        self.param_canvas.canvas.mpl_connect(
            "button_press_event", self._on_param_click)
        self.phase_canvas.canvas.mpl_connect(
            "button_press_event", self._on_phase_click)

    def _on_param_click(self, event):
        if not (event.inaxes and event.dblclick):
            return
        mu1, mu2 = event.xdata, event.ydata
        self.current_mu = (mu1, mu2)
        if self.param_marker:
            self.param_marker.remove()
        self.param_marker, = self.param_canvas.ax.plot(
            mu1, mu2, "xr", ms=8)
        self.param_canvas.canvas.draw_idle()

        self._draw_nullclines(mu1, mu2)
        if self.show_field:
            self._draw_vector_field()
        self.statusBar().showMessage(
            f"Selected μ=({mu1:.3g},{mu2:.3g})")

    def _draw_nullclines(self, mu1, mu2):
        # clear old nullclines
        for art in self.nullcline_artists:
            if hasattr(art, "collections"):
                for c in art.collections:
                    c.remove()
            else:
                try:
                    art.remove()
                except:
                    pass
        self.nullcline_artists.clear()

        # clear old equilibrium points/texts
        for pt in self.nullcline_points:
            try:
                pt.remove()
            except NotImplementedError:
                pt.set_visible(False)
        self.nullcline_points.clear()

        # clear table
        self.eq_table.setRowCount(0)

        ax = self.phase_canvas.ax
        xmin, xmax = ax.get_xlim(); ymin, ymax = ax.get_ylim()
        xx, yy = np.meshgrid(
            np.linspace(xmin, xmax, 300),
            np.linspace(ymin, ymax, 300)
        )
        cf = ax.contour(xx, yy, self.f_lam(xx, yy, mu1, mu2),
                        levels=[0], colors="blue",
                        linestyles="--", linewidths=1.2)
        cg = ax.contour(xx, yy, self.g_lam(xx, yy, mu1, mu2),
                        levels=[0], colors="green",
                        linestyles="-", linewidths=1.2)
        self.nullcline_artists += [cf, cg]

        # find & classify equilibria
        x, y, m1, m2 = sp.symbols("x y mu1 mu2")
        f0 = sp.sympify(self.f_expr).subs({m1: mu1, m2: mu2})
        g0 = sp.sympify(self.g_expr).subs({m1: mu1, m2: mu2})
        sols = sp.solve([f0, g0], [x, y], dict=True)
        for sol in sols:
            xr, yr = sol[x], sol[y]
            if xr.is_real and yr.is_real:
                xf, yf = float(xr), float(yr)
                if not (xmin <= xf <= xmax and ymin <= yf <= ymax):
                    continue
                a = self.J11(xf, yf, mu1, mu2)
                b = self.J12(xf, yf, mu1, mu2)
                c = self.J21(xf, yf, mu1, mu2)
                d = self.J22(xf, yf, mu1, mu2)
                ev = np.linalg.eigvals([[a, b], [c, d]])
                re, im = np.real(ev), np.imag(ev)

                if abs(re[0]) < 1e-6 and abs(re[1]) < 1e-6 and np.any(im != 0):
                    typ, color = "center", "green"
                elif np.any(im != 0):
                    if np.all(re < 0):
                        typ, color = "stable focus", "purple"
                    else:
                        typ, color = "unstable focus", "magenta"
                elif re[0] * re[1] < 0:
                    typ, color = "saddle", "red"
                else:
                    if np.all(re < 0):
                        typ, color = "stable node", "blue"
                    else:
                        typ, color = "unstable node", "cyan"

                pt, = ax.plot(xf, yf, "o", color=color, ms=8)
                txt = ax.text(xf, yf, typ, color=color,
                              fontsize="small", va="bottom", ha="right")
                self.nullcline_points += [pt, txt]

                # add to table
                row = self.eq_table.rowCount()
                self.eq_table.insertRow(row)
                for col, val in enumerate([xf, yf, typ, f"{ev[0]:.3g}", f"{ev[1]:.3g}"]):
                    self.eq_table.setItem(row, col, QTableWidgetItem(str(val)))

        handles = [
            Line2D([], [], marker="o", color="red", linestyle="", label="saddle"),
            Line2D([], [], marker="o", color="blue", linestyle="", label="stable node"),
            Line2D([], [], marker="o", color="cyan", linestyle="", label="unstable node"),
            Line2D([], [], marker="o", color="purple", linestyle="", label="stable focus"),
            Line2D([], [], marker="o", color="magenta", linestyle="", label="unstable focus"),
            Line2D([], [], marker="o", color="green", linestyle="", label="center"),
        ]
        ax.legend(handles=handles, loc="upper right", fontsize="small")
        self.phase_canvas.canvas.draw_idle()

    def _toggle_vector_field(self, checked: bool):
        self.show_field = checked
        if checked:
            self._draw_vector_field()
        else:
            for art in self.field_artists:
                art.remove()
            self.field_artists.clear()
            self.phase_canvas.canvas.draw_idle()

    def _draw_vector_field(self):
        for art in self.field_artists:
            art.remove()
        self.field_artists.clear()

        if self.current_mu is None:
            return
        mu1, mu2 = self.current_mu
        ax = self.phase_canvas.ax
        xmin, xmax = ax.get_xlim(); ymin, ymax = ax.get_ylim()
        nx, ny = 20, 20
        xs = np.linspace(xmin, xmax, nx)
        ys = np.linspace(ymin, ymax, ny)
        XX, YY = np.meshgrid(xs, ys)
        U = self.f_lam(XX, YY, mu1, mu2)
        V = self.g_lam(XX, YY, mu1, mu2)
        M = np.hypot(U, V)
        M[M == 0] = 1
        U2, V2 = U / M, V / M
        Q = ax.quiver(XX, YY, U2, V2, angles="xy", pivot="mid", alpha=0.6)
        self.field_artists.append(Q)
        self.phase_canvas.canvas.draw_idle()

    def _on_phase_click(self, event):
        if not (event.inaxes and event.dblclick) or self.current_mu is None:
            return
        x0, y0 = event.xdata, event.ydata
        mu1, mu2 = self.current_mu
        method = self.integrator_cb.currentText()
        T, N = 20.0, 400
        sol_f = solve_ivp(lambda t, y: self.rhs_func(t, y, mu1, mu2),
                          (0, T), [x0, y0],
                          t_eval=np.linspace(0, T, N//2), method=method)
        sol_b = solve_ivp(lambda t, y: self.rhs_func(t, y, mu1, mu2),
                          (0, -T), [x0, y0],
                          t_eval=np.linspace(0, -T, N//2), method=method)
        xb, yb = sol_b.y[0][::-1][:-1], sol_b.y[1][::-1][:-1]
        xf, yf = sol_f.y
        xfull = np.concatenate([xb, xf]); yfull = np.concatenate([yb, yf])
        line, = self.phase_canvas.ax.plot(xfull, yfull,
                                          color=next(self.color_cycle))
        self.traj_lines.append(line)
        self.phase_canvas.canvas.draw_idle()
        self.statusBar().showMessage(
            f"μ=({mu1:.3g},{mu2:.3g}); IC=({x0:.3g},{y0:.3g}); traj#{len(self.traj_lines)}"
        )

    def _clear_trajectories(self):
        for ln in self.traj_lines:
            ln.remove()
        self.traj_lines.clear()
        self.phase_canvas.canvas.draw_idle()
        self.statusBar().showMessage("Trajectories cleared")

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
                self.statusBar().showMessage("System updated")
            except Exception as e:
                QMessageBox.critical(self, "Error", str(e))

    def _edit_range(self):
        dlg = RangeDialog(self.range, self)
        if dlg.exec():
            self.range = dlg.values()
            self._configure_param_axes()
            self.statusBar().showMessage("Parameter range updated")

    def _edit_phase_range(self):
        dlg = PhaseRangeDialog(self.phase_range, self)
        if dlg.exec():
            self.phase_range = dlg.values()
            self._configure_phase_axes()
            self.statusBar().showMessage("Phase range updated")

    def _toggle_max_param(self, checked: bool):
        if checked:
            self.phase_canvas.hide()
            self.splitter.setStretchFactor(0, 1)
            self.splitter.setStretchFactor(1, 0)
            self.act_max_phase.setChecked(False)
        else:
            self.phase_canvas.show()
            self.splitter.setStretchFactor(0, 1)
            self.splitter.setStretchFactor(1, 1)

    def _toggle_max_phase(self, checked: bool):
        if checked:
            self.param_canvas.hide()
            self.splitter.setStretchFactor(0, 0)
            self.splitter.setStretchFactor(1, 1)
            self.act_max_param.setChecked(False)
        else:
            self.param_canvas.show()
            self.splitter.setStretchFactor(0, 1)
            self.splitter.setStretchFactor(1, 1)


def main() -> None:
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
