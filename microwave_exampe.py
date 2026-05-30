#!/usr/bin/env python3
# =====================================================
# py-microwave: Coupling Matrix Tuner GUI
# =====================================================
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import json
from pathlib import Path
import traceback

# ====================== IMPORTS ======================
try:
    import mwave as mw
    from filter import fit_to_M, MofChebychev, RespM2
except ImportError as e:
    print("ERROR: Missing dependencies!")
    print("Please ensure 'mwave' and 'filter.py' are available.")
    raise

F0_DEFAULT = 2.0e9
BW_DEFAULT = 0.10
ORDER = 5


class CouplingMatrixTunerGUI:
    def __init__(self):
        self.f0 = F0_DEFAULT
        self.bw = BW_DEFAULT
        self.order = ORDER
      
        # Load measured data
        try:
            self.flist, self.S_original = mw.load_touchstone(
                "fitted_cheby_5th_10pct.s2p"
            )
        except Exception as e:
            messagebox.showerror("File Error",
                f"Could not load fitted_cheby_5th_10pct.s2p\n\n{str(e)}")
            raise

        # Fit coupling matrix
        try:
            self.Mfit, self.Qfit, self.phase_in, self.phase_out, _ = fit_to_M(
                flist=self.flist,
                Smeas=self.S_original,
                Minitial=MofChebychev(self.order, 0.0),
                f0=self.f0,
                BW=self.bw,
                Qinitial=1e9,
                NRNlist=[],
                Mindiceslist=[
                    (1,0),(1,2),(2,3),(3,4),(4,5),(5,6),
                    (1,1),(2,2),(3,3),(4,4),(5,5)
                ],
                is_symmetric=False,
                deci=2,
                fstart=1.7e9,
                fstop=2.3e9,
                maxerr=200
            )
        except Exception as e:
            messagebox.showerror("Fitting Error", f"fit_to_M failed:\n{str(e)}")
            raise

        self.phase_in = float(self.phase_in)
        self.phase_out = float(self.phase_out)
        self.M_tuned = self.Mfit.copy().astype(float)
        self.history: list[np.ndarray] = []
        self.frange = np.arange(1.6e9, 2.4e9, 0.5e6)

        # Marker settings
        self.show_markers = True
        self.f_low = self.f0 - self.f0 * self.bw / 2
        self.f_high = self.f0 + self.f0 * self.bw / 2
        self.f_center = self.f0
       
        self.setup_gui()

    # =====================================================
    # GUI SETUP (unchanged except marker frame)
    # =====================================================
    def setup_gui(self):
        self.root = tk.Tk()
        self.root.title("py-microwave — Coupling Matrix Tuner")
        self.root.state('zoomed')
        self.root.minsize(1200, 800)

        main_pane = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_pane.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        # Left Panel - Matrix
        left_frame = ttk.LabelFrame(main_pane, text="Coupling Matrix - Double-click to edit", padding=10)
        main_pane.add(left_frame, weight=1)

        cols = ['i'] + [str(i) for i in range(self.order + 2)]
        self.tree = ttk.Treeview(left_frame, columns=cols, show='headings', height=18)
      
        self.tree.heading('i', text='Row')
        for i in range(self.order + 2):
            if i == 0:
                header_text = "Source"
            elif i == self.order + 1:
                header_text = "Load"
            else:
                header_text = f"R{i}"
            self.tree.heading(str(i), text=header_text)
            self.tree.column(str(i), width=92, anchor='center')

        self.tree.pack(fill=tk.BOTH, expand=True)
        self.tree.bind("<Double-1>", self.on_double_click)

        # Buttons
        btn_frame = ttk.Frame(left_frame)
        btn_frame.pack(fill=tk.X, pady=10)
        ttk.Button(btn_frame, text="Undo", command=self.undo).pack(side=tk.LEFT, padx=4)
        ttk.Button(btn_frame, text="Reset", command=self.reset).pack(side=tk.LEFT, padx=4)
        ttk.Button(btn_frame, text="Perturb", command=self.perturb).pack(side=tk.LEFT, padx=4)
        ttk.Button(btn_frame, text="Force Symmetry", command=self.force_symmetry).pack(side=tk.LEFT, padx=4)

        # Marker Controls
        marker_frame = ttk.LabelFrame(left_frame, text="Markers", padding=8)
        marker_frame.pack(fill=tk.X, pady=8)
        ttk.Button(marker_frame, text="Toggle Markers", command=self.toggle_markers).pack(side=tk.LEFT, padx=4)
        ttk.Button(marker_frame, text="Set Markers", command=self.configure_markers).pack(side=tk.LEFT, padx=4)

        ttk.Button(btn_frame, text="Save", command=self.save_matrix).pack(side=tk.RIGHT, padx=4)
        ttk.Button(btn_frame, text="Load", command=self.load_matrix).pack(side=tk.RIGHT, padx=4)

        # Right Panel - Plot
        right_frame = ttk.LabelFrame(main_pane, text="Filter Response", padding=8)
        main_pane.add(right_frame, weight=3)

        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
        self.fig.tight_layout()
        self.canvas = FigureCanvasTkAgg(self.fig, right_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        NavigationToolbar2Tk(self.canvas, right_frame)

        # Status bars
        self.status = tk.StringVar(value="Ready")
        self.debug_label = tk.StringVar(value="")
        ttk.Label(self.root, textvariable=self.status, relief=tk.SUNKEN, anchor=tk.W).pack(
            fill=tk.X, side=tk.BOTTOM, padx=6, pady=2)
        ttk.Label(self.root, textvariable=self.debug_label, relief=tk.SUNKEN,
                 anchor=tk.W, foreground="red").pack(fill=tk.X, side=tk.BOTTOM, padx=6)

        self.update_matrix_table()
        self.create_plot()
        self.update_plot()

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.mainloop()

    # =====================================================
    # PLOT SETUP
    # =====================================================
    def create_plot(self):
        self.ax1.clear()
        self.ax2.clear()
      
        self.ax1.plot(self.flist/1e9, 20*np.log10(np.abs(self.S_original[:,0,0])),
                     'b-', lw=2, label='Measured')
        self.ax2.plot(self.flist/1e9, 20*np.log10(np.abs(self.S_original[:,0,1])),
                     'b-', lw=2, label='Measured')

        self.s11_line, = self.ax1.plot([], [], 'r-', lw=2.3, label='Tuned')
        self.s21_line, = self.ax2.plot([], [], 'r-', lw=2.3, label='Tuned')

        self.add_markers()

        for ax in (self.ax1, self.ax2):
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper right')

        self.ax1.set_ylabel('S11 (dB)')
        self.ax2.set_ylabel('S21 (dB)')
        self.ax2.set_xlabel('Frequency (GHz)')
        self.fig.suptitle('5th Order Chebyshev Bandpass Filter Tuner', fontsize=14)

    # =====================================================
    # MARKERS - Mrk 1, Mrk 2, Mrk 3 with dB values
    # =====================================================
    def add_markers(self):
        if not self.show_markers:
            return

        # Get current tuned response
        try:
            S_tuned = RespM2(self.M_tuned.copy(), self.f0, self.bw, self.Qfit, self.frange)
            beta = np.pi * (self.frange - self.f0) / (self.bw * self.f0)
            S_tuned[:,0,0] *= np.exp(2j * self.phase_in * beta)
            S_tuned[:,0,1] *= np.exp(1j * (self.phase_in + self.phase_out) * beta)
            s11_db = 20 * np.log10(np.maximum(np.abs(S_tuned[:,0,0]), 1e-15))
            s21_db = 20 * np.log10(np.maximum(np.abs(S_tuned[:,0,1]), 1e-15))
        except:
            s11_db = np.zeros_like(self.frange)
            s21_db = np.zeros_like(self.frange)

        f_low_g = self.f_low / 1e9
        f_high_g = self.f_high / 1e9
        f_c_g = self.f_center / 1e9

        for ax_idx, ax in enumerate([self.ax1, self.ax2]):
            db_array = s11_db if ax_idx == 0 else s21_db
            param = "S11" if ax_idx == 0 else "S21"

            idx_low = np.argmin(np.abs(self.frange - self.f_low))
            idx_high = np.argmin(np.abs(self.frange - self.f_high))
            idx_c = np.argmin(np.abs(self.frange - self.f_center))

            # Mrk 1 - Low Edge
            ax.axvline(f_low_g, color='black', ls='--', lw=1.5, alpha=0.85)
            ax.text(f_low_g, ax.get_ylim()[1]*0.96,
                    f'Mrk 1\n{f_low_g:.3f} GHz\n{param} = {db_array[idx_low]:.2f} dB',
                    color='black', fontsize=9.5, ha='right', va='top', fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))

            # Mrk 2 - High Edge
            ax.axvline(f_high_g, color='black', ls='--', lw=1.5, alpha=0.85)
            ax.text(f_high_g, ax.get_ylim()[1]*0.96,
                    f'Mrk 2\n{f_high_g:.3f} GHz\n{param} = {db_array[idx_high]:.2f} dB',
                    color='black', fontsize=9.5, ha='left', va='top', fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))

            # Mrk 3 - Center
            ax.axvline(f_c_g, color='black', ls='--', lw=2.0, alpha=0.9)
            ax.text(f_c_g, ax.get_ylim()[1]*0.78,
                    f'Mrk 3\n{f_c_g:.3f} GHz\n{param} = {db_array[idx_c]:.2f} dB',
                    color='black', fontsize=10, ha='center', va='top', fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", alpha=0.8))

    def toggle_markers(self):
        self.show_markers = not self.show_markers
        self.create_plot()
        self.update_plot()
        status = "Markers ON" if self.show_markers else "Markers OFF"
        self.status.set(status)

    def configure_markers(self):
        popup = tk.Toplevel(self.root)
        popup.title("Set Markers")
        popup.geometry("380x280")
        popup.grab_set()

        tk.Label(popup, text="Marker Frequencies (GHz)", font=("Arial", 11, "bold")).pack(pady=10)

        frame = ttk.Frame(popup)
        frame.pack(pady=10, padx=20)

        ttk.Label(frame, text="Mrk 1 - Lower Edge:").grid(row=0, column=0, sticky='w', pady=4)
        f_low_var = tk.DoubleVar(value=round(self.f_low/1e9, 4))
        tk.Entry(frame, textvariable=f_low_var, width=12).grid(row=0, column=1, padx=8)

        ttk.Label(frame, text="Mrk 2 - Upper Edge:").grid(row=1, column=0, sticky='w', pady=4)
        f_high_var = tk.DoubleVar(value=round(self.f_high/1e9, 4))
        tk.Entry(frame, textvariable=f_high_var, width=12).grid(row=1, column=1, padx=8)

        ttk.Label(frame, text="Mrk 3 - Center Fc:").grid(row=2, column=0, sticky='w', pady=4)
        f_c_var = tk.DoubleVar(value=round(self.f_center/1e9, 4))
        tk.Entry(frame, textvariable=f_c_var, width=12).grid(row=2, column=1, padx=8)

        def apply():
            try:
                self.f_low = f_low_var.get() * 1e9
                self.f_high = f_high_var.get() * 1e9
                self.f_center = f_c_var.get() * 1e9
                self.create_plot()
                self.update_plot()
                self.status.set("Markers updated")
                popup.destroy()
            except ValueError:
                messagebox.showerror("Input Error", "Please enter valid numbers")

        ttk.Button(popup, text="Apply", command=apply).pack(pady=15)
        ttk.Button(popup, text="Cancel", command=popup.destroy).pack()

    # =====================================================
    # Rest of the methods (Matrix, Update Plot, Slider, etc.)
    # =====================================================
    def update_matrix_table(self):
        for item in self.tree.get_children():
            self.tree.delete(item)
        n = self.M_tuned.shape[0]
        for i in range(n):
            row_values = [str(i)] + [f"{self.M_tuned[i,j]:.4f}" for j in range(n)]
            self.tree.insert('', 'end', values=row_values)

    def update_plot(self):
        self.debug_label.set("Updating plot...")
        try:
            M_for_calc = self.M_tuned.copy()
            S_tuned = RespM2(M_for_calc, self.f0, self.bw, self.Qfit, self.frange)
            beta = np.pi * (self.frange - self.f0) / (self.bw * self.f0)
          
            S_tuned[:,0,0] *= np.exp(2j * self.phase_in * beta)
            S_tuned[:,0,1] *= np.exp(1j * (self.phase_in + self.phase_out) * beta)
            S_tuned[:,1,0] = S_tuned[:,0,1].copy()
            S_tuned[:,1,1] *= np.exp(2j * self.phase_out * beta)
           
            eps = 1e-15
            s11_db = 20 * np.log10(np.maximum(np.abs(S_tuned[:,0,0]), eps))
            s21_db = 20 * np.log10(np.maximum(np.abs(S_tuned[:,0,1]), eps))
           
            self.s11_line.set_data(self.frange/1e9, s11_db)
            self.s21_line.set_data(self.frange/1e9, s21_db)
           
            for ax in (self.ax1, self.ax2):
                ax.relim()
                ax.autoscale_view()
            self.canvas.draw_idle()
            self.status.set("Plot updated")
            self.debug_label.set("")
        except Exception as e:
            error_msg = f"Plot Error: {e}"
            print(error_msg)
            traceback.print_exc()
            self.status.set(error_msg)
            self.debug_label.set(error_msg)

    def save_history(self):
        self.history.append(self.M_tuned.copy())
        if len(self.history) > 60:
            self.history.pop(0)

    def on_double_click(self, event):
        item = self.tree.identify_row(event.y)
        column = self.tree.identify_column(event.x)
        if not item or column == '#1':
            return
        try:
            col_idx = int(column[1:]) - 2
            row_idx = self.tree.index(item)
        except:
            return
        if col_idx < 0 or col_idx >= self.M_tuned.shape[0]:
            return
        current_value = float(self.M_tuned[row_idx, col_idx])
        self.show_slider_editor(row_idx, col_idx, current_value)

    def show_slider_editor(self, row_idx: int, col_idx: int, initial_value: float):
        popup = tk.Toplevel(self.root)
        popup.grab_set()
        if row_idx == col_idx:
            title = f"Editing Resonator R{row_idx}"
            value_prefix = f"R{row_idx}"
        else:
            title = f"Editing Coupling M{row_idx},{col_idx}"
            value_prefix = f"M{row_idx},{col_idx}"
        popup.title(title)
        popup.geometry("500x300")
        original_value = initial_value
        self.save_history()

        tk.Label(popup, text=title, font=("Arial", 12, "bold")).pack(pady=10)
        value_label = tk.Label(popup, text=f"{value_prefix} = {initial_value:.5f}",
                             font=("Consolas", 14, "bold"))
        value_label.pack(pady=8)

        def update_live(val):
            try:
                new_val = float(val)
                self.M_tuned[row_idx, col_idx] = new_val
                if row_idx != col_idx:
                    self.M_tuned[col_idx, row_idx] = new_val
                display_text = f"{value_prefix} = {new_val:.5f}"
                value_label.config(text=display_text)
                self.update_matrix_table()
                self.update_plot()
            except:
                pass

        slider = ttk.Scale(
            popup, from_=-1.8, to=1.8, orient=tk.HORIZONTAL,
            length=420, variable=tk.DoubleVar(value=initial_value), command=update_live
        )
        slider.pack(pady=12, padx=30)

        btn_frame = ttk.Frame(popup)
        btn_frame.pack(pady=8)
        for step in [-0.1, -0.01, -0.001, 0.001, 0.01, 0.1]:
            def make_step(s=step):
                def action():
                    new_val = slider.get() + s
                    slider.set(new_val)
                    update_live(new_val)
                return action
            ttk.Button(btn_frame, text=f"{step:+.3f}", width=8, command=make_step()).pack(side=tk.LEFT, padx=3)

        bottom = ttk.Frame(popup)
        bottom.pack(pady=15)
        def ok():
            self.status.set(f"✓ {value_prefix} = {slider.get():.5f}")
            popup.destroy()
        def cancel():
            self.M_tuned[row_idx, col_idx] = original_value
            if row_idx != col_idx:
                self.M_tuned[col_idx, row_idx] = original_value
            self.update_matrix_table()
            self.update_plot()
            self.status.set("Edit cancelled")
            popup.destroy()

        ttk.Button(bottom, text="OK", command=ok).pack(side=tk.LEFT, padx=20)
        ttk.Button(bottom, text="Cancel", command=cancel).pack(side=tk.LEFT, padx=20)

        popup.bind("<Return>", lambda e: ok())
        popup.bind("<Escape>", lambda e: cancel())

    def undo(self):
        if self.history:
            self.M_tuned = self.history.pop()
            self.update_matrix_table()
            self.update_plot()
            self.status.set("Undo successful")
        else:
            self.status.set("Nothing to undo")

    def reset(self):
        if messagebox.askyesno("Reset", "Reset to original fitted matrix?"):
            self.M_tuned = self.Mfit.copy().astype(float)
            self.history.clear()
            self.update_matrix_table()
            self.update_plot()
            self.status.set("Reset completed")

    def perturb(self):
        if messagebox.askyesno("Perturb", "Apply small random perturbation?"):
            self.save_history()
            noise = np.random.normal(0, 0.008, self.M_tuned.shape)
            noise = (noise + noise.T) / 2
            np.fill_diagonal(noise, 0)
            self.M_tuned += noise
            self.update_matrix_table()
            self.update_plot()
            self.status.set("Perturbation applied")

    def force_symmetry(self):
        if messagebox.askyesno("Force Symmetry", "Make matrix symmetric?"):
            self.save_history()
            n = self.M_tuned.shape[0]
            for i in range(n):
                for j in range(i + 1, n):
                    avg = (self.M_tuned[i,j] + self.M_tuned[j,i]) / 2.0
                    self.M_tuned[i,j] = avg
                    self.M_tuned[j,i] = avg
            self.update_matrix_table()
            self.update_plot()
            self.status.set("Matrix symmetrized")

    def save_matrix(self):
        fn = filedialog.asksaveasfilename(defaultextension=".json")
        if fn:
            data = {
                "M": self.M_tuned.tolist(),
                "f0": self.f0,
                "BW": self.bw,
                "f_low": self.f_low,
                "f_high": self.f_high,
                "f_center": self.f_center
            }
            Path(fn).write_text(json.dumps(data, indent=2))
            self.status.set(f"Saved to {Path(fn).name}")

    def load_matrix(self):
        fn = filedialog.askopenfilename(filetypes=[("JSON", "*.json")])
        if fn:
            try:
                data = json.loads(Path(fn).read_text())
                self.M_tuned = np.array(data["M"], dtype=float)
                if "f_low" in data:
                    self.f_low = data["f_low"]
                    self.f_high = data["f_high"]
                    self.f_center = data["f_center"]
                self.update_matrix_table()
                self.create_plot()
                self.update_plot()
                self.status.set(f"Loaded {Path(fn).name}")
            except Exception as e:
                messagebox.showerror("Load Error", str(e))

    def on_close(self):
        if messagebox.askokcancel("Exit", "Close the tuner?"):
            plt.close('all')
            self.root.destroy()


# =========================================================
if __name__ == "__main__":
    print("Starting py-microwave Coupling Matrix Tuner...")
    try:
        CouplingMatrixTunerGUI()
    except Exception as e:
        print("Fatal error:", e)
        traceback.print_exc()