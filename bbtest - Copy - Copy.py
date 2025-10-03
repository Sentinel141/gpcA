<<<<<<< Updated upstream
import os
import re
import json
import math
import tkinter as tk
from tkinter import filedialog, messagebox, colorchooser
import tkinter.ttk as ttk
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.font_manager import FontProperties
from scipy.ndimage import gaussian_filter1d
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image, ImageTk

# RDKit support for SMILES rendering (optional)
try:
    from rdkit import Chem
    from rdkit.Chem import Draw
    RDKit_AVAILABLE = True
except ImportError:
    RDKit_AVAILABLE = False


class DraggableMolecule:
    def __init__(self, parent_canvas, image=None, widget=None, label=""):
        self.parent = parent_canvas
        self.label = label
        if image is not None:
            self.id = parent_canvas.create_image(10, 10, anchor='nw', image=image)
            self.image = image
        else:
            self.id = parent_canvas.create_window(10, 10, anchor='nw', window=widget)
            self.widget = widget
        self._drag_data = {"x": 0, "y": 0}
        parent_canvas.tag_bind(self.id, "<ButtonPress-1>", self.on_press)
        parent_canvas.tag_bind(self.id, "<ButtonRelease-1>", self.on_release)
        parent_canvas.tag_bind(self.id, "<B1-Motion>", self.on_motion)

    def on_press(self, event):
        self._drag_data["x"] = event.x
        self._drag_data["y"] = event.y

    def on_motion(self, event):
        dx = event.x - self._drag_data["x"]
        dy = event.y - self._drag_data["y"]
        self.parent.move(self.id, dx, dy)
        self._drag_data["x"] = event.x
        self._drag_data["y"] = event.y

    def on_release(self, event):
        pass


def smooth_series(x, y, sigma):
    """Smooth *y* samples measured at coordinates *x* using a Gaussian kernel."""
    if x is None or y is None:
        return y

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if x.size == 0 or y.size == 0 or x.size != y.size:
        return y

    if sigma is None:
        return y

    try:
        sigma = float(sigma)
    except (TypeError, ValueError):
        return y

    if sigma <= 0 or not np.isfinite(sigma):
        return y

    diffs = np.diff(x)
    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    if diffs.size == 0:
        return y

    step = float(np.median(diffs))
    if not np.isfinite(step) or step <= 0:
        return y

    # Treat the user-provided ``sigma`` control as the desired full width at
    # half maximum (FWHM) instead of the raw Gaussian sigma to keep the
    # smoothing gentler, especially for finely sampled data sets.  Convert the
    # requested FWHM to the standard deviation expected by ``gaussian_filter1d``
    # and normalise by the median spacing so wildly irregular step sizes do not
    # explode the kernel width.  The conversion constant comes from
    # ``sigma = fwhm / (2 * sqrt(2 * ln(2)))``.
    fwhm_to_sigma = 0.5 / math.sqrt(2.0 * math.log(2.0))
    sigma_samples = (sigma * fwhm_to_sigma) / step
    if sigma_samples <= 0 or not np.isfinite(sigma_samples):
        return y

    try:
        return gaussian_filter1d(y, sigma=sigma_samples, mode='nearest')
    except Exception:
        # If SciPy cannot honour the requested sigma (for example, extremely
        # large values triggering numerical issues), fall back to the original
        # data instead of crashing the application.
        return y


def load_and_smooth(path, detector, sigma=1):
    """
    Reads GPC text/CSV with at least 4 columns:
    Time, RI, Light Scattering, Viscometry.
    Accepts space/tab or CSV. Smooths chosen detector.
    """
    try:
        try:
            df = pd.read_csv(path, delim_whitespace=True, header=None)
        except Exception:
            df = pd.read_csv(path, header=None)
    except Exception:
        return None, None

    if df.shape[1] < 4:
        return None, None

    df = df.iloc[:, :4]
    df.columns = ['Time', 'RI', 'Light Scattering', 'Viscometry']
    ser = pd.to_numeric(df[detector], errors='coerce')
    times = pd.to_numeric(df['Time'], errors='coerce')
    valid = ser.notna() & times.notna()
    if not valid.any():
        return None, None

    times = times[valid].to_numpy(dtype=float)
    values = ser[valid].to_numpy(dtype=float)

    order = np.argsort(times)
    times = times[order]
    values = values[order]

    smoothed = smooth_series(times, values, sigma)
    return times, smoothed


def load_mass_distribution(path):
    """
    Parses the 'Molar mass' block exported by ChromPilot/WINGPC.
    Returns (M, signal, integral_pct) as numpy arrays or (None, None, None) if not found.
    """
    import re
    import numpy as np

    try:
        with open(path, 'r', encoding='latin-1') as f:
            lines = f.read().splitlines()
    except Exception:
        return None, None, None

    idx = None
    for i, line in enumerate(lines):
        if line.strip().lower().startswith('molar mass'):
            idx = i
    if idx is None:
        return None, None, None

    j = idx + 1
    while j < len(lines) and not lines[j].strip():
        j += 1

    Ms, sig, integ = [], [], []
    for k in range(j, len(lines)):
        parts = re.split(r'[\t,;]+', lines[k].strip())
        if len(parts) < 3:
            break
        try:
            m = float(parts[0].replace(' ', ''))
            y1 = float(parts[1])
            y2 = float(parts[2])
        except ValueError:
            break
        Ms.append(m); sig.append(y1); integ.append(y2)

    if not Ms:
        return None, None, None
    return np.array(Ms, dtype=float), np.array(sig, dtype=float), np.array(integ, dtype=float)


def compute_weight_fraction(masses, signal):
    """Return masses and the weight fraction w(log M) for finite, positive masses."""
    masses = np.asarray(masses, dtype=float)
    signal = np.asarray(signal, dtype=float)

    mask = np.isfinite(masses) & np.isfinite(signal) & (masses > 0)
    if not mask.any():
        return np.array([]), np.array([])

    masses = masses[mask]
    signal = np.clip(signal[mask], 0.0, None)

    order = np.argsort(masses)
    masses = masses[order]
    signal = signal[order]

    if masses.size < 2:
        if signal.size:
            peak = signal.max()
            if peak > 0:
                signal = signal / peak
        return masses, signal

    log_m = np.log10(masses)
    area = np.trapz(signal, log_m)
    if area > 0:
        weights = signal / area
    else:
        peak = signal.max()
        weights = signal / peak if peak > 0 else signal
    return masses, weights


class GPCApp:
    def __init__(self, root):
        self.root = root
        root.title("GPC Analyzer Live Viewer")

        screen_w = root.winfo_screenwidth()
        screen_h = root.winfo_screenheight()
        w = min(1200, int(screen_w * 0.9))
        h = min(800, int(screen_h * 0.9))
        x = (screen_w - w) // 2
        y = (screen_h - h) // 2
        root.geometry(f"{w}x{h}+{x}+{y}")
        root.minsize(900, 600)

        # ---------- state ----------
        self.file_vars = {}
        self.file_labels = {}
        self.file_colors = {}
        self.legend_order = []

        self.gradient_var = tk.BooleanVar(value=False)
        self.inline_var = tk.BooleanVar(value=False)
        self.hide_x_var = tk.BooleanVar(value=False)
        self.hide_y_var = tk.BooleanVar(value=False)
        self.auto_legend = tk.BooleanVar(value=True)
        self.show_title = tk.BooleanVar(value=False)
        self.custom_title_var = tk.StringVar(value="")
        self.normalize_var = tk.BooleanVar(value=True)
        self.xlabel_var = tk.StringVar(value="")
        self.ylabel_var = tk.StringVar(value="")
        self.axis_fontsize_var = tk.IntVar(value=10)
        self.font_family_var = tk.StringVar(value="DejaVu Sans")
        self.gradient_start_color = "#000000"
        self.gradient_end_color = "#FFFFFF"

        # Baseline controls
        self.extend_baseline_var = tk.BooleanVar(value=True)
        self.baseline_len_var = tk.DoubleVar(value=2.0)
        self.baseline_mode = tk.StringVar(value="X-limits")

        # Taper controls
        self.taper_var = tk.BooleanVar(value=True)
        self.taper_len_var = tk.DoubleVar(value=0.3)

        # Smoothing (sigma)
        self.sigma_var = tk.DoubleVar(value=1.0)

        # NEW: math vs unicode scripts
        self.use_mathtext = tk.BooleanVar(value=True)  # OFF -> Unicode subs/sups so normal fonts (e.g., Calibri) apply

        # undo/redo history (minimal)
        self._history = []
        self._hist_index = -1

        # ---------- shortcuts ----------
        root.bind_all("<Control-p>", lambda e: self.update_plot())
        root.bind_all("<Control-s>", lambda e: self.save_plot())
        root.bind_all("<Control-o>", lambda e: self.browse_folder())
        root.bind_all("<Control-z>", lambda e: self.undo())
        root.bind_all("<Control-y>", lambda e: self.redo())

        self.show_title.trace_add("write", lambda *args: self._push_history())
        self.custom_title_var.trace_add("write", lambda *args: self._push_history())
        self.auto_legend.trace_add("write", lambda *args: self._push_history())

        # ---------- top scroller ----------
        container = ttk.Frame(root)
        container.pack(side='top', fill='x', padx=6, pady=4)

        h_canvas = tk.Canvas(container, height=220)
        h_scroll = ttk.Scrollbar(container, orient='horizontal', command=h_canvas.xview)
        h_canvas.configure(xscrollcommand=h_scroll.set)
        h_canvas.pack(side='top', fill='x', expand=True)
        h_scroll.pack(side='top', fill='x')

        top_inner = ttk.Frame(h_canvas)
        h_canvas.create_window((0, 0), window=top_inner, anchor='nw')
        top_inner.bind("<Configure>", lambda e: h_canvas.configure(scrollregion=h_canvas.bbox("all")))

        # ---------- Folder ----------
        folder_frame = ttk.LabelFrame(top_inner, text="Folder", padding=6)
        folder_frame.grid(row=0, column=0, sticky='nw', padx=4, pady=2)
        self.folder_var = tk.StringVar()
        ttk.Label(folder_frame, text="Folder:").grid(row=0, column=0, sticky='w')
        ttk.Entry(folder_frame, textvariable=self.folder_var, width=30).grid(row=0, column=1, padx=4)
        ttk.Button(folder_frame, text="Browse", command=self.browse_folder).grid(row=0, column=2, padx=4)
        ttk.Button(folder_frame, text="Refresh Files", command=self.rebuild_file_list).grid(row=0, column=3, padx=4)

        # ---------- Files ----------
        self.files_frame = ttk.LabelFrame(top_inner, text="Files", padding=6)
        self.files_frame.grid(row=0, column=1, sticky='nw', padx=4, pady=2)
        self.file_list_canvas = tk.Canvas(self.files_frame, width=250, height=120)
        self.file_list_scroll = ttk.Scrollbar(self.files_frame, orient='vertical', command=self.file_list_canvas.yview)
        self.file_list_inner = ttk.Frame(self.file_list_canvas)
        self.file_list_inner.bind("<Configure>", lambda e: self.file_list_canvas.configure(scrollregion=self.file_list_canvas.bbox("all")))
        self.file_list_canvas.create_window((0, 0), window=self.file_list_inner, anchor='nw')
        self.file_list_canvas.configure(yscrollcommand=self.file_list_scroll.set)
        self.file_list_canvas.grid(row=0, column=0, sticky='nsew')
        self.file_list_scroll.grid(row=0, column=1, sticky='ns')

        # ---------- Legend & Colors ----------
        legend_frame = ttk.LabelFrame(top_inner, text="Legend & Colors", padding=6)
        legend_frame.grid(row=0, column=2, sticky='nw', padx=4, pady=2)
        ttk.Button(legend_frame, text="Set Colors", command=self.configure_colors).grid(row=0, column=0, padx=4, pady=2)
        ttk.Button(legend_frame, text="Rename Legends", command=self.configure_labels).grid(row=0, column=1, padx=4, pady=2)
        ttk.Button(legend_frame, text="Legend Order", command=self.configure_order).grid(row=0, column=2, padx=4, pady=2)
        ttk.Checkbutton(legend_frame, text="Auto Legend", variable=self.auto_legend).grid(row=1, column=0, padx=4, pady=2, sticky='w')
        ttk.Button(legend_frame, text="Add Legend", command=self.add_legend).grid(row=1, column=1, padx=4)
        ttk.Button(legend_frame, text="Remove Legend", command=self.remove_legend).grid(row=1, column=2, padx=4)

        ttk.Label(legend_frame, text="Legend loc:").grid(row=2, column=0, sticky='e', padx=2)
        legend_opts = ['best','upper right','upper left','lower right','lower left','right','center left',
                       'center right','lower center','upper center','center']
        self.legend_loc_cb = ttk.Combobox(legend_frame, values=legend_opts, state='readonly', width=12)
        self.legend_loc_cb.current(0)
        self.legend_loc_cb.grid(row=2, column=1, padx=2, sticky='w')

        # Font picker
        ttk.Label(legend_frame, text="Font:").grid(row=3, column=0, padx=2, sticky='e')
        common_fonts = ["Calibri", "Cambria", "Georgia", "DejaVu Sans", "Arial", "Times New Roman",
                        "Courier New", "Liberation Sans", "Verdana", "Segoe UI", "Tahoma"]
        installed = {f.name for f in fm.fontManager.ttflist}
        font_choices = [f for f in common_fonts if f in installed] or sorted(list(installed))[:12]
        self.font_cb = ttk.Combobox(legend_frame, values=font_choices, textvariable=self.font_family_var, state='readonly', width=18)
        if self.font_family_var.get() not in font_choices:
            self.font_family_var.set(font_choices[0])
        self.font_cb.set(self.font_family_var.get())
        self.font_cb.grid(row=3, column=1, padx=2, sticky='w')
        self.font_cb.bind('<<ComboboxSelected>>', lambda e: self.update_plot())
        self.font_family_var.trace_add('write', lambda *args: self.update_plot())
        self.font_display = ttk.Label(legend_frame, textvariable=self.font_family_var)
        self.font_display.grid(row=3, column=2, padx=4)
        ttk.Label(legend_frame, text="Legend font size:").grid(row=4, column=0, padx=2, sticky='e')
        self.legend_font = ttk.Entry(legend_frame, width=4); self.legend_font.insert(0,'10'); self.legend_font.grid(row=4, column=1, padx=2)

        # ---------- Detector & Axis ----------
        det_axis_frame = ttk.LabelFrame(top_inner, text="Detector & Axis", padding=6)
        det_axis_frame.grid(row=0, column=3, sticky='nw', padx=4, pady=2)

        ttk.Label(det_axis_frame, text="Detector:").grid(row=0, column=0, padx=2, sticky='e')
        self.detector_cb = ttk.Combobox(det_axis_frame, values=['RI','Light Scattering','Viscometry'], state='readonly', width=12)
        self.detector_cb.current(0); self.detector_cb.grid(row=0, column=1, padx=2)

        self.plot_type_var = tk.StringVar(value="Chromatogram")
        ttk.Label(det_axis_frame, text="Plot Type:").grid(row=0, column=4, padx=6, sticky='e')
        self.plot_type_cb = ttk.Combobox(
            det_axis_frame,
            values=["Chromatogram", "Mass Distribution (Signal)", "Mass Distribution (Integral %)"],
            state='readonly', width=26, textvariable=self.plot_type_var
        )
        self.plot_type_cb.grid(row=0, column=5, padx=2, sticky='w')
        self.plot_type_cb.bind('<<ComboboxSelected>>', lambda e: self.update_plot())

        ttk.Label(det_axis_frame, text="X-min:").grid(row=1, column=0, padx=2, sticky='e')
        self.xmin = ttk.Entry(det_axis_frame, width=6); self.xmin.insert(0,'0'); self.xmin.grid(row=1, column=1, padx=2)
        ttk.Label(det_axis_frame, text="X-max:").grid(row=1, column=2, padx=2, sticky='e')
        self.xmax = ttk.Entry(det_axis_frame, width=6); self.xmax.insert(0,'10'); self.xmax.grid(row=1, column=3, padx=2)

        self.logx_mass_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(det_axis_frame, text="Log₁₀(M)", variable=self.logx_mass_var,
                        command=self.update_plot).grid(row=1, column=5, padx=2, sticky='w')

        # Baseline extension controls
        self.baseline_toggle_btn = ttk.Button(det_axis_frame, text="Baseline: On", command=self.toggle_baseline)
        self.baseline_toggle_btn.grid(row=2, column=0, padx=4, pady=(6,2), sticky='w')

        ttk.Label(det_axis_frame, text="Extend ± (x units):").grid(row=2, column=1, padx=2, sticky='e')
        self.baseline_scale = ttk.Scale(det_axis_frame, from_=0.0, to=10.0, orient='horizontal',
                                        variable=self.baseline_len_var, command=lambda *_: self.update_plot())
        self.baseline_scale.grid(row=2, column=2, padx=2, sticky='we')
        det_axis_frame.grid_columnconfigure(2, weight=1)

        self.baseline_spin = ttk.Spinbox(det_axis_frame, from_=0.0, to=100.0, increment=0.1,
                                         textvariable=self.baseline_len_var, width=6, command=self.update_plot)
        self.baseline_spin.grid(row=2, column=3, padx=2, sticky='w')

        ttk.Label(det_axis_frame, text="Baseline ref:").grid(row=3, column=0, padx=4, sticky='e')
        self.baseline_mode_cb = ttk.Combobox(det_axis_frame, values=["X-limits", "Data range"],
                                             textvariable=self.baseline_mode, state='readonly', width=12)
        self.baseline_mode_cb.grid(row=3, column=1, padx=4, sticky='w')
        self.baseline_mode_cb.bind('<<ComboboxSelected>>', lambda e: self.update_plot())

        # Taper controls
        ttk.Checkbutton(det_axis_frame, text="Taper to zero", variable=self.taper_var,
                        command=self.update_plot).grid(row=4, column=0, padx=4, sticky='w')
        ttk.Label(det_axis_frame, text="Taper length (x units):").grid(row=4, column=1, padx=2, sticky='e')
        self.taper_scale = ttk.Scale(det_axis_frame, from_=0.0, to=2.0, orient='horizontal',
                                     variable=self.taper_len_var, command=lambda *_: self.update_plot())
        self.taper_scale.grid(row=4, column=2, padx=2, sticky='we')
        self.taper_spin = ttk.Spinbox(det_axis_frame, from_=0.0, to=10.0, increment=0.1,
                                      textvariable=self.taper_len_var, width=6, command=self.update_plot)
        self.taper_spin.grid(row=4, column=3, padx=2, sticky='w')

        # Smoothing sigma
        ttk.Label(det_axis_frame, text="Smoothing σ:").grid(row=5, column=0, padx=2, sticky='e')
        self.sigma_spin = ttk.Spinbox(det_axis_frame, from_=0.0, to=10.0, increment=0.1,
                                      textvariable=self.sigma_var, width=6, command=self.update_plot)
        self.sigma_spin.grid(row=5, column=1, padx=2, sticky='w')

        # ---------- Display Options ----------
        display_frame = ttk.LabelFrame(top_inner, text="Display Options", padding=6)
        display_frame.grid(row=0, column=4, sticky='nw', padx=4, pady=2)
        ttk.Checkbutton(display_frame, text="Inline labels", variable=self.inline_var).grid(row=0, column=0, padx=4, sticky='w')
        ttk.Checkbutton(display_frame, text="Use Gradient", variable=self.gradient_var).grid(row=0, column=1, padx=4, sticky='w')
        ttk.Checkbutton(display_frame, text="Use Math ($…$)", variable=self.use_mathtext,
                        command=self.update_plot).grid(row=0, column=2, padx=4, sticky='w')
        ttk.Label(display_frame, text="Gradient Start:").grid(row=1, column=0, padx=2, sticky='e')
        ttk.Button(display_frame, text="Set", command=self.pick_gradient_start).grid(row=1, column=1, padx=2)
        ttk.Label(display_frame, text="Gradient End:").grid(row=1, column=2, padx=2, sticky='e')
        ttk.Button(display_frame, text="Set", command=self.pick_gradient_end).grid(row=1, column=3, padx=2)
        ttk.Checkbutton(display_frame, text="Hide X-axis", variable=self.hide_x_var).grid(row=2, column=0, padx=4, sticky='w')
        ttk.Checkbutton(display_frame, text="Hide Y-axis", variable=self.hide_y_var).grid(row=2, column=1, padx=4, sticky='w')
        ttk.Label(display_frame, text="X Axis Title:").grid(row=3, column=0, padx=2, sticky='e')
        ttk.Entry(display_frame, textvariable=self.xlabel_var, width=15).grid(row=3, column=1, padx=2)
        ttk.Label(display_frame, text="Y Axis Title:").grid(row=3, column=2, padx=2, sticky='e')
        ttk.Entry(display_frame, textvariable=self.ylabel_var, width=15).grid(row=3, column=3, padx=2)
        ttk.Label(display_frame, text="Axis Font Size:").grid(row=4, column=0, padx=2, sticky='e')
        ttk.Spinbox(display_frame, from_=6, to=30, textvariable=self.axis_fontsize_var, width=5).grid(row=4, column=1, padx=2, sticky='w')

        # ---------- Molecules (optional overlay) ----------
        mol_frame = ttk.LabelFrame(top_inner, text="Molecules", padding=6)
        mol_frame.grid(row=0, column=5, sticky='nw', padx=4, pady=2)
        ttk.Button(mol_frame, text="Add SMILES File", command=self.add_smiles_file).grid(row=0, column=0, padx=4)
        self.mol_overlay = tk.Canvas(root, width=300, height=200, bg=root['bg'], highlightthickness=0)
        self.mol_overlay.place(relx=0.7, rely=0.3)
        self.molecule_widgets = []

        # ---------- Actions ----------
        action_frame = ttk.LabelFrame(top_inner, text="Actions", padding=6)
        action_frame.grid(row=0, column=6, sticky='nw', padx=4, pady=2)
        ttk.Button(action_frame, text="Plot", command=self.update_plot).grid(row=0, column=0, padx=6, pady=4)
        ttk.Button(action_frame, text="Save Figure", command=self.save_plot).grid(row=0, column=1, padx=6, pady=4)
        ttk.Button(action_frame, text="Exit", command=self.root.destroy).grid(row=0, column=2, padx=6, pady=4)
        ttk.Button(action_frame, text="Save Session", command=self.save_session).grid(row=1, column=0, padx=6, pady=4, sticky='w')
        ttk.Button(action_frame, text="Load Session", command=self.load_session).grid(row=1, column=1, padx=6, pady=4, sticky='w')

        # ---------- Figure ----------
        self.fig, self.ax = plt.subplots()
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().pack(side='top', fill='both', expand=True, padx=6, pady=4)

        if not self.legend_order:
            self.legend_order = [f for f in sorted(self.file_vars.keys()) if self.file_vars.get(f, tk.BooleanVar(value=False)).get()]
        self._push_history()
        self.update_plot()

    # ===== helpers =====
    def toggle_baseline(self):
        self.extend_baseline_var.set(not self.extend_baseline_var.get())
        self.baseline_toggle_btn.config(text=f"Baseline: {'On' if self.extend_baseline_var.get() else 'Off'}")
        self.update_plot()

    def pick_gradient_start(self):
        c = colorchooser.askcolor(title="Gradient start color", initialcolor=self.gradient_start_color, parent=self.root)
        if c and c[1]:
            self.gradient_start_color = c[1]

    def pick_gradient_end(self):
        c = colorchooser.askcolor(title="Gradient end color", initialcolor=self.gradient_end_color, parent=self.root)
        if c and c[1]:
            self.gradient_end_color = c[1]

    def add_smiles_file(self):
        fp = filedialog.askopenfilename(
            title="Select molecule file",
            filetypes=[("Molecule/SMILES/Mol/SDF", "*.smi *.txt *.mol *.sdf"), ("All", "*.*")]
        )
        if not fp:
            return
        ext = os.path.splitext(fp)[1].lower()
        display_text = os.path.basename(fp)
        mol = None
        try:
            if ext in ('.smi', '.txt'):
                with open(fp, 'r') as f:
                    line = f.readline().strip()
                display_text = line
                if RDKit_AVAILABLE:
                    mol = Chem.MolFromSmiles(line)
            elif ext == '.mol':
                if RDKit_AVAILABLE:
                    mol = Chem.MolFromMolFile(fp, sanitize=True)
            elif ext == '.sdf':
                if RDKit_AVAILABLE:
                    suppl = Chem.SDMolSupplier(fp)
                    mols = [m for m in suppl if m is not None]
                    mol = mols[0] if mols else None
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load molecule: {e}")
            return
        if RDKit_AVAILABLE and mol is not None:
            try:
                img = Draw.MolToImage(mol, size=(200, 150))
                photo = ImageTk.PhotoImage(img)
                DraggableMolecule(self.mol_overlay, image=photo, label=Chem.MolToSmiles(mol))
                self.add_molecule_to_axes(mol)
            except Exception:
                lbl = tk.Label(self.mol_overlay, text=display_text or "Molecule", bg='white', bd=1, relief='solid')
                DraggableMolecule(self.mol_overlay, widget=lbl, label=display_text)
        else:
            lbl = tk.Label(self.mol_overlay, text=display_text, bg='white', bd=1, relief='solid')
            DraggableMolecule(self.mol_overlay, widget=lbl, label=display_text)

    def add_molecule_to_axes(self, mol):
        try:
            img = Draw.MolToImage(mol, size=(150, 150))
            oi = OffsetImage(img)
            ab = AnnotationBbox(oi, (0.5, 0.5), xycoords='axes fraction', frameon=True)
            self.ax.add_artist(ab)
            self.canvas.draw()
        except Exception as e:
            print(f"[DEBUG] Failed to add molecule to axes: {e}")

    def _capture_state(self):
        return {
            "show_title": self.show_title.get(),
            "custom_title": self.custom_title_var.get(),
            "auto_legend": self.auto_legend.get(),
            "legend_loc": self.legend_loc_cb.get() if hasattr(self, 'legend_loc_cb') else 'best',
            "legend_font": self.legend_font.get() if hasattr(self, 'legend_font') else '10',
        }

    def _apply_state(self, state):
        self.show_title.set(state.get("show_title", False))
        self.custom_title_var.set(state.get("custom_title", ""))
        self.auto_legend.set(state.get("auto_legend", True))
        self.legend_loc_cb.set(state.get("legend_loc", "best"))
        self.legend_font.delete(0, tk.END)
        self.legend_font.insert(0, state.get("legend_font", "10"))

    def _push_history(self):
        if self._hist_index < len(self._history) - 1:
            self._history = self._history[: self._hist_index + 1]
        self._history.append(self._capture_state())
        self._hist_index = len(self._history) - 1

    def undo(self):
        if self._hist_index <= 0:
            return
        self._hist_index -= 1
        self._apply_state(self._history[self._hist_index])
        self.update_plot()

    def redo(self):
        if self._hist_index >= len(self._history) - 1:
            return
        self._hist_index += 1
        self._apply_state(self._history[self._hist_index])
        self.update_plot()

    def browse_folder(self):
        d = filedialog.askdirectory(title="Select folder with data")
        if d:
            self.folder_var.set(d)
            self.rebuild_file_list()

    def rebuild_file_list(self):
        folder = self.folder_var.get()
        if not folder or not os.path.isdir(folder):
            return
        existing = dict(self.file_vars)
        self.file_vars.clear()
        self.file_labels.clear()
        self.file_colors.clear()
        for widget in self.file_list_inner.winfo_children():
            widget.destroy()
        for idx, fname in enumerate(sorted(os.listdir(folder))):
            if fname.lower().endswith(('.txt', '.dat', '.csv')):
                full = os.path.join(folder, fname)
                prev = existing.get(full)
                self.file_vars[full] = tk.BooleanVar(value=prev.get() if prev else True)
                self.file_labels[full] = os.path.basename(full)
                self.file_colors[full] = None
                cb = ttk.Checkbutton(self.file_list_inner, text=os.path.basename(full), variable=self.file_vars[full], command=self.update_plot)
                cb.grid(row=idx, column=0, sticky='w', padx=2, pady=1)
        if not self.legend_order:
            self.legend_order = [f for f in sorted(self.file_vars.keys()) if self.file_vars[f].get()]
        else:
            existing_order = [f for f in self.legend_order if f in self.file_vars]
            new_items = [f for f in sorted(self.file_vars.keys()) if f not in existing_order and self.file_vars[f].get()]
            self.legend_order = existing_order + new_items
        self.update_plot()

    # ---- helper for cosine taper ----
    def _cos_taper(self, x0, y0, x1, y1, n=60):
        xs = np.linspace(x0, x1, max(2, n))
        t = (xs - x0) / max(1e-12, (x1 - x0))
        ys = y0 + (y1 - y0) * 0.5 * (1 - np.cos(np.pi * t))
        return xs.tolist(), ys.tolist()

    # ===== Unicode / mathtext converters =====
    _SUP = str.maketrans("0123456789+-=()", "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾")
    _SUB = str.maketrans("0123456789+-=()n", "₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎ₙ")

    @staticmethod
    def _to_unicode_scripts(s: str) -> str:
        """Convert e.g. X_55->X₅₅, Mw^2->Mw². Removes any '$'."""
        txt = s.replace("$", "")

        def sub_sub(m): return f"{m.group(1)}{m.group(2).translate(GPCApp._SUB)}"
        def sub_sup(m): return f"{m.group(1)}{m.group(2).translate(GPCApp._SUP)}"
        txt = re.sub(r"([A-Za-z\\][A-Za-z0-9\\]*)_\{([^}]+)\}", sub_sub, txt)
        txt = re.sub(r"([A-Za-z\\][A-Za-z0-9\\]*)\^{([^}]+)\}", sub_sup, txt)

        txt = re.sub(r"([A-Za-z\\][A-Za-z0-9\\]*)_(\d+|n)", lambda m: m.group(1) + m.group(2).translate(GPCApp._SUB), txt)
        txt = re.sub(r"([A-Za-z\\][A-Za-z0-9\\]*)\^(\d+)", lambda m: m.group(1) + m.group(2).translate(GPCApp._SUP), txt)
        return txt

    @staticmethod
    def _auto_mathify_text(core: str) -> str:
        """Sanitize for mathtext mode."""
        core = core.replace(" ^", "^").replace("^ ", "^")
        core = core.replace(" _", "_").replace("_ ", "_")
        def sub_sub(m): return f"{m.group(1)}_{{{m.group(2)}}}"
        def sub_sup(m): return f"{m.group(1)}^{{{m.group(2)}}}"
        core = re.sub(r"([A-Za-z\\][A-Za-z0-9\\]*)_(\w+(?:\.\w+)*)", sub_sub, core)
        core = re.sub(r"([A-Za-z\\][A-Za-z0-9\\]*)\^(\w+(?:\.\w+)*)", sub_sup, core)
        return core

    def _auto_mathify(self, s: str) -> str:
        """
        If math mode ON: wrap as $…$ after sanitizing.
        If OFF: return Unicode-sub/sup version so normal fonts apply (e.g., Calibri).
        """
        if not s:
            return s
        base = s.strip()
        if self.use_mathtext.get():
            core = base.replace("$", "").strip()
            if not core:
                return ""
            core = self._auto_mathify_text(core)
            return f"${core}$"
        else:
            return self._to_unicode_scripts(base)

    def update_plot(self):
        self.ax.clear()
        self._legend_handles = []
        self._legend_labels = []

        detector = self.detector_cb.get()
        xmin = float(self.xmin.get()) if self.xmin.get() else None
        xmax = float(self.xmax.get()) if self.xmax.get() else None
        sigma = float(self.sigma_var.get())
        fontname = self.font_family_var.get()

        # --- Force Calibri-first everywhere, with sane fallbacks ---
        # Try to pin the exact Calibri font file if available
        try:
            cal_path = fm.findfont(FontProperties(family=fontname, style='normal', weight='normal'),
                                   fallback_to_default=False)
            use_fname = os.path.basename(cal_path).lower().startswith('calibri')
        except Exception:
            cal_path = None
            use_fname = False

        mpl.rcParams['font.family'] = [fontname, 'Segoe UI Symbol', 'DejaVu Sans']
        mpl.rcParams['font.style'] = 'normal'
        mpl.rcParams['text.usetex'] = False
        mpl.rcParams['mathtext.default'] = 'regular'

        if use_fname and os.path.exists(cal_path):
            fp_main = FontProperties(fname=cal_path, size=self.axis_fontsize_var.get(),
                                     style='normal', weight='normal')
        else:
            fp_main = FontProperties(family=[fontname, 'Segoe UI Symbol', 'DejaVu Sans'],
                                     size=self.axis_fontsize_var.get(), style='normal', weight='normal')

        mode = self.plot_type_var.get()
        mass_mode = mode.startswith("Mass Distribution")

        traces = []
        data_min, data_max = None, None

        def _safe_set_xlim(left, right):
            """Apply x-limits only if finite; otherwise fall back to autoscale."""
            left = left if left is None or np.isfinite(left) else None
            right = right if right is None or np.isfinite(right) else None
            if left is None and right is None:
                self.ax.relim()
                self.ax.autoscale_view(scalex=True, scaley=False)
                return False
            self.ax.set_xlim(left=left, right=right)
            return True

        if mass_mode:
            for f in self.legend_order:
                var = self.file_vars.get(f)
                if not var or not var.get():
                    continue
                M, sig_md, integ_md = load_mass_distribution(f)
                if M is None:
                    continue

                if "Integral" in mode:
                    y = integ_md
                else:
                    y = sig_md

                xvals = np.asarray(M, dtype=float)
                y = np.asarray(y, dtype=float)

                finite = np.isfinite(xvals) & np.isfinite(y)
                if self.logx_mass_var.get():
                    finite &= xvals > 0

                if not np.any(finite):
                    continue

                xvals = xvals[finite]
                y = y[finite]

                order = np.argsort(xvals)
                xvals = xvals[order]
                y = y[order]

                if self.logx_mass_var.get():
                    x_for_smooth = np.log10(np.clip(xvals, a_min=np.finfo(float).tiny, a_max=None))
                else:
                    x_for_smooth = xvals

                y = smooth_series(x_for_smooth, y, sigma)
                y = np.clip(y, 0.0, None)

                if self.normalize_var.get():
                    if "Integral" in mode:
                        if y.size:
                            maxv = float(np.nanmax(y))
                            if np.isfinite(maxv) and maxv > 0:
                                y = y / maxv
                    else:
                        xvals, y = compute_weight_fraction(xvals, y)
                        if not len(xvals):
                            continue
                        y = np.clip(y, 0.0, None)
                        if y.size:
                            peak = float(np.nanmax(y))
                            if np.isfinite(peak) and peak > 0:
                                y = y / peak
                elif len(y):
                    maxv = float(np.nanmax(y))
                    if np.isfinite(maxv) and maxv > 0:
                        y = y / maxv

                if self.logx_mass_var.get():
                    plot_x = np.log10(np.clip(xvals, a_min=np.finfo(float).tiny, a_max=None))
                else:
                    plot_x = xvals

                if plot_x.size == 0 or y.size == 0:
                    continue

                tmin = float(np.nanmin(plot_x))
                tmax = float(np.nanmax(plot_x))
                if not (np.isfinite(tmin) and np.isfinite(tmax)):
                    continue

                label_raw = self.file_labels.get(f, f)
                label = self._auto_mathify(label_raw)
                traces.append({"x": plot_x, "y": y, "label": label, "color": self.file_colors.get(f, None)})
                data_min = tmin if data_min is None else min(data_min, tmin)
                data_max = tmax if data_max is None else max(data_max, tmax)

            if not traces:
                self.canvas.draw()
                return

            if self.baseline_mode.get() == "X-limits":
                ref_left = xmin if xmin is not None else data_min
                ref_right = xmax if xmax is not None else data_max
            else:
                ref_left, ref_right = data_min, data_max
            if ref_right < ref_left:
                ref_left, ref_right = ref_right, ref_left

            ext = float(self.baseline_len_var.get() or 0.0)
            ext_left, ext_right = ref_left - ext, ref_right + ext

            if self.extend_baseline_var.get():
                x_left_final = xmin if xmin is not None else ext_left
                x_right_final = xmax if xmax is not None else ext_right
            else:
                x_left_final = xmin
                x_right_final = xmax

            taper_on = self.taper_var.get()
            taper_len = max(0.0, float(self.taper_len_var.get() or 0.0))

            for tr in traces:
                plot_x = tr["x"]
                plot_y = tr["y"]
                label = tr["label"]
                color = tr["color"]

                if not self.extend_baseline_var.get():
                    line, = self.ax.plot(plot_x, plot_y, color=color)
                    line.set_label(label)
                    self._legend_handles.append(line)
                    self._legend_labels.append(label)
                    continue

                left_data = float(plot_x[0])
                right_data = float(plot_x[-1])
                y_left = float(plot_y[0])
                y_right = float(plot_y[-1])

                xplot = []
                yplot = []

                if taper_on:
                    left_taper = min(
                        taper_len,
                        max(0.0, left_data - (x_left_final if x_left_final is not None else left_data))
                    )
                else:
                    left_taper = 0.0
                left_zero_end = left_data - left_taper

                if x_left_final is not None and x_left_final < left_zero_end:
                    xplot.extend([x_left_final, left_zero_end]); yplot.extend([0.0, 0.0])

                if left_taper > 0:
                    xs, ys = self._cos_taper(left_zero_end, 0.0, left_data, y_left)
                    if xs and xplot and xs[0] == xplot[-1]:
                        xs = xs[1:]; ys = ys[1:]
                    xplot.extend(xs); yplot.extend(ys)

                xplot.extend(plot_x.tolist()); yplot.extend(plot_y.tolist())

                if taper_on:
                    right_taper = min(
                        taper_len,
                        max(0.0, (x_right_final if x_right_final is not None else right_data) - right_data)
                    )
                else:
                    right_taper = 0.0
                right_zero_start = right_data + right_taper

                if right_taper > 0:
                    xs, ys = self._cos_taper(right_data, y_right, right_zero_start, 0.0)
                    if xs and xplot and xs[0] == xplot[-1]:
                        xs = xs[1:]; ys = ys[1:]
                    xplot.extend(xs); yplot.extend(ys)

                if x_right_final is not None and right_zero_start < x_right_final:
                    xplot.extend([right_zero_start, x_right_final]); yplot.extend([0.0, 0.0])

                line, = self.ax.plot(xplot, yplot, color=color)
                line.set_label(label)
                self._legend_handles.append(line)
                self._legend_labels.append(label)

            if self.hide_x_var.get():
                self.ax.get_xaxis().set_visible(False)
            if self.hide_y_var.get():
                self.ax.get_yaxis().set_visible(False)

            xlab = "log\u2081\u2080(M) [g/mol]" if self.logx_mass_var.get() else "Molar mass [g/mol]"
            xlabel = self._auto_mathify(self.xlabel_var.get().strip()) or xlab
            ylabel = self._auto_mathify(self.ylabel_var.get().strip()) or \
                     ("Integral [%]" if "Integral" in mode else "Distribution (arb.)")
            self.ax.set_xlabel(xlabel, fontproperties=fp_main)
            self.ax.set_ylabel(ylabel, fontproperties=fp_main)

            for lbl in self.ax.get_xticklabels() + self.ax.get_yticklabels():
                lbl.set_fontproperties(fp_main)

            if self.auto_legend.get():
                self._apply_legend()

            if self.extend_baseline_var.get() and (x_left_final is not None or x_right_final is not None):
                _safe_set_xlim(x_left_final, x_right_final)
            if xmin is not None or xmax is not None:
                _safe_set_xlim(xmin, xmax)

        else:
            for f in self.legend_order:
                var = self.file_vars.get(f)
                if not var or not var.get():
                    continue
                times, vals = load_and_smooth(f, detector, sigma=sigma)
                if times is None or len(times) == 0:
                    continue
                if self.normalize_var.get():
                    maxv = vals.max() if len(vals) else 1.0
                    if maxv != 0:
                        vals = vals / maxv
                label_raw = self.file_labels.get(f, f)
                label = self._auto_mathify(label_raw)
                traces.append({"times": times, "vals": vals,
                               "label": label, "color": self.file_colors.get(f, None)})

            if not traces:
                self.canvas.draw()
                return

            data_min = min(float(tr["times"].min()) for tr in traces)
            data_max = max(float(tr["times"].max()) for tr in traces)

            if self.baseline_mode.get() == "X-limits":
                ref_left  = xmin if xmin is not None else data_min
                ref_right = xmax if xmax is not None else data_max
            else:
                ref_left, ref_right = data_min, data_max
            if ref_right < ref_left:
                ref_left, ref_right = ref_right, ref_left

            ext = float(self.baseline_len_var.get() or 0.0)
            ext_left, ext_right = ref_left - ext, ref_right + ext

            if self.extend_baseline_var.get():
                x_left_final = xmin if xmin is not None else ext_left
                x_right_final = xmax if xmax is not None else ext_right
            else:
                x_left_final = xmin
                x_right_final = xmax

            taper_on = self.taper_var.get()
            taper_len = max(0.0, float(self.taper_len_var.get() or 0.0))

            for tr in traces:
                times = tr["times"]; vals = tr["vals"]
                label = tr["label"]; color = tr["color"]

                if not self.extend_baseline_var.get():
                    line, = self.ax.plot(times, vals, color=color)
                    line.set_label(label)
                    self._legend_handles.append(line)
                    self._legend_labels.append(label)
                    continue

                left_data = float(times.min())
                right_data = float(times.max())
                y_left = float(vals[0])
                y_right = float(vals[-1])

                xplot = []
                yplot = []

                left_taper = min(taper_len, max(0.0, left_data - (x_left_final if x_left_final is not None else left_data))) if taper_on else 0.0
                left_zero_end = left_data - left_taper

                if x_left_final is not None and x_left_final < left_zero_end:
                    xplot.extend([x_left_final, left_zero_end]); yplot.extend([0.0, 0.0])

                if left_taper > 0:
                    xs, ys = self._cos_taper(left_zero_end, 0.0, left_data, y_left)
                    if xs and xplot and xs[0] == xplot[-1]:
                        xs = xs[1:]; ys = ys[1:]
                    xplot.extend(xs); yplot.extend(ys)

                xplot.extend(times.tolist()); yplot.extend(vals.tolist())

                right_taper = min(taper_len, max(0.0, (x_right_final if x_right_final is not None else right_data) - right_data)) if taper_on else 0.0
                right_zero_start = right_data + right_taper

                if right_taper > 0:
                    xs, ys = self._cos_taper(right_data, y_right, right_zero_start, 0.0)
                    if xs and xplot and xs[0] == xplot[-1]:
                        xs = xs[1:]; ys = ys[1:]
                    xplot.extend(xs); yplot.extend(ys)

                if x_right_final is not None and right_zero_start < x_right_final:
                    xplot.extend([right_zero_start, x_right_final]); yplot.extend([0.0, 0.0])

                line, = self.ax.plot(xplot, yplot, color=color)
                line.set_label(label)
                self._legend_handles.append(line)
                self._legend_labels.append(label)

            if self.hide_x_var.get():
                self.ax.get_xaxis().set_visible(False)
            if self.hide_y_var.get():
                self.ax.get_yaxis().set_visible(False)

            xlabel = self._auto_mathify(self.xlabel_var.get().strip())
            ylabel = self._auto_mathify(self.ylabel_var.get().strip())
            self.ax.set_xlabel(xlabel if xlabel else "", fontproperties=fp_main)
            self.ax.set_ylabel(ylabel if ylabel else "", fontproperties=fp_main)

            for lbl in self.ax.get_xticklabels() + self.ax.get_yticklabels():
                lbl.set_fontproperties(fp_main)

            if self.inline_var.get():
                for line, label in zip(self._legend_handles, self._legend_labels):
                    xdata = line.get_xdata(); ydata = line.get_ydata()
                    if len(xdata) and len(ydata):
                        self.ax.text(xdata[-1], ydata[-1], label, fontproperties=fp_main)

            if self.auto_legend.get():
                self._apply_legend()

            if x_left_final is not None or x_right_final is not None:
                _safe_set_xlim(x_left_final, x_right_final)
            if xmin is not None or xmax is not None:
                _safe_set_xlim(xmin, xmax)

        if self.show_title.get():
            custom = self.custom_title_var.get().strip()
            title_base = f"{mode}"
            if not mass_mode:
                title_base = f"Detector: {detector} (σ={sigma:.1f})"
                if self.normalize_var.get():
                    title_base += " (normalized)"
            elif self.normalize_var.get():
                title_base += " (normalized)"
            self.ax.set_title(self._auto_mathify(custom) if custom else title_base, fontproperties=fp_main)
        else:
            self.ax.set_title("")

        self.canvas.draw()

    def _apply_legend(self):
        legend_fontsize = float(self.legend_font.get()) if self.legend_font.get() else 10
        if self.ax.get_legend():
            self.ax.get_legend().remove()

        # Mirror the same font logic used in update_plot, but with legend size
        try:
            cal_path = fm.findfont(FontProperties(family=self.font_family_var.get(),
                                                  style='normal', weight='normal'),
                                   fallback_to_default=False)
            use_fname = os.path.basename(cal_path).lower().startswith('calibri')
        except Exception:
            cal_path = None
            use_fname = False

        if use_fname and os.path.exists(cal_path):
            fp_leg = FontProperties(fname=cal_path, size=legend_fontsize,
                                    style='normal', weight='normal')
        else:
            fp_leg = FontProperties(family=[self.font_family_var.get(), 'Segoe UI Symbol', 'DejaVu Sans'],
                                    size=legend_fontsize, style='normal', weight='normal')

        if getattr(self, "_legend_handles", None) and getattr(self, "_legend_labels", None):
            self.ax.legend(self._legend_handles, self._legend_labels,
                           loc=self.legend_loc_cb.get(), prop=fp_leg)
        else:
            self.ax.legend(loc=self.legend_loc_cb.get(), prop=fp_leg)

    def add_legend(self):
        self._apply_legend()
        self.canvas.draw()

    def remove_legend(self):
        if self.ax.get_legend():
            self.ax.get_legend().remove()
            self.canvas.draw()

    def save_plot(self):
        fp = filedialog.asksaveasfilename(defaultextension='.png',
                                          filetypes=[('PNG','*.png'),('JPEG','*.jpg'),('All','*.*')])
        if not fp:
            return
        dlg = tk.Toplevel(self.root); dlg.title("Save Plot Options")

        cur_w, cur_h = self.fig.get_size_inches()
        default_side = f"{min(cur_w, cur_h):.2f}"
        ttk.Label(dlg, text="Width:").grid(row=0, column=0, padx=5, pady=5, sticky='e')
        width_var = tk.StringVar(value=default_side)
        w_entry = ttk.Entry(dlg, textvariable=width_var, width=8)
        w_entry.grid(row=0, column=1, padx=5, pady=5)

        ttk.Label(dlg, text="Height:").grid(row=1, column=0, padx=5, pady=5, sticky='e')
        height_var = tk.StringVar(value=default_side)
        h_entry = ttk.Entry(dlg, textvariable=height_var, width=8)
        h_entry.grid(row=1, column=1, padx=5, pady=5)

        square_var = tk.BooleanVar(value=True)
        def toggle_square():
            if square_var.get():
                height_var.set(width_var.get())
                h_entry.configure(state='disabled')
            else:
                h_entry.configure(state='normal')
        ttk.Checkbutton(dlg, text="Square", variable=square_var, command=toggle_square).grid(row=2, column=0, columnspan=2, pady=4)
        def on_width_change(*args):
            if square_var.get():
                height_var.set(width_var.get())
        width_var.trace_add('write', on_width_change)
        toggle_square()

        ttk.Label(dlg, text="Units:").grid(row=3, column=0, padx=5, pady=5, sticky='e')
        unit_cb = ttk.Combobox(dlg, values=['inches', 'pixels'], state='readonly', width=10)
        unit_cb.current(0); unit_cb.grid(row=3, column=1, padx=5, pady=5)

        ttk.Label(dlg, text="DPI:").grid(row=4, column=0, padx=5, pady=5, sticky='e')
        dpi_var = tk.StringVar(value='100')
        ttk.Entry(dlg, textvariable=dpi_var, width=8).grid(row=4, column=1, padx=5, pady=5)

        def do_save():
            try:
                dpi = float(dpi_var.get())
                unit = unit_cb.get()
                w_val = float(width_var.get())
                h_val = float(height_var.get())
                if 'pixels' in unit:
                    w = w_val / dpi
                    h = h_val / dpi
                else:
                    w = w_val
                    h = h_val
                orig_size = self.fig.get_size_inches().copy()
                self.fig.set_size_inches(w, h)
                self.fig.tight_layout()
                self.fig.savefig(fp, dpi=dpi)
                self.fig.set_size_inches(orig_size)
                self.canvas.draw()
                messagebox.showinfo("Saved", f"Saved to:\n{fp}")
                dlg.destroy()
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save: {e}")

        btn_frame = ttk.Frame(dlg)
        btn_frame.grid(row=5, column=0, columnspan=2, pady=10)
        ttk.Button(btn_frame, text="Save", command=do_save).grid(row=0, column=0, padx=5)
        ttk.Button(btn_frame, text="Cancel", command=dlg.destroy).grid(row=0, column=1, padx=5)

    # ===== Session save/load =====
    def save_session(self):
        try:
            state = {
                "folder": self.folder_var.get(),
                "selected_files": {path: var.get() for path, var in self.file_vars.items()},
                "file_labels": self.file_labels,
                "file_colors": self.file_colors,
                "legend_order": self.legend_order,

                "detector": self.detector_cb.get(),
                "xmin": self.xmin.get(),
                "xmax": self.xmax.get(),
                "extend_baseline": self.extend_baseline_var.get(),
                "baseline_len": float(self.baseline_len_var.get() or 0.0),
                "baseline_mode": self.baseline_mode.get(),
                "taper": self.taper_var.get(),
                "taper_len": float(self.taper_len_var.get() or 0.0),
                "sigma": float(self.sigma_var.get() or 0.0),
                "normalize": self.normalize_var.get(),
                "xlabel": self.xlabel_var.get(),
                "ylabel": self.ylabel_var.get(),
                "axis_fontsize": int(self.axis_fontsize_var.get() or 10),
                "font_family": self.font_family_var.get(),
                "auto_legend": self.auto_legend.get(),
                "legend_loc": self.legend_loc_cb.get(),
                "legend_font": self.legend_font.get(),
                "show_title": self.show_title.get(),
                "custom_title": self.custom_title_var.get(),
                "hide_x": self.hide_x_var.get(),
                "hide_y": self.hide_y_var.get(),
            }

            fp = filedialog.asksaveasfilename(
                title="Save Session As",
                defaultextension=".gpc.json",
                filetypes=[("GPC Session", "*.gpc.json"), ("JSON", "*.json"), ("All", "*.*")]
            )
            if not fp:
                return
            with open(fp, "w", encoding="utf-8") as f:
                json.dump(state, f, indent=2)
            messagebox.showinfo("Saved", f"Session saved to:\n{fp}")
        except Exception as e:
            messagebox.showerror("Error", f"Could not save session:\n{e}")

    def load_session(self):
        try:
            fp = filedialog.askopenfilename(
                title="Open Session",
                filetypes=[("GPC Session", "*.gpc.json"), ("JSON", "*.json"), ("All", "*.*")]
            )
            if not fp:
                return
            with open(fp, "r", encoding="utf-8") as f:
                state = json.load(f)

            folder = state.get("folder") or ""
            if folder and os.path.isdir(folder):
                self.folder_var.set(folder)
                self.rebuild_file_list()

            selected_files = state.get("selected_files", {})
            for path, on in selected_files.items():
                if path in self.file_vars:
                    self.file_vars[path].set(bool(on))
                    if path in state.get("file_labels", {}):
                        self.file_labels[path] = state["file_labels"][path]
                    if path in state.get("file_colors", {}):
                        self.file_colors[path] = state["file_colors"][path]

            saved_order = [p for p in state.get("legend_order", []) if p in self.file_vars]
            if saved_order:
                self.legend_order = saved_order

            det = state.get("detector", "RI")
            if det in ["RI", "Light Scattering", "Viscometry"]:
                self.detector_cb.set(det)

            self.xmin.delete(0, tk.END); self.xmin.insert(0, state.get("xmin", ""))
            self.xmax.delete(0, tk.END); self.xmax.insert(0, state.get("xmax", ""))

            self.extend_baseline_var.set(bool(state.get("extend_baseline", True)))
            self.baseline_toggle_btn.config(text=f"Baseline: {'On' if self.extend_baseline_var.get() else 'Off'}")
            self.baseline_len_var.set(float(state.get("baseline_len", 0.0)))
            self.baseline_mode.set(state.get("baseline_mode", "X-limits"))

            self.taper_var.set(bool(state.get("taper", True)))
            self.taper_len_var.set(float(state.get("taper_len", 0.0)))

            self.sigma_var.set(float(state.get("sigma", 1.0)))
            self.normalize_var.set(bool(state.get("normalize", True)))

            self.xlabel_var.set(state.get("xlabel", ""))
            self.ylabel_var.set(state.get("ylabel", ""))
            try:
                self.axis_fontsize_var.set(int(state.get("axis_fontsize", 10)))
            except Exception:
                pass

            ff = state.get("font_family", self.font_family_var.get())
            self.font_family_var.set(ff)

            self.auto_legend.set(bool(state.get("auto_legend", True)))
            self.legend_loc_cb.set(state.get("legend_loc", "best"))
            self.legend_font.delete(0, tk.END); self.legend_font.insert(0, state.get("legend_font", "10"))

            self.show_title.set(bool(state.get("show_title", False)))
            self.custom_title_var.set(state.get("custom_title", ""))

            self.hide_x_var.set(bool(state.get("hide_x", False)))
            self.hide_y_var.set(bool(state.get("hide_y", False)))

            self.update_plot()
            messagebox.showinfo("Loaded", f"Session loaded from:\n{fp}")
        except Exception as e:
            messagebox.showerror("Error", f"Could not load session:\n{e}")

    # ===== Color/label/order helpers =====
    def configure_colors(self):
        sel = [f for f, v in self.file_vars.items() if v.get()]
        if not sel:
            messagebox.showwarning("No Selection", "Select files first")
            return
        cfg = tk.Toplevel(self.root); cfg.title("Set Colors")
        for idx, f in enumerate(sel):
            ttk.Label(cfg, text=os.path.basename(f)).grid(row=idx, column=0, padx=5, pady=2, sticky='w')
            btn = ttk.Button(cfg, text="Pick Color")
            btn.grid(row=idx, column=1, padx=5)
            btn.config(command=lambda f=f, b=btn: self._pick_file_color(f, b))
        ttk.Button(cfg, text="OK", command=cfg.destroy).grid(row=len(sel), column=0, columnspan=2, pady=10)

    def _pick_file_color(self, f, b):
        c = colorchooser.askcolor(parent=self.root)[1]
        if c:
            self.file_colors[f] = c
            b.configure(style="")

    def configure_labels(self):
        sel = [f for f, v in self.file_vars.items() if v.get()]
        if not sel:
            messagebox.showwarning("No Selection", "Select files first")
            return

        dlg = tk.Toplevel(self.root); dlg.title("Rename Legends (with superscript/subscript)")

        toolbar = ttk.Frame(dlg)
        toolbar.grid(row=0, column=0, columnspan=4, sticky='w', padx=6, pady=(6,2))

        def _effective_entry():
            w = self.root.focus_get()
            return w if isinstance(w, ttk.Entry) or isinstance(w, tk.Entry) else None

        def _wrap_math(entry_widget):
            s = entry_widget.get()
            core = s.replace("$", "")
            entry_widget.delete(0, tk.END)
            entry_widget.insert(0, f"${core}$")
            entry_widget.icursor(tk.END)

        def _apply_script_to_entry(entry_widget, script_char):
            if entry_widget is None:
                messagebox.showinfo("Tip", "Click inside a label field first, then press the button.")
                return
            try:
                sel_start = entry_widget.index("sel.first")
                sel_end   = entry_widget.index("sel.last")
                selected  = entry_widget.get(sel_start, sel_end)
                entry_widget.delete(sel_start, sel_end)
                entry_widget.insert(sel_start, f"{script_char}{{{selected}}}")
                entry_widget.icursor(f"{sel_start}+{len(script_char)+2+len(selected)}c")
            except tk.TclError:
                i = entry_widget.index("insert")
                entry_widget.insert(i, f"{script_char}{{}}")
                entry_widget.icursor(f"{i}+2c")

        ttk.Button(toolbar, text="Superscript (x^y)", command=lambda: _apply_script_to_entry(_effective_entry(), '^')).grid(row=0, column=0, padx=2)
        ttk.Button(toolbar, text="Subscript (x_y)",   command=lambda: _apply_script_to_entry(_effective_entry(), '_')).grid(row=0, column=1, padx=2)
        ttk.Button(toolbar, text="Wrap $…$",         command=lambda: (_wrap_math(_effective_entry()) if _effective_entry() else None)).grid(row=0, column=2, padx=6)

        entries = {}
        entry_widgets = {}
        for idx, f in enumerate(sel, start=1):
            ttk.Label(dlg, text=os.path.basename(f)).grid(row=idx, column=0, padx=6, pady=3, sticky='w')
            var = tk.StringVar(value=self.file_labels.get(f, os.path.basename(f)))
            ent = ttk.Entry(dlg, textvariable=var, width=40)
            ent.grid(row=idx, column=1, padx=6, pady=3, sticky='w')
            entries[f] = var
            entry_widgets[f] = ent

            row_btns = ttk.Frame(dlg)
            row_btns.grid(row=idx, column=2, padx=4, sticky='w')
            ttk.Button(row_btns, text="x^", width=3, command=lambda e=ent: _apply_script_to_entry(e, '^')).grid(row=0, column=0, padx=1)
            ttk.Button(row_btns, text="x_", width=3, command=lambda e=ent: _apply_script_to_entry(e, '_')).grid(row=0, column=1, padx=1)
            ttk.Button(row_btns, text="$",  width=3, command=lambda e=ent: _wrap_math(e)).grid(row=0, column=2, padx=1)

        if entry_widgets:
            first_ent = next(iter(entry_widgets.values()))
            first_ent.focus_set()
            first_ent.icursor(tk.END)

        def save():
            for f, var in entries.items():
                txt = var.get()
                txt = self._auto_mathify(txt)  # mathtext or unicode, depending on toggle
                self.file_labels[f] = txt
            dlg.destroy()
            self.update_plot()
            try:
                self.add_legend()
            except Exception:
                pass

        ttk.Button(dlg, text="OK", command=save).grid(row=len(sel)+1, column=0, columnspan=3, pady=10)

    def configure_order(self):
        sel = [f for f, v in self.file_vars.items() if v.get()]
        if not sel:
            messagebox.showwarning("No Selection", "Select files first")
            return
        dlg = tk.Toplevel(self.root); dlg.title("Reorder Legend / Plot Order")
        ttk.Label(dlg, text="Use Up/Down to reorder:").grid(row=0, column=0, columnspan=3, pady=4)
        lb = tk.Listbox(dlg, height=10, width=50)
        lb.grid(row=1, column=0, columnspan=3, padx=5)

        ordered = [f for f in self.legend_order if f in sel] or sel
        for f in ordered:
            lb.insert("end", self.file_labels.get(f, os.path.basename(f)))

        def move_up():
            idxs = lb.curselection()
            if not idxs or idxs[0] == 0:
                return
            i = idxs[0]
            txt = lb.get(i)
            lb.delete(i)
            lb.insert(i - 1, txt)
            lb.select_set(i - 1)

        def move_down():
            idxs = lb.curselection()
            if not idxs or idxs[0] == lb.size() - 1:
                return
            i = idxs[0]
            txt = lb.get(i)
            lb.delete(i)
            lb.insert(i + 1, txt)
            lb.select_set(i + 1)

        ttk.Button(dlg, text="Up", command=move_up).grid(row=2, column=0, padx=4, pady=4)
        ttk.Button(dlg, text="Down", command=move_down).grid(row=2, column=1, padx=4, pady=4)

        def apply_order():
            label_to_file = {self.file_labels.get(f, os.path.basename(f)): f for f in sel}
            new_order = []
            for i in range(lb.size()):
                label = lb.get(i)
                f = label_to_file.get(label)
                if f and f not in new_order:
                    new_order.append(f)
            for f in sel:
                if f not in new_order:
                    new_order.append(f)
            self.legend_order = new_order
            dlg.destroy()
            self.update_plot()

        ttk.Button(dlg, text="Apply", command=apply_order).grid(row=2, column=2, padx=4, pady=4)


if __name__ == '__main__':
    root = tk.Tk()
    app = GPCApp(root)
    root.mainloop()
=======
import os
import json
import math
import tkinter as tk
from tkinter import filedialog, messagebox, colorchooser
import tkinter.ttk as ttk
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.font_manager import FontProperties
from scipy.ndimage import gaussian_filter1d
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image, ImageTk

# RDKit support for SMILES rendering (optional)
try:
    from rdkit import Chem
    from rdkit.Chem import Draw
    RDKit_AVAILABLE = True
except ImportError:
    RDKit_AVAILABLE = False


class DraggableMolecule:
    def __init__(self, parent_canvas, image=None, widget=None, label=""):
        self.parent = parent_canvas
        self.label = label
        if image is not None:
            self.id = parent_canvas.create_image(10, 10, anchor='nw', image=image)
            self.image = image
        else:
            self.id = parent_canvas.create_window(10, 10, anchor='nw', window=widget)
            self.widget = widget
        self._drag_data = {"x": 0, "y": 0}
        parent_canvas.tag_bind(self.id, "<ButtonPress-1>", self.on_press)
        parent_canvas.tag_bind(self.id, "<ButtonRelease-1>", self.on_release)
        parent_canvas.tag_bind(self.id, "<B1-Motion>", self.on_motion)

    def on_press(self, event):
        self._drag_data["x"] = event.x
        self._drag_data["y"] = event.y

    def on_motion(self, event):
        dx = event.x - self._drag_data["x"]
        dy = event.y - self._drag_data["y"]
        self.parent.move(self.id, dx, dy)
        self._drag_data["x"] = event.x
        self._drag_data["y"] = event.y

    def on_release(self, event):
        pass


def smooth_series(x, y, sigma):
    """Smooth *y* samples measured at coordinates *x* using a Gaussian kernel."""
    if x is None or y is None:
        return y

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if x.size == 0 or y.size == 0 or x.size != y.size:
        return y

    if sigma is None:
        return y

    try:
        sigma = float(sigma)
    except (TypeError, ValueError):
        return y

    if sigma <= 0 or not np.isfinite(sigma):
        return y

    diffs = np.diff(x)
    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    if diffs.size == 0:
        return y

    step = float(np.median(diffs))
    if not np.isfinite(step) or step <= 0:
        return y

    # Treat the user-provided ``sigma`` control as the desired full width at
    # half maximum (FWHM) instead of the raw Gaussian sigma to keep the
    # smoothing gentler, especially for finely sampled data sets.  Convert the
    # requested FWHM to the standard deviation expected by ``gaussian_filter1d``
    # and normalise by the median spacing so wildly irregular step sizes do not
    # explode the kernel width.  The conversion constant comes from
    # ``sigma = fwhm / (2 * sqrt(2 * ln(2)))``.
    fwhm_to_sigma = 0.5 / math.sqrt(2.0 * math.log(2.0))
    sigma_samples = (sigma * fwhm_to_sigma) / step
    if sigma_samples <= 0 or not np.isfinite(sigma_samples):
        return y

    try:
        return gaussian_filter1d(y, sigma=sigma_samples, mode='nearest')
    except Exception:
        # If SciPy cannot honour the requested sigma (for example, extremely
        # large values triggering numerical issues), fall back to the original
        # data instead of crashing the application.
        return y


def load_and_smooth(path, detector, sigma=1):
    """
    Reads GPC text/CSV with at least 4 columns:
    Time, RI, Light Scattering, Viscometry.
    Accepts space/tab or CSV. Smooths chosen detector.
    """
    try:
        try:
            df = pd.read_csv(path, delim_whitespace=True, header=None)
        except Exception:
            df = pd.read_csv(path, header=None)
    except Exception:
        return None, None

    if df.shape[1] < 4:
        return None, None

    df = df.iloc[:, :4]
    df.columns = ['Time', 'RI', 'Light Scattering', 'Viscometry']
    ser = pd.to_numeric(df[detector], errors='coerce')
    times = pd.to_numeric(df['Time'], errors='coerce')
    valid = ser.notna() & times.notna()
    if not valid.any():
        return None, None

    times = times[valid].to_numpy(dtype=float)
    values = ser[valid].to_numpy(dtype=float)

    order = np.argsort(times)
    times = times[order]
    values = values[order]

    smoothed = smooth_series(times, values, sigma)
    return times, smoothed


def load_mass_distribution(path):
    """
    Parses the 'Molar mass' block exported by ChromPilot/WINGPC.
    Returns (M, signal, integral_pct) as numpy arrays or (None, None, None) if not found.
    """
    import re
    import numpy as np

    try:
        with open(path, 'r', encoding='latin-1') as f:
            lines = f.read().splitlines()
    except Exception:
        return None, None, None

    idx = None
    for i, line in enumerate(lines):
        if line.strip().lower().startswith('molar mass'):
            idx = i
    if idx is None:
        return None, None, None

    j = idx + 1
    while j < len(lines) and not lines[j].strip():
        j += 1

    Ms, sig, integ = [], [], []
    for k in range(j, len(lines)):
        parts = re.split(r'[\t,;]+', lines[k].strip())
        if len(parts) < 3:
            break
        try:
            m = float(parts[0].replace(' ', ''))
            y1 = float(parts[1])
            y2 = float(parts[2])
        except ValueError:
            break
        Ms.append(m); sig.append(y1); integ.append(y2)

    if not Ms:
        return None, None, None
    return np.array(Ms, dtype=float), np.array(sig, dtype=float), np.array(integ, dtype=float)


def compute_weight_fraction(masses, signal):
    """Return masses and the weight fraction w(log M) for finite, positive masses."""
    masses = np.asarray(masses, dtype=float)
    signal = np.asarray(signal, dtype=float)

    mask = np.isfinite(masses) & np.isfinite(signal) & (masses > 0)
    if not mask.any():
        return np.array([]), np.array([])

    masses = masses[mask]
    signal = np.clip(signal[mask], 0.0, None)

    order = np.argsort(masses)
    masses = masses[order]
    signal = signal[order]

    if masses.size < 2:
        if signal.size:
            peak = signal.max()
            if peak > 0:
                signal = signal / peak
        return masses, signal

    log_m = np.log10(masses)
    area = np.trapz(signal, log_m)
    if area > 0:
        weights = signal / area
    else:
        peak = signal.max()
        weights = signal / peak if peak > 0 else signal
    return masses, weights


class GPCApp:
    def __init__(self, root):
        self.root = root
        root.title("GPC Analyzer Live Viewer")

        screen_w = root.winfo_screenwidth()
        screen_h = root.winfo_screenheight()
        w = min(1200, int(screen_w * 0.9))
        h = min(800, int(screen_h * 0.9))
        x = (screen_w - w) // 2
        y = (screen_h - h) // 2
        root.geometry(f"{w}x{h}+{x}+{y}")
        root.minsize(900, 600)

        # ---------- state ----------
        self.file_vars = {}
        self.file_labels = {}
        self.file_colors = {}
        self.legend_order = []

        self.gradient_var = tk.BooleanVar(value=False)
        self.inline_var = tk.BooleanVar(value=False)
        self.hide_x_var = tk.BooleanVar(value=False)
        self.hide_y_var = tk.BooleanVar(value=False)
        self.auto_legend = tk.BooleanVar(value=True)
        self.show_title = tk.BooleanVar(value=False)
        self.custom_title_var = tk.StringVar(value="")
        self.normalize_var = tk.BooleanVar(value=True)
        self.xlabel_var = tk.StringVar(value="")
        self.ylabel_var = tk.StringVar(value="")
        self.axis_fontsize_var = tk.IntVar(value=10)
        self.font_family_var = tk.StringVar(value="DejaVu Sans")
        self.gradient_start_color = "#000000"
        self.gradient_end_color = "#FFFFFF"

        # Baseline controls
        self.extend_baseline_var = tk.BooleanVar(value=True)
        self.baseline_len_var = tk.DoubleVar(value=2.0)
        self.baseline_mode = tk.StringVar(value="X-limits")

        # Taper controls
        self.taper_var = tk.BooleanVar(value=True)
        self.taper_len_var = tk.DoubleVar(value=0.3)

        # Smoothing (sigma)
        self.sigma_var = tk.DoubleVar(value=1.0)

        # NEW: math vs unicode scripts
        self.use_mathtext = tk.BooleanVar(value=True)  # OFF -> Unicode subs/sups so normal fonts (e.g., Calibri) apply

        # undo/redo history (minimal)
        self._history = []
        self._hist_index = -1

        # ---------- shortcuts ----------
        root.bind_all("<Control-p>", lambda e: self.update_plot())
        root.bind_all("<Control-s>", lambda e: self.save_plot())
        root.bind_all("<Control-o>", lambda e: self.browse_folder())
        root.bind_all("<Control-z>", lambda e: self.undo())
        root.bind_all("<Control-y>", lambda e: self.redo())

        self.show_title.trace_add("write", lambda *args: self._push_history())
        self.custom_title_var.trace_add("write", lambda *args: self._push_history())
        self.auto_legend.trace_add("write", lambda *args: self._push_history())

        # ---------- top scroller ----------
        container = ttk.Frame(root)
        container.pack(side='top', fill='x', padx=6, pady=4)

        h_canvas = tk.Canvas(container, height=220)
        h_scroll = ttk.Scrollbar(container, orient='horizontal', command=h_canvas.xview)
        h_canvas.configure(xscrollcommand=h_scroll.set)
        h_canvas.pack(side='top', fill='x', expand=True)
        h_scroll.pack(side='top', fill='x')

        top_inner = ttk.Frame(h_canvas)
        h_canvas.create_window((0, 0), window=top_inner, anchor='nw')
        top_inner.bind("<Configure>", lambda e: h_canvas.configure(scrollregion=h_canvas.bbox("all")))

        # ---------- Folder ----------
        folder_frame = ttk.LabelFrame(top_inner, text="Folder", padding=6)
        folder_frame.grid(row=0, column=0, sticky='nw', padx=4, pady=2)
        self.folder_var = tk.StringVar()
        ttk.Label(folder_frame, text="Folder:").grid(row=0, column=0, sticky='w')
        ttk.Entry(folder_frame, textvariable=self.folder_var, width=30).grid(row=0, column=1, padx=4)
        ttk.Button(folder_frame, text="Browse", command=self.browse_folder).grid(row=0, column=2, padx=4)
        ttk.Button(folder_frame, text="Refresh Files", command=self.rebuild_file_list).grid(row=0, column=3, padx=4)

        # ---------- Files ----------
        self.files_frame = ttk.LabelFrame(top_inner, text="Files", padding=6)
        self.files_frame.grid(row=0, column=1, sticky='nw', padx=4, pady=2)
        self.file_list_canvas = tk.Canvas(self.files_frame, width=250, height=120)
        self.file_list_scroll = ttk.Scrollbar(self.files_frame, orient='vertical', command=self.file_list_canvas.yview)
        self.file_list_inner = ttk.Frame(self.file_list_canvas)
        self.file_list_inner.bind("<Configure>", lambda e: self.file_list_canvas.configure(scrollregion=self.file_list_canvas.bbox("all")))
        self.file_list_canvas.create_window((0, 0), window=self.file_list_inner, anchor='nw')
        self.file_list_canvas.configure(yscrollcommand=self.file_list_scroll.set)
        self.file_list_canvas.grid(row=0, column=0, sticky='nsew')
        self.file_list_scroll.grid(row=0, column=1, sticky='ns')

        # ---------- Legend & Colors ----------
        legend_frame = ttk.LabelFrame(top_inner, text="Legend & Colors", padding=6)
        legend_frame.grid(row=0, column=2, sticky='nw', padx=4, pady=2)
        ttk.Button(legend_frame, text="Set Colors", command=self.configure_colors).grid(row=0, column=0, padx=4, pady=2)
        ttk.Button(legend_frame, text="Rename Legends", command=self.configure_labels).grid(row=0, column=1, padx=4, pady=2)
        ttk.Button(legend_frame, text="Legend Order", command=self.configure_order).grid(row=0, column=2, padx=4, pady=2)
        ttk.Checkbutton(legend_frame, text="Auto Legend", variable=self.auto_legend).grid(row=1, column=0, padx=4, pady=2, sticky='w')
        ttk.Button(legend_frame, text="Add Legend", command=self.add_legend).grid(row=1, column=1, padx=4)
        ttk.Button(legend_frame, text="Remove Legend", command=self.remove_legend).grid(row=1, column=2, padx=4)

        ttk.Label(legend_frame, text="Legend loc:").grid(row=2, column=0, sticky='e', padx=2)
        legend_opts = ['best','upper right','upper left','lower right','lower left','right','center left',
                       'center right','lower center','upper center','center']
        self.legend_loc_cb = ttk.Combobox(legend_frame, values=legend_opts, state='readonly', width=12)
        self.legend_loc_cb.current(0)
        self.legend_loc_cb.grid(row=2, column=1, padx=2, sticky='w')

        # Font picker
        ttk.Label(legend_frame, text="Font:").grid(row=3, column=0, padx=2, sticky='e')
        common_fonts = ["Calibri", "Cambria", "Georgia", "DejaVu Sans", "Arial", "Times New Roman",
                        "Courier New", "Liberation Sans", "Verdana", "Segoe UI", "Tahoma"]
        installed = {f.name for f in fm.fontManager.ttflist}
        font_choices = [f for f in common_fonts if f in installed] or sorted(list(installed))[:12]
        self.font_cb = ttk.Combobox(legend_frame, values=font_choices, textvariable=self.font_family_var, state='readonly', width=18)
        if self.font_family_var.get() not in font_choices:
            self.font_family_var.set(font_choices[0])
        self.font_cb.set(self.font_family_var.get())
        self.font_cb.grid(row=3, column=1, padx=2, sticky='w')
        self.font_cb.bind('<<ComboboxSelected>>', lambda e: self.update_plot())
        self.font_family_var.trace_add('write', lambda *args: self.update_plot())
        self.font_display = ttk.Label(legend_frame, textvariable=self.font_family_var)
        self.font_display.grid(row=3, column=2, padx=4)
        ttk.Label(legend_frame, text="Legend font size:").grid(row=4, column=0, padx=2, sticky='e')
        self.legend_font = ttk.Entry(legend_frame, width=4); self.legend_font.insert(0,'10'); self.legend_font.grid(row=4, column=1, padx=2)

        # ---------- Detector & Axis ----------
        det_axis_frame = ttk.LabelFrame(top_inner, text="Detector & Axis", padding=6)
        det_axis_frame.grid(row=0, column=3, sticky='nw', padx=4, pady=2)

        ttk.Label(det_axis_frame, text="Detector:").grid(row=0, column=0, padx=2, sticky='e')
        self.detector_cb = ttk.Combobox(det_axis_frame, values=['RI','Light Scattering','Viscometry'], state='readonly', width=12)
        self.detector_cb.current(0); self.detector_cb.grid(row=0, column=1, padx=2)

        self.plot_type_var = tk.StringVar(value="Chromatogram")
        ttk.Label(det_axis_frame, text="Plot Type:").grid(row=0, column=4, padx=6, sticky='e')
        self.plot_type_cb = ttk.Combobox(
            det_axis_frame,
            values=["Chromatogram", "Mass Distribution (Signal)", "Mass Distribution (Integral %)"],
            state='readonly', width=26, textvariable=self.plot_type_var
        )
        self.plot_type_cb.grid(row=0, column=5, padx=2, sticky='w')
        self.plot_type_cb.bind('<<ComboboxSelected>>', lambda e: self.update_plot())

        ttk.Label(det_axis_frame, text="X-min:").grid(row=1, column=0, padx=2, sticky='e')
        self.xmin = ttk.Entry(det_axis_frame, width=6); self.xmin.insert(0,'0'); self.xmin.grid(row=1, column=1, padx=2)
        ttk.Label(det_axis_frame, text="X-max:").grid(row=1, column=2, padx=2, sticky='e')
        self.xmax = ttk.Entry(det_axis_frame, width=6); self.xmax.insert(0,'10'); self.xmax.grid(row=1, column=3, padx=2)

        self.logx_mass_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(det_axis_frame, text="Log₁₀(M)", variable=self.logx_mass_var,
                        command=self.update_plot).grid(row=1, column=5, padx=2, sticky='w')

        # Baseline extension controls
        self.baseline_toggle_btn = ttk.Button(det_axis_frame, text="Baseline: On", command=self.toggle_baseline)
        self.baseline_toggle_btn.grid(row=2, column=0, padx=4, pady=(6,2), sticky='w')

        ttk.Label(det_axis_frame, text="Extend ± (x units):").grid(row=2, column=1, padx=2, sticky='e')
        self.baseline_scale = ttk.Scale(det_axis_frame, from_=0.0, to=10.0, orient='horizontal',
                                        variable=self.baseline_len_var, command=lambda *_: self.update_plot())
        self.baseline_scale.grid(row=2, column=2, padx=2, sticky='we')
        det_axis_frame.grid_columnconfigure(2, weight=1)

        self.baseline_spin = ttk.Spinbox(det_axis_frame, from_=0.0, to=100.0, increment=0.1,
                                         textvariable=self.baseline_len_var, width=6, command=self.update_plot)
        self.baseline_spin.grid(row=2, column=3, padx=2, sticky='w')

        ttk.Label(det_axis_frame, text="Baseline ref:").grid(row=3, column=0, padx=4, sticky='e')
        self.baseline_mode_cb = ttk.Combobox(det_axis_frame, values=["X-limits", "Data range"],
                                             textvariable=self.baseline_mode, state='readonly', width=12)
        self.baseline_mode_cb.grid(row=3, column=1, padx=4, sticky='w')
        self.baseline_mode_cb.bind('<<ComboboxSelected>>', lambda e: self.update_plot())

        # Taper controls
        ttk.Checkbutton(det_axis_frame, text="Taper to zero", variable=self.taper_var,
                        command=self.update_plot).grid(row=4, column=0, padx=4, sticky='w')
        ttk.Label(det_axis_frame, text="Taper length (x units):").grid(row=4, column=1, padx=2, sticky='e')
        self.taper_scale = ttk.Scale(det_axis_frame, from_=0.0, to=2.0, orient='horizontal',
                                     variable=self.taper_len_var, command=lambda *_: self.update_plot())
        self.taper_scale.grid(row=4, column=2, padx=2, sticky='we')
        self.taper_spin = ttk.Spinbox(det_axis_frame, from_=0.0, to=10.0, increment=0.1,
                                      textvariable=self.taper_len_var, width=6, command=self.update_plot)
        self.taper_spin.grid(row=4, column=3, padx=2, sticky='w')

        # Smoothing sigma
        ttk.Label(det_axis_frame, text="Smoothing σ:").grid(row=5, column=0, padx=2, sticky='e')
        self.sigma_spin = ttk.Spinbox(det_axis_frame, from_=0.0, to=10.0, increment=0.1,
                                      textvariable=self.sigma_var, width=6, command=self.update_plot)
        self.sigma_spin.grid(row=5, column=1, padx=2, sticky='w')

        # ---------- Display Options ----------
        display_frame = ttk.LabelFrame(top_inner, text="Display Options", padding=6)
        display_frame.grid(row=0, column=4, sticky='nw', padx=4, pady=2)
        ttk.Checkbutton(display_frame, text="Inline labels", variable=self.inline_var).grid(row=0, column=0, padx=4, sticky='w')
        ttk.Checkbutton(display_frame, text="Use Gradient", variable=self.gradient_var).grid(row=0, column=1, padx=4, sticky='w')
        ttk.Checkbutton(display_frame, text="Use Math ($…$)", variable=self.use_mathtext,
                        command=self.update_plot).grid(row=0, column=2, padx=4, sticky='w')
        ttk.Label(display_frame, text="Gradient Start:").grid(row=1, column=0, padx=2, sticky='e')
        ttk.Button(display_frame, text="Set", command=self.pick_gradient_start).grid(row=1, column=1, padx=2)
        ttk.Label(display_frame, text="Gradient End:").grid(row=1, column=2, padx=2, sticky='e')
        ttk.Button(display_frame, text="Set", command=self.pick_gradient_end).grid(row=1, column=3, padx=2)
        ttk.Checkbutton(display_frame, text="Hide X-axis", variable=self.hide_x_var).grid(row=2, column=0, padx=4, sticky='w')
        ttk.Checkbutton(display_frame, text="Hide Y-axis", variable=self.hide_y_var).grid(row=2, column=1, padx=4, sticky='w')
        ttk.Label(display_frame, text="X Axis Title:").grid(row=3, column=0, padx=2, sticky='e')
        ttk.Entry(display_frame, textvariable=self.xlabel_var, width=15).grid(row=3, column=1, padx=2)
        ttk.Label(display_frame, text="Y Axis Title:").grid(row=3, column=2, padx=2, sticky='e')
        ttk.Entry(display_frame, textvariable=self.ylabel_var, width=15).grid(row=3, column=3, padx=2)
        ttk.Label(display_frame, text="Axis Font Size:").grid(row=4, column=0, padx=2, sticky='e')
        ttk.Spinbox(display_frame, from_=6, to=30, textvariable=self.axis_fontsize_var, width=5).grid(row=4, column=1, padx=2, sticky='w')

        # ---------- Molecules (optional overlay) ----------
        mol_frame = ttk.LabelFrame(top_inner, text="Molecules", padding=6)
        mol_frame.grid(row=0, column=5, sticky='nw', padx=4, pady=2)
        ttk.Button(mol_frame, text="Add SMILES File", command=self.add_smiles_file).grid(row=0, column=0, padx=4)
        self.mol_overlay = tk.Canvas(root, width=300, height=200, bg=root['bg'], highlightthickness=0)
        self.mol_overlay.place(relx=0.7, rely=0.3)
        self.molecule_widgets = []

        # ---------- Actions ----------
        action_frame = ttk.LabelFrame(top_inner, text="Actions", padding=6)
        action_frame.grid(row=0, column=6, sticky='nw', padx=4, pady=2)
        ttk.Button(action_frame, text="Plot", command=self.update_plot).grid(row=0, column=0, padx=6, pady=4)
        ttk.Button(action_frame, text="Save Figure", command=self.save_plot).grid(row=0, column=1, padx=6, pady=4)
        ttk.Button(action_frame, text="Exit", command=self.root.destroy).grid(row=0, column=2, padx=6, pady=4)
        ttk.Button(action_frame, text="Save Session", command=self.save_session).grid(row=1, column=0, padx=6, pady=4, sticky='w')
        ttk.Button(action_frame, text="Load Session", command=self.load_session).grid(row=1, column=1, padx=6, pady=4, sticky='w')

        # ---------- Figure ----------
        self.fig, self.ax = plt.subplots()
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().pack(side='top', fill='both', expand=True, padx=6, pady=4)

        if not self.legend_order:
            self.legend_order = [f for f in sorted(self.file_vars.keys()) if self.file_vars.get(f, tk.BooleanVar(value=False)).get()]
        self._push_history()
        self.update_plot()

    # ===== helpers =====
    def toggle_baseline(self):
        self.extend_baseline_var.set(not self.extend_baseline_var.get())
        self.baseline_toggle_btn.config(text=f"Baseline: {'On' if self.extend_baseline_var.get() else 'Off'}")
        self.update_plot()

    def pick_gradient_start(self):
        c = colorchooser.askcolor(title="Gradient start color", initialcolor=self.gradient_start_color, parent=self.root)
        if c and c[1]:
            self.gradient_start_color = c[1]

    def pick_gradient_end(self):
        c = colorchooser.askcolor(title="Gradient end color", initialcolor=self.gradient_end_color, parent=self.root)
        if c and c[1]:
            self.gradient_end_color = c[1]

    def add_smiles_file(self):
        fp = filedialog.askopenfilename(
            title="Select molecule file",
            filetypes=[("Molecule/SMILES/Mol/SDF", "*.smi *.txt *.mol *.sdf"), ("All", "*.*")]
        )
        if not fp:
            return
        ext = os.path.splitext(fp)[1].lower()
        display_text = os.path.basename(fp)
        mol = None
        try:
            if ext in ('.smi', '.txt'):
                with open(fp, 'r') as f:
                    line = f.readline().strip()
                display_text = line
                if RDKit_AVAILABLE:
                    mol = Chem.MolFromSmiles(line)
            elif ext == '.mol':
                if RDKit_AVAILABLE:
                    mol = Chem.MolFromMolFile(fp, sanitize=True)
            elif ext == '.sdf':
                if RDKit_AVAILABLE:
                    suppl = Chem.SDMolSupplier(fp)
                    mols = [m for m in suppl if m is not None]
                    mol = mols[0] if mols else None
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load molecule: {e}")
            return
        if RDKit_AVAILABLE and mol is not None:
            try:
                img = Draw.MolToImage(mol, size=(200, 150))
                photo = ImageTk.PhotoImage(img)
                DraggableMolecule(self.mol_overlay, image=photo, label=Chem.MolToSmiles(mol))
                self.add_molecule_to_axes(mol)
            except Exception:
                lbl = tk.Label(self.mol_overlay, text=display_text or "Molecule", bg='white', bd=1, relief='solid')
                DraggableMolecule(self.mol_overlay, widget=lbl, label=display_text)
        else:
            lbl = tk.Label(self.mol_overlay, text=display_text, bg='white', bd=1, relief='solid')
            DraggableMolecule(self.mol_overlay, widget=lbl, label=display_text)

    def add_molecule_to_axes(self, mol):
        try:
            img = Draw.MolToImage(mol, size=(150, 150))
            oi = OffsetImage(img)
            ab = AnnotationBbox(oi, (0.5, 0.5), xycoords='axes fraction', frameon=True)
            self.ax.add_artist(ab)
            self.canvas.draw()
        except Exception as e:
            print(f"[DEBUG] Failed to add molecule to axes: {e}")

    def _capture_state(self):
        return {
            "show_title": self.show_title.get(),
            "custom_title": self.custom_title_var.get(),
            "auto_legend": self.auto_legend.get(),
            "legend_loc": self.legend_loc_cb.get() if hasattr(self, 'legend_loc_cb') else 'best',
            "legend_font": self.legend_font.get() if hasattr(self, 'legend_font') else '10',
        }

    def _apply_state(self, state):
        self.show_title.set(state.get("show_title", False))
        self.custom_title_var.set(state.get("custom_title", ""))
        self.auto_legend.set(state.get("auto_legend", True))
        self.legend_loc_cb.set(state.get("legend_loc", "best"))
        self.legend_font.delete(0, tk.END)
        self.legend_font.insert(0, state.get("legend_font", "10"))

    def _push_history(self):
        if self._hist_index < len(self._history) - 1:
            self._history = self._history[: self._hist_index + 1]
        self._history.append(self._capture_state())
        self._hist_index = len(self._history) - 1

    def undo(self):
        if self._hist_index <= 0:
            return
        self._hist_index -= 1
        self._apply_state(self._history[self._hist_index])
        self.update_plot()

    def redo(self):
        if self._hist_index >= len(self._history) - 1:
            return
        self._hist_index += 1
        self._apply_state(self._history[self._hist_index])
        self.update_plot()

    def browse_folder(self):
        d = filedialog.askdirectory(title="Select folder with data")
        if d:
            self.folder_var.set(d)
            self.rebuild_file_list()

    def rebuild_file_list(self):
        folder = self.folder_var.get()
        if not folder or not os.path.isdir(folder):
            return
        existing = dict(self.file_vars)
        self.file_vars.clear()
        self.file_labels.clear()
        self.file_colors.clear()
        for widget in self.file_list_inner.winfo_children():
            widget.destroy()
        for idx, fname in enumerate(sorted(os.listdir(folder))):
            if fname.lower().endswith(('.txt', '.dat', '.csv')):
                full = os.path.join(folder, fname)
                prev = existing.get(full)
                self.file_vars[full] = tk.BooleanVar(value=prev.get() if prev else True)
                self.file_labels[full] = os.path.basename(full)
                self.file_colors[full] = None
                cb = ttk.Checkbutton(self.file_list_inner, text=os.path.basename(full), variable=self.file_vars[full], command=self.update_plot)
                cb.grid(row=idx, column=0, sticky='w', padx=2, pady=1)
        if not self.legend_order:
            self.legend_order = [f for f in sorted(self.file_vars.keys()) if self.file_vars[f].get()]
        else:
            existing_order = [f for f in self.legend_order if f in self.file_vars]
            new_items = [f for f in sorted(self.file_vars.keys()) if f not in existing_order and self.file_vars[f].get()]
            self.legend_order = existing_order + new_items
        self.update_plot()

    # ---- helper for cosine taper ----
    def _cos_taper(self, x0, y0, x1, y1, n=60):
        xs = np.linspace(x0, x1, max(2, n))
        t = (xs - x0) / max(1e-12, (x1 - x0))
        ys = y0 + (y1 - y0) * 0.5 * (1 - np.cos(np.pi * t))
        return xs.tolist(), ys.tolist()

    # ===== Unicode / mathtext converters =====
    _SUP = str.maketrans("0123456789+-=()", "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾")
    _SUB = str.maketrans("0123456789+-=()n", "₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎ₙ")

    @staticmethod
    def _to_unicode_scripts(s: str) -> str:
        """Convert e.g. X_55->X₅₅, Mw^2->Mw². Removes any '$'."""
        txt = s.replace("$", "")

        def sub_sub(m): return f"{m.group(1)}{m.group(2).translate(GPCApp._SUB)}"
        def sub_sup(m): return f"{m.group(1)}{m.group(2).translate(GPCApp._SUP)}"
        txt = re.sub(r"([A-Za-z\\][A-Za-z0-9\\]*)_\{([^}]+)\}", sub_sub, txt)
        txt = re.sub(r"([A-Za-z\\][A-Za-z0-9\\]*)\^{([^}]+)\}", sub_sup, txt)

        txt = re.sub(r"([A-Za-z\\][A-Za-z0-9\\]*)_(\d+|n)", lambda m: m.group(1) + m.group(2).translate(GPCApp._SUB), txt)
        txt = re.sub(r"([A-Za-z\\][A-Za-z0-9\\]*)\^(\d+)", lambda m: m.group(1) + m.group(2).translate(GPCApp._SUP), txt)
        return txt

    @staticmethod
    def _auto_mathify_text(core: str) -> str:
        """Sanitize for mathtext mode."""
        core = core.replace(" ^", "^").replace("^ ", "^")
        core = core.replace(" _", "_").replace("_ ", "_")
        def sub_sub(m): return f"{m.group(1)}_{{{m.group(2)}}}"
        def sub_sup(m): return f"{m.group(1)}^{{{m.group(2)}}}"
        core = re.sub(r"([A-Za-z\\][A-Za-z0-9\\]*)_(\w+(?:\.\w+)*)", sub_sub, core)
        core = re.sub(r"([A-Za-z\\][A-Za-z0-9\\]*)\^(\w+(?:\.\w+)*)", sub_sup, core)
        return core

    def _auto_mathify(self, s: str) -> str:
        """
        If math mode ON: wrap as $…$ after sanitizing.
        If OFF: return Unicode-sub/sup version so normal fonts apply (e.g., Calibri).
        """
        if not s:
            return s
        base = s.strip()
        if self.use_mathtext.get():
            core = base.replace("$", "").strip()
            if not core:
                return ""
            core = self._auto_mathify_text(core)
            return f"${core}$"
        else:
            return self._to_unicode_scripts(base)

    def update_plot(self):
        self.ax.clear()
        self._legend_handles = []
        self._legend_labels = []

        detector = self.detector_cb.get()
        xmin = float(self.xmin.get()) if self.xmin.get() else None
        xmax = float(self.xmax.get()) if self.xmax.get() else None
        sigma = float(self.sigma_var.get())
        fontname = self.font_family_var.get()

        # --- Force Calibri-first everywhere, with sane fallbacks ---
        # Try to pin the exact Calibri font file if available
        try:
            cal_path = fm.findfont(FontProperties(family=fontname, style='normal', weight='normal'),
                                   fallback_to_default=False)
            use_fname = os.path.basename(cal_path).lower().startswith('calibri')
        except Exception:
            cal_path = None
            use_fname = False

        mpl.rcParams['font.family'] = [fontname, 'Segoe UI Symbol', 'DejaVu Sans']
        mpl.rcParams['font.style'] = 'normal'
        mpl.rcParams['text.usetex'] = False
        mpl.rcParams['mathtext.default'] = 'regular'

        if use_fname and os.path.exists(cal_path):
            fp_main = FontProperties(fname=cal_path, size=self.axis_fontsize_var.get(),
                                     style='normal', weight='normal')
        else:
            fp_main = FontProperties(family=[fontname, 'Segoe UI Symbol', 'DejaVu Sans'],
                                     size=self.axis_fontsize_var.get(), style='normal', weight='normal')

        mode = self.plot_type_var.get()
        mass_mode = mode.startswith("Mass Distribution")

        traces = []
        data_min, data_max = None, None

        def _safe_set_xlim(left, right):
            """Apply x-limits only if finite; otherwise fall back to autoscale."""
            left = left if left is None or np.isfinite(left) else None
            right = right if right is None or np.isfinite(right) else None
            if left is None and right is None:
                self.ax.relim()
                self.ax.autoscale_view(scalex=True, scaley=False)
                return False
            self.ax.set_xlim(left=left, right=right)
            return True

        if mass_mode:
            for f in self.legend_order:
                var = self.file_vars.get(f)
                if not var or not var.get():
                    continue
                M, sig_md, integ_md = load_mass_distribution(f)
                if M is None:
                    continue

                if "Integral" in mode:
                    y = integ_md
                else:
                    y = sig_md

                xvals = np.asarray(M, dtype=float)
                y = np.asarray(y, dtype=float)

                finite = np.isfinite(xvals) & np.isfinite(y)
                if self.logx_mass_var.get():
                    finite &= xvals > 0

                if not np.any(finite):
                    continue

                xvals = xvals[finite]
                y = y[finite]

                order = np.argsort(xvals)
                xvals = xvals[order]
                y = y[order]

                if self.logx_mass_var.get():
                    x_for_smooth = np.log10(np.clip(xvals, a_min=np.finfo(float).tiny, a_max=None))
                else:
                    x_for_smooth = xvals

                y = smooth_series(x_for_smooth, y, sigma)
                y = np.clip(y, 0.0, None)

                if self.normalize_var.get():
                    if "Integral" in mode:
                        if y.size:
                            maxv = float(np.nanmax(y))
                            if np.isfinite(maxv) and maxv > 0:
                                y = y / maxv
                    else:
                        xvals, y = compute_weight_fraction(xvals, y)
                        if not len(xvals):
                            continue
                        y = np.clip(y, 0.0, None)
                        if y.size:
                            peak = float(np.nanmax(y))
                            if np.isfinite(peak) and peak > 0:
                                y = y / peak
                elif len(y):
                    maxv = float(np.nanmax(y))
                    if np.isfinite(maxv) and maxv > 0:
                        y = y / maxv

                if self.logx_mass_var.get():
                    plot_x = np.log10(np.clip(xvals, a_min=np.finfo(float).tiny, a_max=None))
                else:
                    plot_x = xvals

                if plot_x.size == 0 or y.size == 0:
                    continue

                tmin = float(np.nanmin(plot_x))
                tmax = float(np.nanmax(plot_x))
                if not (np.isfinite(tmin) and np.isfinite(tmax)):
                    continue

                label_raw = self.file_labels.get(f, f)
                label = self._auto_mathify(label_raw)
                traces.append({"x": plot_x, "y": y, "label": label, "color": self.file_colors.get(f, None)})
                data_min = tmin if data_min is None else min(data_min, tmin)
                data_max = tmax if data_max is None else max(data_max, tmax)

            if not traces:
                self.canvas.draw()
                return

            if self.baseline_mode.get() == "X-limits":
                ref_left = xmin if xmin is not None else data_min
                ref_right = xmax if xmax is not None else data_max
            else:
                ref_left, ref_right = data_min, data_max
            if ref_right < ref_left:
                ref_left, ref_right = ref_right, ref_left

            ext = float(self.baseline_len_var.get() or 0.0)
            ext_left, ext_right = ref_left - ext, ref_right + ext

            if self.extend_baseline_var.get():
                x_left_final = xmin if xmin is not None else ext_left
                x_right_final = xmax if xmax is not None else ext_right
            else:
                x_left_final = xmin
                x_right_final = xmax

            taper_on = self.taper_var.get()
            taper_len = max(0.0, float(self.taper_len_var.get() or 0.0))

            for tr in traces:
                plot_x = tr["x"]
                plot_y = tr["y"]
                label = tr["label"]
                color = tr["color"]

                if not self.extend_baseline_var.get():
                    line, = self.ax.plot(plot_x, plot_y, color=color)
                    line.set_label(label)
                    self._legend_handles.append(line)
                    self._legend_labels.append(label)
                    continue

                left_data = float(plot_x[0])
                right_data = float(plot_x[-1])
                y_left = float(plot_y[0])
                y_right = float(plot_y[-1])

                xplot = []
                yplot = []

                if taper_on:
                    left_taper = min(
                        taper_len,
                        max(0.0, left_data - (x_left_final if x_left_final is not None else left_data))
                    )
                else:
                    left_taper = 0.0
                left_zero_end = left_data - left_taper

                if x_left_final is not None and x_left_final < left_zero_end:
                    xplot.extend([x_left_final, left_zero_end]); yplot.extend([0.0, 0.0])

                if left_taper > 0:
                    xs, ys = self._cos_taper(left_zero_end, 0.0, left_data, y_left)
                    if xs and xplot and xs[0] == xplot[-1]:
                        xs = xs[1:]; ys = ys[1:]
                    xplot.extend(xs); yplot.extend(ys)

                xplot.extend(plot_x.tolist()); yplot.extend(plot_y.tolist())

                if taper_on:
                    right_taper = min(
                        taper_len,
                        max(0.0, (x_right_final if x_right_final is not None else right_data) - right_data)
                    )
                else:
                    right_taper = 0.0
                right_zero_start = right_data + right_taper

                if right_taper > 0:
                    xs, ys = self._cos_taper(right_data, y_right, right_zero_start, 0.0)
                    if xs and xplot and xs[0] == xplot[-1]:
                        xs = xs[1:]; ys = ys[1:]
                    xplot.extend(xs); yplot.extend(ys)

                if x_right_final is not None and right_zero_start < x_right_final:
                    xplot.extend([right_zero_start, x_right_final]); yplot.extend([0.0, 0.0])

                line, = self.ax.plot(xplot, yplot, color=color)
                line.set_label(label)
                self._legend_handles.append(line)
                self._legend_labels.append(label)

            if self.hide_x_var.get():
                self.ax.get_xaxis().set_visible(False)
            if self.hide_y_var.get():
                self.ax.get_yaxis().set_visible(False)

            xlab = "log\u2081\u2080(M) [g/mol]" if self.logx_mass_var.get() else "Molar mass [g/mol]"
            xlabel = self._auto_mathify(self.xlabel_var.get().strip()) or xlab
            ylabel = self._auto_mathify(self.ylabel_var.get().strip()) or \
                     ("Integral [%]" if "Integral" in mode else "Distribution (arb.)")
            self.ax.set_xlabel(xlabel, fontproperties=fp_main)
            self.ax.set_ylabel(ylabel, fontproperties=fp_main)

            for lbl in self.ax.get_xticklabels() + self.ax.get_yticklabels():
                lbl.set_fontproperties(fp_main)

            if self.auto_legend.get():
                self._apply_legend()

            if self.extend_baseline_var.get() and (x_left_final is not None or x_right_final is not None):
                _safe_set_xlim(x_left_final, x_right_final)
            if xmin is not None or xmax is not None:
                _safe_set_xlim(xmin, xmax)

        else:
            for f in self.legend_order:
                var = self.file_vars.get(f)
                if not var or not var.get():
                    continue
                times, vals = load_and_smooth(f, detector, sigma=sigma)
                if times is None or len(times) == 0:
                    continue
                if self.normalize_var.get():
                    maxv = vals.max() if len(vals) else 1.0
                    if maxv != 0:
                        vals = vals / maxv
                label_raw = self.file_labels.get(f, f)
                label = self._auto_mathify(label_raw)
                traces.append({"times": times, "vals": vals,
                               "label": label, "color": self.file_colors.get(f, None)})

            if not traces:
                self.canvas.draw()
                return

            data_min = min(float(tr["times"].min()) for tr in traces)
            data_max = max(float(tr["times"].max()) for tr in traces)

            if self.baseline_mode.get() == "X-limits":
                ref_left  = xmin if xmin is not None else data_min
                ref_right = xmax if xmax is not None else data_max
            else:
                ref_left, ref_right = data_min, data_max
            if ref_right < ref_left:
                ref_left, ref_right = ref_right, ref_left

            ext = float(self.baseline_len_var.get() or 0.0)
            ext_left, ext_right = ref_left - ext, ref_right + ext

            if self.extend_baseline_var.get():
                x_left_final = xmin if xmin is not None else ext_left
                x_right_final = xmax if xmax is not None else ext_right
            else:
                x_left_final = xmin
                x_right_final = xmax

            taper_on = self.taper_var.get()
            taper_len = max(0.0, float(self.taper_len_var.get() or 0.0))

            for tr in traces:
                times = tr["times"]; vals = tr["vals"]
                label = tr["label"]; color = tr["color"]

                if not self.extend_baseline_var.get():
                    line, = self.ax.plot(times, vals, color=color)
                    line.set_label(label)
                    self._legend_handles.append(line)
                    self._legend_labels.append(label)
                    continue

                left_data = float(times.min())
                right_data = float(times.max())
                y_left = float(vals[0])
                y_right = float(vals[-1])

                xplot = []
                yplot = []

                left_taper = min(taper_len, max(0.0, left_data - (x_left_final if x_left_final is not None else left_data))) if taper_on else 0.0
                left_zero_end = left_data - left_taper

                if x_left_final is not None and x_left_final < left_zero_end:
                    xplot.extend([x_left_final, left_zero_end]); yplot.extend([0.0, 0.0])

                if left_taper > 0:
                    xs, ys = self._cos_taper(left_zero_end, 0.0, left_data, y_left)
                    if xs and xplot and xs[0] == xplot[-1]:
                        xs = xs[1:]; ys = ys[1:]
                    xplot.extend(xs); yplot.extend(ys)

                xplot.extend(times.tolist()); yplot.extend(vals.tolist())

                right_taper = min(taper_len, max(0.0, (x_right_final if x_right_final is not None else right_data) - right_data)) if taper_on else 0.0
                right_zero_start = right_data + right_taper

                if right_taper > 0:
                    xs, ys = self._cos_taper(right_data, y_right, right_zero_start, 0.0)
                    if xs and xplot and xs[0] == xplot[-1]:
                        xs = xs[1:]; ys = ys[1:]
                    xplot.extend(xs); yplot.extend(ys)

                if x_right_final is not None and right_zero_start < x_right_final:
                    xplot.extend([right_zero_start, x_right_final]); yplot.extend([0.0, 0.0])

                line, = self.ax.plot(xplot, yplot, color=color)
                line.set_label(label)
                self._legend_handles.append(line)
                self._legend_labels.append(label)

            if self.hide_x_var.get():
                self.ax.get_xaxis().set_visible(False)
            if self.hide_y_var.get():
                self.ax.get_yaxis().set_visible(False)

            xlabel = self._auto_mathify(self.xlabel_var.get().strip())
            ylabel = self._auto_mathify(self.ylabel_var.get().strip())
            self.ax.set_xlabel(xlabel if xlabel else "", fontproperties=fp_main)
            self.ax.set_ylabel(ylabel if ylabel else "", fontproperties=fp_main)

            for lbl in self.ax.get_xticklabels() + self.ax.get_yticklabels():
                lbl.set_fontproperties(fp_main)

            if self.inline_var.get():
                for line, label in zip(self._legend_handles, self._legend_labels):
                    xdata = line.get_xdata(); ydata = line.get_ydata()
                    if len(xdata) and len(ydata):
                        self.ax.text(xdata[-1], ydata[-1], label, fontproperties=fp_main)

            if self.auto_legend.get():
                self._apply_legend()

            if x_left_final is not None or x_right_final is not None:
                _safe_set_xlim(x_left_final, x_right_final)
            if xmin is not None or xmax is not None:
                _safe_set_xlim(xmin, xmax)

        if self.show_title.get():
            custom = self.custom_title_var.get().strip()
            title_base = f"{mode}"
            if not mass_mode:
                title_base = f"Detector: {detector} (σ={sigma:.1f})"
                if self.normalize_var.get():
                    title_base += " (normalized)"
            elif self.normalize_var.get():
                title_base += " (normalized)"
>>>>>>> Stashed changes
