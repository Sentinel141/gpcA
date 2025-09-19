#!/usr/bin/env python3
import os, re, json, tkinter as tk
from tkinter import filedialog, messagebox, colorchooser
import tkinter.ttk as ttk
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties, fontManager
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# ================= Shared helpers =================
def _font_choices():
    installed = sorted({f.name for f in fontManager.ttflist})
    prefer = ["Calibri","Cambria","Georgia","DejaVu Sans","Arial","Times New Roman","Verdana","Segoe UI","Tahoma"]
    return [f for f in prefer if f in installed] or installed[:12]

def _default_font():
    installed = {f.name for f in fontManager.ttflist}
    for name in ["Calibri","DejaVu Sans","Arial","Segoe UI","Verdana"]:
        if name in installed: return name
    return sorted(installed)[0] if installed else "DejaVu Sans"

# ================= Elugram (unchanged behavior) =================
from scipy.ndimage import gaussian_filter1d
def load_and_smooth(path, detector, sigma=1):
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
    smoothed = gaussian_filter1d(ser[valid].to_numpy(), sigma=sigma)
    return times[valid].to_numpy(), smoothed

class ElugramPanel(ttk.Frame):
    def __init__(self, master):
        super().__init__(master)
        self.root = master.winfo_toplevel()
        # --- state ---
        self.file_vars, self.file_labels, self.file_colors = {}, {}, {}
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
        self.font_family_var = tk.StringVar(value=_default_font())
        self.extend_baseline_var = tk.BooleanVar(value=True)
        self.baseline_len_var = tk.DoubleVar(value=2.0)
        self.baseline_mode = tk.StringVar(value="X-limits")
        self.taper_var = tk.BooleanVar(value=True)
        self.taper_len_var = tk.DoubleVar(value=0.3)
        self.sigma_var = tk.DoubleVar(value=1.0)
        self.use_mathtext = tk.BooleanVar(value=True)

        container = ttk.Frame(self); container.pack(side='top', fill='x', padx=6, pady=4)
        h_canvas = tk.Canvas(container, height=220)
        h_scroll = ttk.Scrollbar(container, orient='horizontal', command=h_canvas.xview)
        h_canvas.configure(xscrollcommand=h_scroll.set); h_canvas.pack(side='top', fill='x', expand=True)
        h_scroll.pack(side='top', fill='x')
        top_inner = ttk.Frame(h_canvas); h_canvas.create_window((0,0), window=top_inner, anchor='nw')
        top_inner.bind("<Configure>", lambda e: h_canvas.configure(scrollregion=h_canvas.bbox("all")))

        # Folder
        folder_frame = ttk.LabelFrame(top_inner, text="Folder", padding=6); folder_frame.grid(row=0, column=0, sticky='nw', padx=4, pady=2)
        self.folder_var = tk.StringVar()
        ttk.Label(folder_frame, text="Folder:").grid(row=0, column=0, sticky='w')
        ttk.Entry(folder_frame, textvariable=self.folder_var, width=30).grid(row=0, column=1, padx=4)
        ttk.Button(folder_frame, text="Browse", command=self.browse_folder).grid(row=0, column=2, padx=4)
        ttk.Button(folder_frame, text="Refresh Files", command=self.rebuild_file_list).grid(row=0, column=3, padx=4)

        # Files
        self.files_frame = ttk.LabelFrame(top_inner, text="Files", padding=6); self.files_frame.grid(row=0, column=1, sticky='nw', padx=4, pady=2)
        self.file_list_canvas = tk.Canvas(self.files_frame, width=250, height=120)
        self.file_list_scroll = ttk.Scrollbar(self.files_frame, orient='vertical', command=self.file_list_canvas.yview)
        self.file_list_inner = ttk.Frame(self.file_list_canvas)
        self.file_list_inner.bind("<Configure>", lambda e: self.file_list_canvas.configure(scrollregion=self.file_list_canvas.bbox("all")))
        self.file_list_canvas.create_window((0,0), window=self.file_list_inner, anchor='nw')
        self.file_list_canvas.configure(yscrollcommand=self.file_list_scroll.set)
        self.file_list_canvas.grid(row=0, column=0, sticky='nsew')
        self.file_list_scroll.grid(row=0, column=1, sticky='ns')

        # Legend & Colors
        legend_frame = ttk.LabelFrame(top_inner, text="Legend & Colors", padding=6); legend_frame.grid(row=0, column=2, sticky='nw', padx=4, pady=2)
        ttk.Button(legend_frame, text="Set Colors", command=self.configure_colors).grid(row=0, column=0, padx=4, pady=2)
        ttk.Button(legend_frame, text="Rename Legends", command=self.configure_labels).grid(row=0, column=1, padx=4, pady=2)
        ttk.Button(legend_frame, text="Legend Order", command=self.configure_order).grid(row=0, column=2, padx=4, pady=2)
        ttk.Checkbutton(legend_frame, text="Auto Legend", variable=self.auto_legend).grid(row=1, column=0, padx=4, pady=2, sticky='w')
        ttk.Button(legend_frame, text="Add Legend", command=self.add_legend).grid(row=1, column=1, padx=4)
        ttk.Button(legend_frame, text="Remove Legend", command=self.remove_legend).grid(row=1, column=2, padx=4)
        ttk.Label(legend_frame, text="Legend loc:").grid(row=2, column=0, sticky='e', padx=2)
        legend_opts = ['best','upper right','upper left','lower right','lower left','right','center left','center right','lower center','upper center','center']
        self.legend_loc_cb = ttk.Combobox(legend_frame, values=legend_opts, state='readonly', width=12); self.legend_loc_cb.current(0)
        self.legend_loc_cb.grid(row=2, column=1, padx=2, sticky='w')
        ttk.Label(legend_frame, text="Font:").grid(row=3, column=0, padx=2, sticky='e')
        self.font_cb = ttk.Combobox(legend_frame, values=_font_choices(), textvariable=self.font_family_var, state='readonly', width=18)
        self.font_cb.grid(row=3, column=1, padx=2, sticky='w')
        self.font_cb.bind('<<ComboboxSelected>>', lambda e: self.update_plot())
        self.font_display = ttk.Label(legend_frame, textvariable=self.font_family_var); self.font_display.grid(row=3, column=2, padx=4)
        ttk.Label(legend_frame, text="Legend font size:").grid(row=4, column=0, padx=2, sticky='e')
        self.legend_font = ttk.Entry(legend_frame, width=4); self.legend_font.insert(0,'10'); self.legend_font.grid(row=4, column=1, padx=2)

        # Detector & axis
        det_axis_frame = ttk.LabelFrame(top_inner, text="Detector & Axis", padding=6); det_axis_frame.grid(row=0, column=3, sticky='nw', padx=4, pady=2)
        ttk.Label(det_axis_frame, text="Detector:").grid(row=0, column=0, padx=2, sticky='e')
        self.detector_cb = ttk.Combobox(det_axis_frame, values=['RI','Light Scattering','Viscometry'], state='readonly', width=12); self.detector_cb.current(0)
        self.detector_cb.grid(row=0, column=1, padx=2)
        ttk.Label(det_axis_frame, text="X-min:").grid(row=1, column=0, padx=2, sticky='e')
        self.xmin = ttk.Entry(det_axis_frame, width=6); self.xmin.insert(0,'0'); self.xmin.grid(row=1, column=1, padx=2)
        ttk.Label(det_axis_frame, text="X-max:").grid(row=1, column=2, padx=2, sticky='e')
        self.xmax = ttk.Entry(det_axis_frame, width=6); self.xmax.insert(0,'10'); self.xmax.grid(row=1, column=3, padx=2)

        # Actions
        action_frame = ttk.LabelFrame(top_inner, text="Actions", padding=6); action_frame.grid(row=0, column=4, sticky='nw', padx=4, pady=2)
        ttk.Button(action_frame, text="Plot", command=self.update_plot).grid(row=0, column=0, padx=6, pady=4)
        ttk.Button(action_frame, text="Save Figure", command=self.save_plot).grid(row=0, column=1, padx=6, pady=4)

        # Figure
        self.fig, self.ax = plt.subplots()
        self.canvas = FigureCanvasTkAgg(self.fig, master=self); self.canvas.get_tk_widget().pack(side='top', fill='both', expand=True, padx=6, pady=4)

        self.update_plot()

    # --- UI helpers cloned from original ---
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
        self.file_vars.clear(); self.file_labels.clear(); self.file_colors.clear()
        for widget in self.file_list_inner.winfo_children(): widget.destroy()
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

    def configure_colors(self):
        sel = [f for f, v in self.file_vars.items() if v.get()]
        if not sel: messagebox.showwarning("No Selection", "Select files first"); return
        cfg = tk.Toplevel(self.root); cfg.title("Set Colors")
        for idx, f in enumerate(sel):
            ttk.Label(cfg, text=os.path.basename(f)).grid(row=idx, column=0, padx=5, pady=2, sticky='w')
            btn = ttk.Button(cfg, text="Pick Color")
            btn.grid(row=idx, column=1, padx=5)
            btn.config(command=lambda f=f, b=btn: self._pick_file_color(f, b))
        ttk.Button(cfg, text="OK", command=lambda: (self.update_plot(), cfg.destroy())).grid(row=len(sel), column=0, columnspan=2, pady=10)

    def _pick_file_color(self, f, b):
        c = colorchooser.askcolor(parent=self.root)[1]
        if c:
            self.file_colors[f] = c
            self.update_plot()

    def configure_labels(self):
        sel = [f for f, v in self.file_vars.items() if v.get()]
        if not sel: messagebox.showwarning("No Selection", "Select files first"); return
        dlg = tk.Toplevel(self.root); dlg.title("Rename Legends")
        entries = {}
        for idx, f in enumerate(sel, start=1):
            ttk.Label(dlg, text=os.path.basename(f)).grid(row=idx, column=0, padx=6, pady=3, sticky='w')
            var = tk.StringVar(value=self.file_labels.get(f, os.path.basename(f)))
            ent = ttk.Entry(dlg, textvariable=var, width=40); ent.grid(row=idx, column=1, padx=6, pady=3, sticky='w')
            entries[f] = var
        ttk.Button(dlg, text="OK", command=lambda: (self._save_labels(entries), dlg.destroy(), self.update_plot())).grid(row=len(sel)+1, column=0, columnspan=2, pady=10)

    def _save_labels(self, entries):
        for f, var in entries.items(): self.file_labels[f] = var.get()

    def configure_order(self):
        sel = [f for f, v in self.file_vars.items() if v.get()]
        if not sel: messagebox.showwarning("No Selection", "Select files first"); return
        dlg = tk.Toplevel(self.root); dlg.title("Reorder Legend / Plot Order")
        ttk.Label(dlg, text="Use Up/Down to reorder:").grid(row=0, column=0, columnspan=3, pady=4)
        lb = tk.Listbox(dlg, height=10, width=50); lb.grid(row=1, column=0, columnspan=3, padx=5)
        ordered = [f for f in self.legend_order if f in sel] or sel
        for f in ordered: lb.insert("end", self.file_labels.get(f, os.path.basename(f)))
        def move_up():
            idxs = lb.curselection()
            if not idxs or idxs[0] == 0: return
            i = idxs[0]; txt = lb.get(i); lb.delete(i); lb.insert(i-1, txt); lb.select_set(i-1)
        def move_down():
            idxs = lb.curselection()
            if not idxs or idxs[0] == lb.size() - 1: return
            i = idxs[0]; txt = lb.get(i); lb.delete(i); lb.insert(i+1, txt); lb.select_set(i+1)
        ttk.Button(dlg, text="Up", command=move_up).grid(row=2, column=0, padx=4, pady=4)
        ttk.Button(dlg, text="Down", command=move_down).grid(row=2, column=1, padx=4, pady=4)
        def apply_order():
            label_to_file = {self.file_labels.get(f, os.path.basename(f)): f for f in sel}
            new_order = []
            for i in range(lb.size()):
                label = lb.get(i); f = label_to_file.get(label)
                if f and f not in new_order: new_order.append(f)
            for f in sel:
                if f not in new_order: new_order.append(f)
            self.legend_order = new_order; dlg.destroy(); self.update_plot()
        ttk.Button(dlg, text="Apply", command=apply_order).grid(row=2, column=2, padx=4, pady=4)

    def add_legend(self):
        self.ax.legend(loc=self.legend_loc_cb.get(), prop=FontProperties(family=[self.font_family_var.get(),"DejaVu Sans"], size=float(self.legend_font.get() or 10)))
        self.canvas.draw()

    def remove_legend(self):
        if self.ax.get_legend(): self.ax.get_legend().remove(); self.canvas.draw()

    def update_plot(self):
        self.ax.clear(); self._legend_handles=[]; self._legend_labels=[]
        detector = self.detector_cb.get()
        xmin = float(self.xmin.get()) if self.xmin.get() else None
        xmax = float(self.xmax.get()) if self.xmax.get() else None
        sigma = float(self.sigma_var.get())
        fontname = self.font_family_var.get()
        mpl.rcParams['font.family'] = [fontname, 'Segoe UI Symbol', 'DejaVu Sans']
        fp_main = FontProperties(family=[fontname,'Segoe UI Symbol','DejaVu Sans'], size=self.axis_fontsize_var.get())
        traces=[]; data_min=None; data_max=None
        for f in self.legend_order:
            var = self.file_vars.get(f); if not var or not var.get(): continue
            times, vals = load_and_smooth(f, detector, sigma=sigma)
            if times is None or len(times)==0: continue
            if self.normalize_var.get():
                m = np.nanmax(vals); vals = vals/m if m>0 else vals
            label = self.file_labels.get(f, f); color = self.file_colors.get(f, None)
            line, = self.ax.plot(times, vals, label=label, color=color)
            self._legend_handles.append(line); self._legend_labels.append(label)
            tmin=float(np.nanmin(times)); tmax=float(np.nanmax(times))
            data_min = tmin if data_min is None else min(data_min,tmin); data_max = tmax if data_max is None else max(data_max,tmax)
        if self.auto_legend.get(): self.add_legend()
        self.ax.set_xlabel(self.xlabel_var.get(), fontproperties=fp_main)
        self.ax.set_ylabel(self.ylabel_var.get(), fontproperties=fp_main)
        for lbl in self.ax.get_xticklabels()+self.ax.get_yticklabels(): lbl.set_fontproperties(fp_main)
        if xmin is not None or xmax is not None: self.ax.set_xlim(xmin, xmax)
        self.fig.tight_layout(); self.canvas.draw()

    def save_plot(self):
        fp = filedialog.asksaveasfilename(defaultextension='.png', filetypes=[('PNG','*.png'),('JPEG','*.jpg'),('All','*.*')])
        if not fp: return
        self.fig.tight_layout(); self.fig.savefig(fp, dpi=300)
        messagebox.showinfo("Saved", f"Saved to:\n{fp}")

# ================= Mass distribution helpers =================
TAB_SPLIT = re.compile(r"\s*\t\s*")
MWD_HEADER_RE = re.compile(r"^\s*Molar\s+mass", re.IGNORECASE)

def find_mwd_start_line(path: str):
    header_line = None; saw_mwdstart = False
    with open(path, encoding="latin1", errors="ignore") as f:
        for i, line in enumerate(f):
            s=line.strip()
            if not s: continue
            if s.startswith("MWDstart"): saw_mwdstart=True; continue
            if MWD_HEADER_RE.match(s): header_line=i; break
            if saw_mwdstart and MWD_HEADER_RE.match(s): header_line=i; break
    if header_line is not None: return header_line+1
    with open(path, encoding="latin1", errors="ignore") as f:
        in_mwd=False
        for i, line in enumerate(f):
            s=line.strip()
            if s.startswith("MWDstart"): in_mwd=True; continue
            if not in_mwd: continue
            parts=TAB_SPLIT.split(s)
            if len(parts)>=3:
                try:
                    float(parts[0]); float(parts[1]); float(parts[2]); return i
                except Exception:
                    pass
    return None

def _is_monotone_nondec(a): return np.all(np.diff(a) >= -1e-12)
def _looks_percent(a): return (np.nanmax(a) if len(a) else 0) > 1.5 and (np.nanmax(a) <= 105.0)

def load_mwd_table(path: str) -> pd.DataFrame:
    start = find_mwd_start_line(path)
    if start is None: raise ValueError("Could not find MWD table.")
    rows = []
    with open(path, encoding="latin1", errors="ignore") as f:
        for i, line in enumerate(f):
            if i < start: continue
            s=line.strip()
            if not s:
                if rows: break
                else: continue
            if ":" in s and not TAB_SPLIT.split(s)[0].replace(".", "", 1).isdigit():
                break
            parts=TAB_SPLIT.split(s)
            if len(parts) < 3: continue
            try:
                mm=float(parts[0]); c2=float(parts[1]); c3=float(parts[2])
                rows.append((mm,c2,c3))
            except Exception:
                if rows: break
                continue
    if not rows: raise ValueError("Found header but no numeric rows parsed.")
    df = pd.DataFrame(rows, columns=["MolarMass","Col2","Col3"]).dropna()
    c2, c3 = df["Col2"].to_numpy(), df["Col3"].to_numpy()
    c2f = c2/100.0 if _looks_percent(c2) else c2
    c3f = c3/100.0 if _looks_percent(c3) else c3
    c2m = _is_monotone_nondec(c2f); c3m = _is_monotone_nondec(c3f)
    if c2m and (not c3m): integral=c2f; signal=c3
    elif c3m and (not c2m): integral=c3f; signal=c2
    elif c2m and c3m:
        if (np.nanmax(c2f)-np.nanmin(c2f)) >= (np.nanmax(c3f)-np.nanmin(c3f)): integral=c2f; signal=np.full_like(c2f, np.nan)
        else: integral=c3f; signal=np.full_like(c3f, np.nan)
    else:
        integral=c3f if _is_monotone_nondec(c3f) else np.full_like(c3f, np.nan); signal=c2
    out = pd.DataFrame({"MolarMass": df["MolarMass"].to_numpy(), "Signal": signal, "Integral": integral}).dropna(subset=["MolarMass"])
    M = out["MolarMass"].to_numpy()
    use_signal = np.isfinite(out["Signal"]).all() and (np.nanstd(out["Signal"])>1e-12) and (not _is_monotone_nondec(out["Signal"].to_numpy()))
    if use_signal:
        weight = np.clip(out["Signal"].to_numpy(), 0, None); src="signal"
    else:
        I = out["Integral"].to_numpy()
        order=np.argsort(M); M_sorted=M[order]; I_sorted=I[order]; 
        if _looks_percent(I_sorted): I_sorted=I_sorted/100.0
        logM=np.log(M_sorted); dIdlogM=np.gradient(I_sorted, logM); weight_sorted=np.clip(dIdlogM, 0, None)
        weight=np.empty_like(weight_sorted); weight[order]=weight_sorted; src="integral-derivative"
    out["Weight"]=weight
    nz = out["Weight"]>0
    if nz.any():
        first=int(np.argmax(nz)); last=int(len(out)-np.argmax(nz[::-1])-1); out = out.iloc[first:last+1].reset_index(drop=True)
    out.attrs["weight_source"]=src
    return out

def compute_averages(M, w):
    M=np.asarray(M,float); w=np.clip(np.asarray(w,float),0,None)
    mask=(M>0)&np.isfinite(M)&np.isfinite(w); M=M[mask]; w=w[mask]
    if M.size==0 or w.sum()==0: return (np.nan,np.nan,np.nan)
    w=w/w.sum(); Mn=1.0/np.sum(w/M); Mw=np.sum(w*M); PDI=Mw/Mn if np.isfinite(Mn) and Mn>0 else np.nan
    return (Mn,Mw,PDI)

class MWDPanel(ttk.Frame):
    def __init__(self, master):
        super().__init__(master)
        self.root = master.winfo_toplevel()
        # mirror elugram options
        self.file_vars, self.file_labels, self.file_colors = {}, {}, {}
        self.legend_order = []
        self.auto_legend = tk.BooleanVar(value=True)
        self.inline_var = tk.BooleanVar(value=False)
        self.xlabel_var = tk.StringVar(value="Molar Mass (g/mol, log scale)")
        self.ylabel_var = tk.StringVar(value="Weight distribution w(M)")
        self.axis_fontsize_var = tk.IntVar(value=10)
        self.font_family_var = tk.StringVar(value=_default_font())
        self.show_markers = tk.BooleanVar(value=True)

        container = ttk.Frame(self); container.pack(side='top', fill='x', padx=6, pady=4)
        h_canvas = tk.Canvas(container, height=220); h_scroll = ttk.Scrollbar(container, orient='horizontal', command=h_canvas.xview)
        h_canvas.configure(xscrollcommand=h_scroll.set); h_canvas.pack(side='top', fill='x', expand=True); h_scroll.pack(side='top', fill='x')
        top_inner = ttk.Frame(h_canvas); h_canvas.create_window((0,0), window=top_inner, anchor='nw')
        top_inner.bind("<Configure>", lambda e: h_canvas.configure(scrollregion=h_canvas.bbox("all")))

        # Folder
        folder_frame = ttk.LabelFrame(top_inner, text="Folder", padding=6); folder_frame.grid(row=0, column=0, sticky='nw', padx=4, pady=2)
        self.folder_var = tk.StringVar()
        ttk.Label(folder_frame, text="Folder:").grid(row=0, column=0, sticky='w')
        ttk.Entry(folder_frame, textvariable=self.folder_var, width=30).grid(row=0, column=1, padx=4)
        ttk.Button(folder_frame, text="Browse", command=self.browse_folder).grid(row=0, column=2, padx=4)
        ttk.Button(folder_frame, text="Refresh Files", command=self.rebuild_file_list).grid(row=0, column=3, padx=4)

        # Files
        self.files_frame = ttk.LabelFrame(top_inner, text="Files", padding=6); self.files_frame.grid(row=0, column=1, sticky='nw', padx=4, pady=2)
        self.file_list_canvas = tk.Canvas(self.files_frame, width=250, height=120)
        self.file_list_scroll = ttk.Scrollbar(self.files_frame, orient='vertical', command=self.file_list_canvas.yview)
        self.file_list_inner = ttk.Frame(self.file_list_canvas)
        self.file_list_inner.bind("<Configure>", lambda e: self.file_list_canvas.configure(scrollregion=self.file_list_canvas.bbox("all")))
        self.file_list_canvas.create_window((0,0), window=self.file_list_inner, anchor='nw')
        self.file_list_canvas.configure(yscrollcommand=self.file_list_scroll.set)
        self.file_list_canvas.grid(row=0, column=0, sticky='nsew')
        self.file_list_scroll.grid(row=0, column=1, sticky='ns')

        # Legend & Colors
        legend_frame = ttk.LabelFrame(top_inner, text="Legend & Colors", padding=6); legend_frame.grid(row=0, column=2, sticky='nw', padx=4, pady=2)
        ttk.Button(legend_frame, text="Set Colors", command=self.configure_colors).grid(row=0, column=0, padx=4, pady=2)
        ttk.Button(legend_frame, text="Rename Legends", command=self.configure_labels).grid(row=0, column=1, padx=4, pady=2)
        ttk.Button(legend_frame, text="Legend Order", command=self.configure_order).grid(row=0, column=2, padx=4, pady=2)
        ttk.Checkbutton(legend_frame, text="Auto Legend", variable=self.auto_legend).grid(row=1, column=0, padx=4, pady=2, sticky='w')
        ttk.Label(legend_frame, text="Legend loc:").grid(row=2, column=0, sticky='e', padx=2)
        legend_opts = ['best','upper right','upper left','lower right','lower left','right','center left','center right','lower center','upper center','center']
        self.legend_loc_cb = ttk.Combobox(legend_frame, values=legend_opts, state='readonly', width=12); self.legend_loc_cb.current(0)
        self.legend_loc_cb.grid(row=2, column=1, padx=2, sticky='w')
        ttk.Label(legend_frame, text="Font:").grid(row=3, column=0, padx=2, sticky='e')
        self.font_cb = ttk.Combobox(legend_frame, values=_font_choices(), textvariable=self.font_family_var, state='readonly', width=18)
        self.font_cb.grid(row=3, column=1, padx=2, sticky='w')
        ttk.Label(legend_frame, text="Legend font size:").grid(row=4, column=0, padx=2, sticky='e')
        self.legend_font = ttk.Entry(legend_frame, width=4); self.legend_font.insert(0,'10'); self.legend_font.grid(row=4, column=1, padx=2)

        # Axis & options
        axis_frame = ttk.LabelFrame(top_inner, text="Axes & Markers", padding=6); axis_frame.grid(row=0, column=3, sticky='nw', padx=4, pady=2)
        ttk.Checkbutton(axis_frame, text="Inline labels", variable=self.inline_var).grid(row=0, column=0, padx=4, sticky='w')
        ttk.Checkbutton(axis_frame, text="Show Mn/Mw", variable=self.show_markers, command=self.update_plot).grid(row=0, column=1, padx=4, sticky='w')
        ttk.Label(axis_frame, text="X label:").grid(row=1, column=0, sticky='e'); ttk.Entry(axis_frame, textvariable=self.xlabel_var, width=22).grid(row=1, column=1, padx=4)
        ttk.Label(axis_frame, text="Y label:").grid(row=2, column=0, sticky='e'); ttk.Entry(axis_frame, textvariable=self.ylabel_var, width=22).grid(row=2, column=1, padx=4)
        ttk.Label(axis_frame, text="Axis Font Size:").grid(row=3, column=0, sticky='e'); ttk.Spinbox(axis_frame, from_=6, to=30, textvariable=self.axis_fontsize_var, width=5).grid(row=3, column=1, sticky='w')

        # Actions
        action_frame = ttk.LabelFrame(top_inner, text="Actions", padding=6); action_frame.grid(row=0, column=4, sticky='nw', padx=4, pady=2)
        ttk.Button(action_frame, text="Plot", command=self.update_plot).grid(row=0, column=0, padx=6, pady=4)
        ttk.Button(action_frame, text="Save Figure", command=self.save_plot).grid(row=0, column=1, padx=6, pady=4)

        # Figure
        self.fig, self.ax = plt.subplots()
        self.canvas = FigureCanvasTkAgg(self.fig, master=self); self.canvas.get_tk_widget().pack(side='top', fill='both', expand=True, padx=6, pady=4)

        self.update_plot()

    # ----- file workflow (same as elugram) -----
    def browse_folder(self):
        d = filedialog.askdirectory(title="Select folder with CSV exports")
        if d: self.folder_var.set(d); self.rebuild_file_list()

    def rebuild_file_list(self):
        folder = self.folder_var.get()
        if not folder or not os.path.isdir(folder): return
        existing = dict(self.file_vars)
        self.file_vars.clear(); self.file_labels.clear(); self.file_colors.clear()
        for w in self.file_list_inner.winfo_children(): w.destroy()
        for idx, fname in enumerate(sorted(os.listdir(folder))):
            if fname.lower().endswith(('.csv', '.txt')):
                full = os.path.join(folder, fname)
                prev = existing.get(full)
                self.file_vars[full] = tk.BooleanVar(value=prev.get() if prev else True)
                self.file_labels[full] = os.path.basename(full)
                self.file_colors[full] = None
                ttk.Checkbutton(self.file_list_inner, text=os.path.basename(full),
                                variable=self.file_vars[full], command=self.update_plot).grid(row=idx, column=0, sticky='w', padx=2, pady=1)
        if not self.legend_order:
            self.legend_order = [f for f in sorted(self.file_vars.keys()) if self.file_vars[f].get()]
        else:
            existing_order = [f for f in self.legend_order if f in self.file_vars]
            new_items = [f for f in sorted(self.file_vars.keys()) if f not in existing_order and self.file_vars[f].get()]
            self.legend_order = existing_order + new_items
        self.update_plot()

    def configure_colors(self):
        sel = [f for f, v in self.file_vars.items() if v.get()]
        if not sel: messagebox.showwarning("No Selection", "Select files first"); return
        cfg = tk.Toplevel(self.root); cfg.title("Set Colors")
        for idx, f in enumerate(sel):
            ttk.Label(cfg, text=os.path.basename(f)).grid(row=idx, column=0, padx=5, pady=2, sticky='w')
            btn = ttk.Button(cfg, text="Pick Color"); btn.grid(row=idx, column=1, padx=5)
            btn.config(command=lambda f=f: self._pick_file_color(f))
        ttk.Button(cfg, text="OK", command=lambda: (self.update_plot(), cfg.destroy())).grid(row=len(sel), column=0, columnspan=2, pady=10)

    def _pick_file_color(self, f):
        c = colorchooser.askcolor(parent=self.root)[1]
        if c: self.file_colors[f] = c; self.update_plot()

    def configure_labels(self):
        sel = [f for f, v in self.file_vars.items() if v.get()]
        if not sel: messagebox.showwarning("No Selection", "Select files first"); return
        dlg = tk.Toplevel(self.root); dlg.title("Rename Legends")
        entries = {}
        for idx, f in enumerate(sel, start=1):
            ttk.Label(dlg, text=os.path.basename(f)).grid(row=idx, column=0, padx=6, pady=3, sticky='w')
            var = tk.StringVar(value=self.file_labels.get(f, os.path.basename(f)))
            ent = ttk.Entry(dlg, textvariable=var, width=40); ent.grid(row=idx, column=1, padx=6, pady=3, sticky='w')
            entries[f] = var
        ttk.Button(dlg, text="OK", command=lambda: (self._save_labels(entries), dlg.destroy(), self.update_plot())).grid(row=len(sel)+1, column=0, columnspan=2, pady=10)

    def _save_labels(self, entries):
        for f, var in entries.items(): self.file_labels[f] = var.get()

    def configure_order(self):
        sel = [f for f, v in self.file_vars.items() if v.get()]
        if not sel: messagebox.showwarning("No Selection", "Select files first"); return
        dlg = tk.Toplevel(self.root); dlg.title("Reorder Legend / Plot Order")
        ttk.Label(dlg, text="Use Up/Down to reorder:").grid(row=0, column=0, columnspan=3, pady=4)
        lb = tk.Listbox(dlg, height=10, width=50); lb.grid(row=1, column=0, columnspan=3, padx=5)
        ordered = [f for f in self.legend_order if f in sel] or sel
        for f in ordered: lb.insert("end", self.file_labels.get(f, os.path.basename(f)))
        def move_up():
            idxs = lb.curselection()
            if not idxs or idxs[0] == 0: return
            i = idxs[0]; txt = lb.get(i); lb.delete(i); lb.insert(i-1, txt); lb.select_set(i-1)
        def move_down():
            idxs = lb.curselection()
            if not idxs or idxs[0] == lb.size() - 1: return
            i = idxs[0]; txt = lb.get(i); lb.delete(i); lb.insert(i+1, txt); lb.select_set(i+1)
        ttk.Button(dlg, text="Up", command=move_up).grid(row=2, column=0, padx=4, pady=4)
        ttk.Button(dlg, text="Down", command=move_down).grid(row=2, column=1, padx=4, pady=4)
        def apply_order():
            label_to_file = {self.file_labels.get(f, os.path.basename(f)): f for f in sel}
            new_order = []
            for i in range(lb.size()):
                label = lb.get(i); f = label_to_file.get(label)
                if f and f not in new_order: new_order.append(f)
            for f in sel:
                if f not in new_order: new_order.append(f)
            self.legend_order = new_order; dlg.destroy(); self.update_plot()
        ttk.Button(dlg, text="Apply", command=apply_order).grid(row=2, column=2, padx=4, pady=4)

    # ----- plotting -----
    def update_plot(self):
        self.ax.clear()
        fontname = self.font_family_var.get()
        mpl.rcParams['font.family'] = [fontname, 'Segoe UI Symbol', 'DejaVu Sans']
        fp_main = FontProperties(family=[fontname,'Segoe UI Symbol','DejaVu Sans'], size=self.axis_fontsize_var.get())

        any_trace=False
        for f in self.legend_order:
            var = self.file_vars.get(f)
            if not var or not var.get(): continue
            try:
                df = load_mwd_table(f)
            except Exception as e:
                continue
            x = df["MolarMass"].to_numpy(); y = df["Weight"].to_numpy()
            line, = self.ax.plot(x, y, label=self.file_labels.get(f, os.path.basename(f)), color=self.file_colors.get(f))
            self.ax.set_xscale("log")
            if self.show_markers.get():
                Mn, Mw, _ = compute_averages(x,y)
                if np.isfinite(Mn): self.ax.axvline(Mn, linestyle="--", linewidth=0.9, color=line.get_color(), alpha=0.6)
                if np.isfinite(Mw): self.ax.axvline(Mw, linestyle=":",  linewidth=0.9, color=line.get_color(), alpha=0.6)
            any_trace=True

        self.ax.set_xlabel(self.xlabel_var.get(), fontproperties=fp_main)
        self.ax.set_ylabel(self.ylabel_var.get(), fontproperties=fp_main)
        for lbl in self.ax.get_xticklabels()+self.ax.get_yticklabels(): lbl.set_fontproperties(fp_main)
        self.ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)
        if self.auto_legend.get() and any_trace:
            self.ax.legend(loc=self.legend_loc_cb.get(), prop=FontProperties(family=[fontname,'DejaVu Sans'], size=float(self.legend_font.get() or 10)))
        self.fig.tight_layout(); self.canvas.draw()

    def save_plot(self):
        fp = filedialog.asksaveasfilename(defaultextension='.png', filetypes=[('PNG','*.png'),('JPEG','*.jpg'),('All','*.*')])
        if not fp: return
        self.fig.tight_layout(); self.fig.savefig(fp, dpi=300)
        messagebox.showinfo("Saved", f"Saved to:\n{fp}")

# ================= Unified App =================
class UnifiedApp:
    def __init__(self, root):
        root.title("GPC + Mass Distribution Viewer")
        screen_w = root.winfo_screenwidth(); screen_h = root.winfo_screenheight()
        w = min(1200, int(screen_w * 0.9)); h = min(800, int(screen_h * 0.9)); x = (screen_w - w)//2; y = (screen_h - h)//2
        root.geometry(f"{w}x{h}+{x}+{y}"); root.minsize(900,600)
        nb = ttk.Notebook(root); nb.pack(fill="both", expand=True)
        elugram_tab = ElugramPanel(nb); mwd_tab = MWDPanel(nb)
        nb.add(elugram_tab, text="Elugram"); nb.add(mwd_tab, text="Mass Distribution")

if __name__ == "__main__":
    root = tk.Tk(); app = UnifiedApp(root); root.mainloop()
