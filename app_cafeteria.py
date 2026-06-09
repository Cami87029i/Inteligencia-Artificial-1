import tkinter as tk
from tkinter import ttk, messagebox
from datetime import datetime, timedelta
import json
import pickle
import numpy as np
import torch
import torch.nn as nn
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

COLOR_BG       = "#88b8ce"
COLOR_PANEL    = "#fefaf0"
COLOR_ACENTO   = "#a13a1e"
COLOR_VERDE    = "#542916"
COLOR_AZUL     = "#45475A"
COLOR_NARANJA  = "#a13a1e"
COLOR_TEXTO    = "#a13a1e"
COLOR_SUBTEXTO = "#6C7086"
COLOR_BTN      = "#88b8ce"
COLOR_BTN_HOV  = "#45475A"

DIAS_NOMBRE = ['Lunes','Martes','Miércoles','Jueves','Viernes','Sábado','Domingo']

def cargar_artefactos():
    with open('model_config.json',        'r') as f: config   = json.load(f)
    with open('norm_params.json',         'r') as f: norm     = json.load(f)
    with open('perfiles_referencia.json', 'r') as f: perfiles = json.load(f)
    with open('val_data.json',            'r') as f: val_data = json.load(f)
    with open('scaler_kmeans.pkl',        'rb') as f: scaler_km = pickle.load(f)
    with open('kmeans_model.pkl',         'rb') as f: kmeans    = pickle.load(f)

    X_min = np.array(norm['X_min'], dtype=np.float32)
    X_max = np.array(norm['X_max'], dtype=np.float32)
    y_min = np.array(norm['y_min'], dtype=np.float32)
    y_max = np.array(norm['y_max'], dtype=np.float32)

    return config, X_min, X_max, y_min, y_max, perfiles, val_data, scaler_km, kmeans

class modelocafe(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.capa1       = nn.Linear(input_dim, 16)
        self.relu1       = nn.ReLU()
        self.dropout1    = nn.Dropout(0.2)
        self.capa2       = nn.Linear(16, 8)
        self.relu2       = nn.ReLU()
        self.capa_salida = nn.Linear(8, output_dim)

    def forward(self, x):
        x = self.dropout1(self.relu1(self.capa1(x)))
        x = self.relu2(self.capa2(x))
        return self.capa_salida(x)

def cargar_modelo(config):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = modelocafe(config['input_dim'], config['output_dim']).to(device)
    model.load_state_dict(
        torch.load('checkpoint_cafeteria.pt', map_location=device, weights_only=True)
    )
    model.eval()
    return model, device

def predecir(fecha_str, turno_str, model, device,
             X_min, X_max, y_min, y_max,
             scaler_km, kmeans, perfiles, features_kmeans):

    fecha      = datetime.strptime(fecha_str, "%Y-%m-%d")
    dia_semana = fecha.weekday()
    mes        = fecha.month
    turno      = 1 if turno_str == 'morning' else 0
    es_finde   = 1 if dia_semana >= 5 else 0

    dia_sin = np.sin(2 * np.pi * dia_semana / 7)
    dia_cos = np.cos(2 * np.pi * dia_semana / 7)
    mes_sin = np.sin(2 * np.pi * mes / 12)
    mes_cos = np.cos(2 * np.pi * mes / 12)

    clave = str((dia_semana, turno))
    if clave in perfiles:
        perfil = np.array([[perfiles[clave][f] for f in features_kmeans]],
                          dtype=np.float32)
    else:
        perfil = np.zeros((1, len(features_kmeans)), dtype=np.float32)

    # ← línea única, sin duplicado
    cluster = kmeans.predict(scaler_km.transform(perfil.astype(np.float64)))[0]

    x = np.array([[dia_sin, dia_cos, mes_sin, mes_cos,
                   turno, es_finde, float(cluster)]], dtype=np.float32)
    x_scaled = (x - X_min) / (X_max - X_min + 1e-8)

    with torch.no_grad():
        pred_s = model(
            torch.tensor(x_scaled, dtype=torch.float32).to(device)
        ).cpu().numpy()[0]

    pred_real = (pred_s * (y_max - y_min + 1e-8)) + y_min
    return [max(0, int(np.round(v))) for v in pred_real], int(cluster)

def sep(parent):
    return tk.Frame(parent, bg=COLOR_SUBTEXTO, height=1)

def btn(parent, texto, comando):
    return tk.Button(parent, text=texto, bg=COLOR_ACENTO, fg=COLOR_BG,
                     font=("Segoe UI", 10, "bold"), relief="flat",
                     cursor="hand2", command=comando, padx=10, pady=4)


def construir_tab_prediccion(tab, app):
    tab.configure(style="Dark.TFrame")

    frame_ctrl = tk.Frame(tab, bg=COLOR_PANEL, pady=12)
    frame_ctrl.pack(fill="x", padx=20)

    tk.Label(frame_ctrl, text="📅  Fecha (YYYY-MM-DD):", bg=COLOR_PANEL,
             fg=COLOR_TEXTO, font=("Segoe UI", 11)).grid(row=0, column=0,
             sticky="w", padx=(0, 8))

    app.entry_fecha = tk.Entry(frame_ctrl, width=14, bg=COLOR_BTN,
                               fg=COLOR_TEXTO, insertbackground=COLOR_TEXTO,
                               font=("Segoe UI", 11), relief="flat")
    app.entry_fecha.insert(0, datetime.today().strftime("%Y-%m-%d"))
    app.entry_fecha.grid(row=0, column=1, padx=(0, 20))

    tk.Label(frame_ctrl, text="⏰  Turno:", bg=COLOR_PANEL,
             fg=COLOR_TEXTO, font=("Segoe UI", 11)).grid(row=0, column=2,
             sticky="w", padx=(0, 8))

    app.combo_turno = ttk.Combobox(frame_ctrl, values=["morning", "afternoon"],
                                   width=10, state="readonly",
                                   font=("Segoe UI", 11))
    app.combo_turno.current(0)
    app.combo_turno.grid(row=0, column=3, padx=(0, 20))

    btn(frame_ctrl, "  Predecir  ",
        lambda: _actualizar_prediccion(app)).grid(row=0, column=4)

    sep(tab).pack(fill="x", padx=20, pady=4)

    app.lbl_info = tk.Label(tab, text="", bg=COLOR_PANEL,
                            fg=COLOR_SUBTEXTO, font=("Segoe UI", 10))
    app.lbl_info.pack(padx=20, anchor="w", pady=(4, 0))

    frame_tabla = tk.Frame(tab, bg=COLOR_PANEL)
    frame_tabla.pack(fill="both", expand=True, padx=20, pady=8)

    cols = ("Producto", "Unidades a Producir")
    app.tree_pred = ttk.Treeview(frame_tabla, columns=cols,
                                  show="headings", height=8)
    app.tree_pred.heading("Producto",            text="Producto")
    app.tree_pred.heading("Unidades a Producir", text="Unidades a Producir")
    app.tree_pred.column("Producto",             width=280, anchor="w")
    app.tree_pred.column("Unidades a Producir",  width=180, anchor="center")
    app.tree_pred.pack(fill="both", expand=True)

    app.tree_pred.tag_configure("par",   background=COLOR_PANEL, foreground=COLOR_TEXTO)
    app.tree_pred.tag_configure("impar", background=COLOR_BTN,   foreground=COLOR_VERDE)

def _actualizar_prediccion(app):
    try:
        fecha = app.entry_fecha.get().strip()
        turno = app.combo_turno.get()
        datetime.strptime(fecha, "%Y-%m-%d")
    except ValueError:
        messagebox.showerror("Error", "Fecha inválida. Usa el formato YYYY-MM-DD")
        return

    try:
        preds, cluster = predecir(
            fecha, turno, app.model, app.device,
            app.X_min, app.X_max, app.y_min, app.y_max,
            app.scaler_km, app.kmeans, app.perfiles,
            app.config['features_kmeans']
        )
    except Exception as e:
        messagebox.showerror("Error en predicción", str(e))
        return

    dia_nombre = DIAS_NOMBRE[datetime.strptime(fecha, "%Y-%m-%d").weekday()]
    app.lbl_info.config(
        text=f"  {fecha}  —  {dia_nombre}  —  {turno.upper()}  "
             f"[Cluster K-Means = {cluster}]"
    )

    app.tree_pred.delete(*app.tree_pred.get_children())
    for i, (prod, uds) in enumerate(zip(app.config['nombres_productos'], preds)):
        tag = "impar" if i % 2 else "par"
        app.tree_pred.insert("", "end", values=(prod, f"{uds} uds"), tags=(tag,))

    dia_nombre = DIAS_NOMBRE[datetime.strptime(fecha, "%Y-%m-%d").weekday()]
    app.lbl_info.config(
        text=f"  {fecha}  —  {dia_nombre}  —  {turno.upper()}  "
             f"[Cluster K-Means = {cluster}]"
    )

    app.tree_pred.delete(*app.tree_pred.get_children())
    for i, (prod, uds) in enumerate(zip(app.config['nombres_productos'], preds)):
        tag = "impar" if i % 2 else "par"
        app.tree_pred.insert("", "end", values=(prod, f"{uds} uds"), tags=(tag,))

def construir_tab_grafica(tab, app):
    tab.configure(style="Dark.TFrame")

    frame_ctrl = tk.Frame(tab, bg=COLOR_PANEL, pady=10)
    frame_ctrl.pack(fill="x", padx=20)

    tk.Label(frame_ctrl, text="Producto:", bg=COLOR_PANEL,
             fg=COLOR_TEXTO, font=("Segoe UI", 11)).grid(row=0, column=0,
             padx=(0, 8), sticky="w")

    app.combo_prod = ttk.Combobox(frame_ctrl,
                                   values=app.config['nombres_productos'],
                                   width=28, state="readonly",
                                   font=("Segoe UI", 10))
    app.combo_prod.current(0)
    app.combo_prod.grid(row=0, column=1, padx=(0, 16))

    btn(frame_ctrl, "  Ver gráfica  ",
        lambda: _actualizar_grafica(app)).grid(row=0, column=2)

    app.frame_fig = tk.Frame(tab, bg=COLOR_PANEL)
    app.frame_fig.pack(fill="both", expand=True, padx=10, pady=6)

def _actualizar_grafica(app):
    idx  = app.combo_prod.current()
    prod = app.config['nombres_productos'][idx]
    real = np.array(app.val_data['real'])[:, idx]
    pred = np.array(app.val_data['predicho'])[:, idx]
    n    = min(50, len(real))

    for w in app.frame_fig.winfo_children():
        w.destroy()

    fig = Figure(figsize=(9, 3.8), facecolor=COLOR_BG)
    ax  = fig.add_subplot(111, facecolor=COLOR_PANEL)

    ax.plot(real[:n], color=COLOR_VERDE,   linewidth=1.8,
            marker='o', markersize=3, label='Demanda Real', alpha=0.85)
    ax.plot(pred[:n], color=COLOR_NARANJA, linewidth=1.8,
            marker='s', markersize=3, linestyle='--',
            label='Predicción IA', alpha=0.9)

    mae = np.mean(np.abs(real - pred))
    ax.set_title(f"{prod}  —  MAE = {mae:.1f} uds/turno",
                 color=COLOR_ACENTO, fontsize=11, fontweight='bold')
    ax.set_xlabel("Turno (set de validación)", color=COLOR_SUBTEXTO)
    ax.set_ylabel("Unidades",                  color=COLOR_SUBTEXTO)
    ax.tick_params(colors=COLOR_SUBTEXTO)
    ax.legend(facecolor=COLOR_BTN, labelcolor=COLOR_TEXTO, fontsize=9)
    ax.grid(True, linestyle=':', alpha=0.3, color=COLOR_SUBTEXTO)
    for spine in ax.spines.values():
        spine.set_edgecolor(COLOR_BTN_HOV)

    fig.tight_layout()
    canvas = FigureCanvasTkAgg(fig, master=app.frame_fig)
    canvas.draw()
    canvas.get_tk_widget().pack(fill="both", expand=True)

def construir_tab_metricas(tab, app):
    tab.configure(style="Dark.TFrame")

    real = np.array(app.val_data['real'])
    pred = np.array(app.val_data['predicho'])
    maes = [np.mean(np.abs(real[:, i] - pred[:, i]))
            for i in range(real.shape[1])]
    mae_max = max(maes)

    tk.Label(tab, text="  MAE por Producto — Set de Validación",
             bg=COLOR_PANEL, fg=COLOR_ACENTO,
             font=("Segoe UI", 12, "bold")).pack(anchor="w", padx=20, pady=(14, 4))

    sep(tab).pack(fill="x", padx=20, pady=4)

    frame = tk.Frame(tab, bg=COLOR_PANEL)
    frame.pack(fill="both", expand=True, padx=24, pady=8)

    for i, (prod, mae) in enumerate(zip(app.config['nombres_productos'], maes)):
        fila = tk.Frame(frame, bg=COLOR_PANEL)
        fila.pack(fill="x", pady=5)

        tk.Label(fila, text=prod, bg=COLOR_PANEL, fg=COLOR_TEXTO,
                 font=("Segoe UI", 10), width=26, anchor="w").pack(side="left")

        ancho_barra = int((mae / mae_max) * 280)
        barra_frame = tk.Frame(fila, bg=COLOR_BTN, width=300, height=18)
        barra_frame.pack(side="left", padx=8)
        barra_frame.pack_propagate(False)
        tk.Frame(barra_frame, bg=COLOR_ACENTO,
                 width=max(4, ancho_barra), height=18).place(x=0, y=0)

        tk.Label(fila, text=f"{mae:.1f} uds", bg=COLOR_PANEL,
                 fg=COLOR_VERDE, font=("Segoe UI", 10, "bold")).pack(side="left")

    sep(tab).pack(fill="x", padx=20, pady=8)
    mae_global = np.mean(maes)
    tk.Label(tab, text=f"  MAE promedio global:  {mae_global:.2f} unidades / turno",
             bg=COLOR_PANEL, fg=COLOR_VERDE,
             font=("Segoe UI", 11, "bold")).pack(anchor="w", padx=20)

def construir_tab_semanal(tab, app):
    tab.configure(style="Dark.TFrame")

    frame_ctrl = tk.Frame(tab, bg=COLOR_PANEL, pady=10)
    frame_ctrl.pack(fill="x", padx=20)

    tk.Label(frame_ctrl, text="📅  Semana desde (YYYY-MM-DD):",
             bg=COLOR_PANEL, fg=COLOR_TEXTO,
             font=("Segoe UI", 11)).grid(row=0, column=0, sticky="w", padx=(0, 8))

    hoy   = datetime.today()
    lunes = hoy - timedelta(days=hoy.weekday())
    app.entry_semana = tk.Entry(frame_ctrl, width=14, bg=COLOR_BTN,
                                fg=COLOR_TEXTO, insertbackground=COLOR_TEXTO,
                                font=("Segoe UI", 11), relief="flat")
    app.entry_semana.insert(0, lunes.strftime("%Y-%m-%d"))
    app.entry_semana.grid(row=0, column=1, padx=(0, 20))

    btn(frame_ctrl, "  Generar Plan  ",
        lambda: _actualizar_semanal(app)).grid(row=0, column=2)

    sep(tab).pack(fill="x", padx=20, pady=4)

    frame_tabla = tk.Frame(tab, bg=COLOR_PANEL)
    frame_tabla.pack(fill="both", expand=True, padx=20, pady=8)

    cols = ["Día", "Turno"] + [p[:12] for p in app.config['nombres_productos']]
    app.tree_sem = ttk.Treeview(frame_tabla, columns=cols,
                                 show="headings", height=14)
    for c in cols:
        app.tree_sem.heading(c, text=c)
        w = 95 if c not in ("Día", "Turno") else 105
        app.tree_sem.column(c, width=w, anchor="center")

    scroll_x = ttk.Scrollbar(frame_tabla, orient="horizontal",
                              command=app.tree_sem.xview)
    app.tree_sem.configure(xscrollcommand=scroll_x.set)
    scroll_x.pack(side="bottom", fill="x")
    app.tree_sem.pack(fill="both", expand=True)

    app.tree_sem.tag_configure("morning",   background=COLOR_BTN,   foreground=COLOR_AZUL)
    app.tree_sem.tag_configure("afternoon", background=COLOR_PANEL, foreground=COLOR_NARANJA)

def _actualizar_semanal(app):
    try:
        inicio = datetime.strptime(app.entry_semana.get().strip(), "%Y-%m-%d")
    except ValueError:
        messagebox.showerror("Error", "Fecha inválida. Usa el formato YYYY-MM-DD")
        return

    app.tree_sem.delete(*app.tree_sem.get_children())

    for d in range(7):
        fecha      = inicio + timedelta(days=d)
        fecha_str  = fecha.strftime("%Y-%m-%d")
        dia_nombre = DIAS_NOMBRE[fecha.weekday()]

        for turno in ["morning", "afternoon"]:
            preds, _ = predecir(
                fecha_str, turno, app.model, app.device,
                app.X_min, app.X_max, app.y_min, app.y_max,
                app.scaler_km, app.kmeans, app.perfiles,
                app.config['features_kmeans']
            )
            fila = [dia_nombre, turno] + preds
            app.tree_sem.insert("", "end", values=fila, tags=(turno,))


class App:
    def __init__(self, root):
        root.title("Sistema de Predicción — Cafetería")
        root.configure(bg=COLOR_BG)
        root.geometry("880x560")
        root.resizable(True, True)

        try:
            (self.config, self.X_min, self.X_max,
             self.y_min, self.y_max, self.perfiles,
             self.val_data, self.scaler_km, self.kmeans) = cargar_artefactos()
            self.model, self.device = cargar_modelo(self.config)
        except FileNotFoundError as e:
            messagebox.showerror("Error al cargar",
                f"No se encontró un archivo necesario:\n{e}\n\n"
                "Ejecuta primero la celda de guardado en el notebook.")
            root.destroy()
            return

        estilo = ttk.Style()
        estilo.theme_use("clam")
        estilo.configure("Dark.TFrame",      background=COLOR_PANEL)
        estilo.configure("TNotebook",        background=COLOR_BG,    borderwidth=0)
        estilo.configure("TNotebook.Tab",    background=COLOR_BTN,   foreground=COLOR_TEXTO,
                         padding=[14, 6],    font=("Segoe UI", 10))
        estilo.map("TNotebook.Tab",
                   background=[("selected", COLOR_PANEL)],
                   foreground=[("selected", COLOR_ACENTO)])
        estilo.configure("Treeview",         background=COLOR_PANEL, foreground=COLOR_TEXTO,
                         fieldbackground=COLOR_PANEL, rowheight=26,
                         font=("Segoe UI", 10))
        estilo.configure("Treeview.Heading", background=COLOR_BTN,   foreground=COLOR_ACENTO,
                         font=("Segoe UI", 10, "bold"))
        estilo.configure("TCombobox",        fieldbackground=COLOR_BTN, foreground=COLOR_TEXTO)
        estilo.configure("TScrollbar",       background=COLOR_BTN)

        # ── Header ──
        header = tk.Frame(root, bg=COLOR_BG, pady=10)
        header.pack(fill="x", padx=20)
        tk.Label(header, text="☕  Sistema de Predicción de Producción",
                 bg=COLOR_BG, fg=COLOR_ACENTO,
                 font=("Segoe UI", 14, "bold")).pack(side="left")
        tk.Label(header,
                 text=f"K-Means (k={self.config['mejor_k']}) + MLP  |  "
                      f"MAE global: {self.val_data['mae_global']:.2f} uds/turno",
                 bg=COLOR_BG, fg=COLOR_SUBTEXTO,
                 font=("Segoe UI", 9)).pack(side="right")

        nb = ttk.Notebook(root)
        nb.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        tabs = {
            "🔮  Predicción":   construir_tab_prediccion,
            "📈  Comparativa":  construir_tab_grafica,
            "📊  Métricas MAE": construir_tab_metricas,
            "📅  Plan Semanal": construir_tab_semanal,
        }
        for nombre, fn in tabs.items():
            frame = ttk.Frame(nb, style="Dark.TFrame")
            nb.add(frame, text=nombre)
            fn(frame, self)

        nb.bind("<<NotebookTabChanged>>",
                lambda e: _on_tab_change(e, nb, self))

def _on_tab_change(event, nb, app):
    idx = nb.index(nb.select())
    if idx == 1:
        _actualizar_grafica(app)
    elif idx == 3:
        _actualizar_semanal(app)

if __name__ == "__main__":
    root = tk.Tk()
    App(root)
    root.mainloop()