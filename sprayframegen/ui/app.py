"""Минимальный рабочий редактор конфигурации, без имитации генерации."""

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText

from .. import __version__
from . import InterfaceError
from ..configuration import ConfigurationError, dumps, load, loads, new_configuration, save


def create_window() -> tk.Tk:
    root = tk.Tk()
    root.title(f"SprayFrameGen {__version__} — Настройки")
    root.geometry("900x760")
    root.minsize(640, 480)
    toolbar = ttk.Frame(root, padding=8)
    toolbar.pack(fill="x")
    ttk.Label(root, text="Настройки камеры, капель и оптических эффектов (JSON)",
              padding=(10, 4)).pack(anchor="w")
    editor = ScrolledText(root, wrap="none", font=("Consolas", 11), undo=True)
    editor.pack(fill="both", expand=True, padx=10, pady=6)
    status = tk.StringVar(value="Каркас приложения. Генерация кадров появится в следующих задачах.")
    ttk.Label(root, textvariable=status, padding=10, wraplength=850).pack(fill="x")

    def show(config):
        editor.delete("1.0", "end")
        editor.insert("1.0", dumps(config))

    def guarded(action):
        try:
            action()
        except (ConfigurationError, OSError, UnicodeError) as exc:
            messagebox.showerror("Ошибка настроек", str(exc), parent=root)

    def report(result):
        status.set("Настройки проверены. " + " ".join(result.warnings))
        if result.warnings:
            messagebox.showwarning("Предупреждения", "\n".join(result.warnings), parent=root)

    def open_file():
        path = filedialog.askopenfilename(parent=root, filetypes=[("Настройки JSON", "*.json")])
        if path:
            result = load(path)
            report(result)
            show(result.configuration)

    def save_file():
        result = loads(editor.get("1.0", "end-1c"))
        report(result)
        path = filedialog.asksaveasfilename(parent=root, defaultextension=".json",
                                          filetypes=[("Настройки JSON", "*.json")])
        if path:
            save(result.configuration, path)
            status.set(f"Настройки сохранены: {path}")

    def reset(mode):
        show(new_configuration(render_mode=mode))
        status.set("Новая конфигурация. Seed создан автоматически.")

    for label, action in [
        ("Новая: реалистичный", lambda: reset("realistic")),
        ("Новая: идеальный", lambda: reset("ideal")),
        ("Открыть…", open_file), ("Сохранить…", save_file),
        ("Проверить", lambda: report(loads(editor.get("1.0", "end-1c"))))]:
        ttk.Button(toolbar, text=label, command=lambda fn=action: guarded(fn)).pack(side="left", padx=3)
    show(new_configuration())
    return root


def run() -> None:
    try:
        create_window().mainloop()
    except tk.TclError as exc:
        raise InterfaceError(
            "Не удалось запустить окно настроек. Проверьте установку Python с Tcl/Tk "
            "и доступ к интерактивному рабочему столу. "
            "Проверить JSON без окна можно командой run.bat --check <файл>."
        ) from exc
