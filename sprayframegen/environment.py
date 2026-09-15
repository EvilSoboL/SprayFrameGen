"""Фактическое окружение и идентификатор содержимого сборки."""

import hashlib
import platform
from pathlib import Path
import re
import sys
import tkinter
import zlib

import numpy
import PIL
from PIL import features

from .configuration import Environment, Runtime


def build_id() -> str:
    root = Path(__file__).parent
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes().replace(b"\r\n", b"\n"))
        digest.update(b"\0")
    return "sha256:" + digest.hexdigest()


def _tk_version(name: str, abi: float) -> str:
    # Read CPython's bundled Windows script version without opening a GUI window.
    filename = "init.tcl" if name == "Tcl" else "tk.tcl"
    script = Path(sys.base_prefix) / "tcl" / f"{name.lower()}{abi}" / filename
    if script.is_file():
        match = re.search(r"package require -exact " + name + r"\s+([\d.]+)",
                          script.read_text(encoding="utf-8"))
        if match:
            return match.group(1)
    return f"{abi} (patch unknown)"


def current_environment() -> Environment:
    return Environment(
        os=platform.platform(),
        architecture=platform.machine() + "/" + platform.architecture()[0],
        processor=platform.processor() or platform.machine() or "unknown",
        runtime=Runtime(platform.python_implementation(), platform.python_version()),
        dependencies={
            "numpy": numpy.__version__, "Pillow": PIL.__version__,
            "zlib": zlib.ZLIB_RUNTIME_VERSION,
            "pillow-zlib": features.version_codec("zlib") or "unavailable",
            "libtiff": features.version_codec("libtiff") or "unavailable",
            "Tcl": _tk_version("Tcl", tkinter.TclVersion),
            "Tk": _tk_version("Tk", tkinter.TkVersion),
        },
        build_id=build_id(), numerical_profile="standard-v1",
    )
