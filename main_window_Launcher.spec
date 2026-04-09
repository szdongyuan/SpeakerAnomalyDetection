# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path

from PyInstaller.utils.hooks import collect_dynamic_libs


project_root = Path(SPECPATH)
onnxruntime_binaries = collect_dynamic_libs("onnxruntime")

a = Analysis(
    ["main_window_Launcher.py"],
    pathex=[str(project_root)],
    binaries=onnxruntime_binaries,
    datas=[],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "PyQt6",
        "PySide2",
        "PySide6",
        "pytest",
        "IPython",
        "jupyter_client",
        "jupyter_core",
        "jedi",
        "parso",
        "tkinter",
        "torch",
        "torchvision",
        "torchaudio",
        "functorch",
        "torchgen",
        "onnx",
        "onnxscript",
        "onnx_ir",
        "onnxruntime.transformers",
    ],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="main_window_Launcher",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    console=False,
    contents_directory=".",
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=[str(project_root / "ui" / "ui_pic" / "logo_pic" / "ting.ico")],
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    Tree(str(project_root / "ui"), prefix="ui", excludes=["__pycache__", "*.pyc", "*.pyo"]),
    Tree(str(project_root / "configs"), prefix="configs", excludes=["__pycache__", "*.pyc", "*.pyo"]),
    Tree(str(project_root / "database"), prefix="database", excludes=["__pycache__", "*.pyc", "*.pyo"]),
    Tree(str(project_root / "audio_data"), prefix="audio_data", excludes=["__pycache__", "*.pyc", "*.pyo"]),
    strip=False,
    upx=True,
    upx_exclude=[],
    name="main_window_Launcher",
)
