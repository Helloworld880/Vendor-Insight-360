from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType


def _load_root_frontend() -> ModuleType:
    root_app_path = Path(__file__).resolve().parents[1] / "app.py"
    spec = importlib.util.spec_from_file_location("vendor_insight_streamlit_app", root_app_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load Streamlit entrypoint from {root_app_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_root_frontend = _load_root_frontend()
main = _root_frontend.main


if __name__ == "__main__":
    main()
