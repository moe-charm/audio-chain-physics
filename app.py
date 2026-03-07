from pathlib import Path
import runpy
import sys


ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Streamlit reruns the entry script on every widget interaction. A plain import
# would be cached after the first run, causing the UI body not to execute on
# rerun. Running the packaged module each time keeps the root entrypoint thin
# while preserving Streamlit's rerun behavior.
runpy.run_module("audiofilter.app", run_name="__main__")
