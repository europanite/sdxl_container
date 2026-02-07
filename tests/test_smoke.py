import subprocess
import sys
from pathlib import Path

def test_scripts_compile():
    target = "/scripts" if Path("/scripts").is_dir() else "scripts"
    subprocess.run([sys.executable, "-m", "compileall", target], check=True)