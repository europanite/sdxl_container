import subprocess
import sys

def test_scripts_compile():
    subprocess.run([sys.executable, "-m", "compileall", "scripts"], check=True)
