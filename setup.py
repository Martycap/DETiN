import os
import subprocess
import venv

def setup():
    venv_dir = "venv"
    if not os.path.exists(venv_dir):
        print(f"Creating virtual environment")
        venv.create(venv_dir, with_pip=True)

    if os.name == 'nt':
        python_bin = os.path.join(venv_dir, 'Scripts', 'python.exe')
    else:
        python_bin = os.path.join(venv_dir, 'bin', 'python')

    requirements_file = 'requirements.txt'
    if os.path.exists(requirements_file):
        print(f"Installing dependecies")
        subprocess.check_call([python_bin, '-m', 'pip', 'install', '-r', requirements_file])

    subprocess.check_call([python_bin, '-m', 'pip', 'install', '-r', requirements_file])
    
if __name__ == "__main__":
    setup()
    # venv\Scripts\activate