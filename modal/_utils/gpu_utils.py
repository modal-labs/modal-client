import subprocess

def nvidia_gpu_is_running():
    output = subprocess.check_output(["sh", "-c", "nvidia-smi -q | grep Processes"]).decode('utf-8')
    return 'None' not in output