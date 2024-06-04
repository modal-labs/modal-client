import subprocess

def nvidia_gpu_is_running(debug=False):
    output = subprocess.check_output(["sh", "-c", "nvidia-smi -q | grep Processes"]).decode('utf-8')
    if debug:
        print(subprocess.check_output(["nvidia-smi", "-q"]).decode('utf-8'))
    return 'None' not in output

def nvidia_cuda_pids():
    query = subprocess.check_output(["nvidia-smi", "-q"]).decode('utf-8').splitlines(0)
    procs = [int(line.split(":")[1]) for line in query if "Process ID" in line]
    return procs