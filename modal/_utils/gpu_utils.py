import subprocess

def nvidia_gpu_is_running(debug=False):
    output = subprocess.check_output(["sh", "-c", "nvidia-smi -q | grep Processes"]).decode('utf-8')
    if debug:
        print(output)
        # print(subprocess.check_output(["nvidia-smi"]).decode('utf-8'))
    return 'None' not in output