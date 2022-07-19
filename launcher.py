import os
import time
import subprocess

# params
exec_file = 'misc.py'
sleep_time = 5
proc_N = 4
proc_list = list(range(0, 4))
gpu_N = 1
gpu_id = [x % gpu_N for x in proc_list]

# launch processes
for proc_idx in range(len(proc_list)):
    proc = proc_list[proc_idx]
    my_env = os.environ.copy()
    my_env["CUDA_VISIBLE_DEVICES"] = '%g' % gpu_id[proc_idx]
    subprocess.Popen(['python3', '%s' % exec_file, '--part', '%d' % proc, '--parts', '%d' % proc_N], env=my_env)
    # subprocess.Popen(['python3', '%s' % exec_file], env=my_env)
    print('%g/%g process launched' % (proc, proc_N))
    time.sleep(sleep_time)