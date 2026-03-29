import subprocess
import sys

def run():
    # Write a wrapper script to find torch
    wrapper = """
import sys
import torch
print(torch.__version__)
"""
    with open('test_torch.py', 'w') as f:
        f.write(wrapper)
        
    try:
        # try the existing conda environment from earlier conversations
        # The user has python3 in their path, wait, let's just see if conda exists
        subprocess.run("source ~/miniconda3/bin/activate && python3 test_apkvc.py", shell=True, executable='/bin/bash')
    except Exception as e:
        print(e)
run()
