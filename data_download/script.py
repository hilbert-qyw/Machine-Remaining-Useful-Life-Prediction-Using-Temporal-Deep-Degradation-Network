import os
import subprocess
import sys

if len(sys.argv)<3:
    exit()


file = sys.argv[2]
dire = os.path.splitext(file)[0]
subprocess.popen('tar','-zxvf',file)
os.chdir(os.path.join(os.path.curdir,dire))
subprocess.popen('python','setup.py','install')
