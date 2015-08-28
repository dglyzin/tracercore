import sys
import subprocess

argc = len(sys.argv)

SAVE_FILE = 0x01
LOAD_FILE = 0x02
TIME_EXECUTION = 0x04
STEP_EXECUTION = 0x08
STATISTICS = 0x10

thread = 1
inputFile = "empty"
outputFile = "empty"

flags = 0
step = 0
time = 0
save = "empty"
load = "empty"
stat = "empty"

fileToRun = "bin/HS"

if argc < 5 :
	print "Error! Need 4 parameters"
	print "salloc / srun"
	print "Thread count"
	print "File blocks"
	print "Result file"
	exit(0)

if sys.argv[1] != "salloc" and sys.argv[1] != "srun" :
	print "Error! Only salloc / srun"
	exit(0)
	
thread = sys.argv[2]
inputFile = sys.argv[3]
outputFile = sys.argv[4]
	
i = 5
while i < argc :
	if sys.argv[i] == "-time" :
		flags = flags | TIME_EXECUTION
		time = sys.argv[i+1]
		i = i+2
		continue
	
	if sys.argv[i] == "-step" :
		flags = flags | STEP_EXECUTION
		step = sys.argv[i+1]
		i = i+2
		continue
		
	if sys.argv[i] == "-save" :
		flags = flags | SAVE_FILE
		save = sys.argv[i+1]
		i = i+2
		continue
	
	if sys.argv[i] == "-load" :
		flags = flags | LOAD_FILE
		load = sys.argv[i+1]
		i = i+2
		continue
	
	if sys.argv[i] == "-stat" :
		flags = flags | STATISTICS
		stat = sys.argv[i+1]
		i = i+2
		continue
	
	print "Error! Need another argument"
	exit(0)

arg = inputFile + ' ' + outputFile + ' ' + str(flags) + ' ' + str(step) + ' ' + str(time) + ' ' + save + ' ' + load + ' ' + stat

command = ""

if sys.argv[1] == "salloc" :
	command = "salloc -N" + str(thread) + " -n" + str(thread) + " mpirun " + fileToRun + ' ' + arg
else :
	command = "srun -N" + str(thread) + " -n" + str(thread) + ' ' + fileToRun + ' ' + arg

print command
command ="export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:./bin &&"+ command

PIPE = subprocess.PIPE
p = subprocess.Popen(command, shell=True, stdin=PIPE, stdout=PIPE, stderr=subprocess.STDOUT, close_fds=True)
print p.stdout.read()