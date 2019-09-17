import ast
import subprocess


command = "source /Users/petermyers/Documents/pbots_calc-master/venv/bin/activate; /Users/petermyers/Documents/pbots_calc-master/python/calculator.sh KK+:QQ+ As2s3s"
process = subprocess.Popen(command,stdout=subprocess.PIPE, shell=True)
proc_stdout = ast.literal_eval(process.communicate()[0].strip().decode("utf-8"))[0][1]
# proc_stdout = process.communicate()[0].strip()
# print(type(str(proc_stdout)))
# print(str(proc_stdout))
print(proc_stdout)
print(type(proc_stdout))

# subprocess_cmd('echo a; echo b')
#
# result = ast.literal_eval(subprocess.check_output(["/Users/petermyers/Documents/pbots_calc-master/python/calculator.sh", "KK+:QQ+", "As2s3s"]))[0][1]
# print result
