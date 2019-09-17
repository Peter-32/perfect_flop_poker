import ast
import subprocess
result = ast.literal_eval(subprocess.check_output(["/Users/petermyers/Documents/pbots_calc-master/python/calculator.sh", "KK+:QQ+", "As2s3s"]))[0][1]
print result
