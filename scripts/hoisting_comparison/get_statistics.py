import json
import os

tmp_results_path = os.environ.get('TMP_RESULTS_PATH', 'tmp')
json_results_path = os.environ.get('JSON_RESULTS_PATH', '.')
model_name = os.environ.get('MODEL_NAME', 'model_1')

methods = ['base', 'alloc', 'dealloc', 'alloc_dealloc']
dt = {}

for method in methods:
    dt[method] = {}
    model = tmp_results_path+'/'+model_name+'_'+method
    mlir_log = model+'.mlir.log'
    compile_logs = [model+'.ll.log', model+'.log']
    execution_log = model+'.exe.log'

    with open(mlir_log,'r') as f:
        a = f.readlines()
        t = float(a[0].split()[-1])
        dt[method]['mlir_lowering_time'] = t
        dt[method]['total_compilation_time'] = t

    for compile_log in compile_logs:
        with open(compile_log, 'r') as f:
            a = f.readlines()
            dt[method]['total_compilation_time'] += float(a[0].split()[-1])

    with open(execution_log, 'r') as f:
        a = f.readlines()
        dt[method]['execution_time'] = float(a[0].split()[-1])
        dt[method]['max_memory'] = round(float(a[1].split()[-1])/1024.0,2)

json.dump(dt, open(json_results_path+'/'+model_name+'.json', 'w'))