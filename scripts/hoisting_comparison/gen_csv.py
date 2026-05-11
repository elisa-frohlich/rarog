import csv
import json
import os

json_results_path = os.environ.get('JSON_RESULTS_PATH', '.')
csv_result_path = os.environ.get('CSV_RESULT_PATH', '.')

# models = ['alexnet', 'googlenet', 'inception_v3', 'mnasnet1_0', 'mobilenet_v2', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'shufflenet', 'squeezenet']
models = ['alexnet', 'shufflenet', 'squeezenet']

csv_data = []

methods = ['base', 'alloc', 'dealloc', 'alloc_dealloc']
metrics = ['mlir_lowering_time', 'total_compilation_time', 'execution_time', 'max_memory']

metrics_naming = {
    'mlir_lowering_time': 'MLIR Lowering Time (s)',
    'total_compilation_time': 'Compilation Time (s)',
    'execution_time': 'Running Time (s)',
    'max_memory': 'Max Memory Usage (mb)'
}

row_idx = 0
for model_name in models:
    model_data = json.load(open(json_results_path+'/'+model_name+'.json','r'))

    csv_data.append([model_name, 'base', 'alloc', 'alloc/base (%)', 'dealloc', 'dealloc/base (%)', 'alloc+dealloc', 'a+d/base (%)'])
    for metric in metrics:
        csv_data.append([metrics_naming[metric], '', '', '', '', '', '', ''])
    
    r_idx = row_idx+1
    for metric in metrics:
        csv_data[r_idx][1] = str(model_data['base'][metric]).replace('.',',')
        c_idx = 2
        for method in methods:
            if (method == 'base'): continue
            csv_data[r_idx][c_idx] = str(model_data[method][metric]).replace('.',',')
            csv_data[r_idx][c_idx+1] = str(round(model_data[method][metric]/model_data['base'][metric],4)).replace('.',',')
            c_idx += 2
        r_idx += 1

    csv_data.append([])
    row_idx += 6

with open(csv_result_path+'/'+'mlir_bennu_data.csv', mode='w', newline='') as file:
    writer = csv.writer(file, delimiter=';', quoting=csv.QUOTE_MINIMAL)
    writer.writerows(csv_data)
