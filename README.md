# Raróg
Memory allocation heuristics for tensor programs

---

## Running models using IREE

### Dependencies

- [IREE's ONNX importer](https://iree.dev/guides/deployment-configurations/cpu/#compile-a-program) (The option [onnx] might not work when running pip on ZSH)
- [IREE compiler](https://iree.dev/guides/deployment-configurations/cpu/#get-the-iree-compiler)
- [IREE runtime](https://iree.dev/guides/deployment-configurations/cpu/#get-the-iree-runtime)

### Running the models

The script [iree.sh](scripts/iree.sh) is responsible to convert a model from ONNX to torch MLIR, compile and run this model using the IREE compiler. The script assumes the IREE compiler is installed inside a Python environment located in a folder called `venv` in this repository. If this is not the case, you must:
- If IREE is installed inside a Python environment, set the variable `PYTHON_VENV_PATH`
- If IREE is built from source out of the `$PATH` variable, set the variables `IREE_IMPORT_ONNX`, `IREE_COMPILE` and `IREE_RUN_MODULE`

The script runs the first model by default, but you can change it by setting variable `MODEL_IDX` to some value between the available models before running the script.

---

## Running models using MLIR

### Dependencies

- [torch-mlir](https://github.com/llvm/torch-mlir?tab=readme-ov-file#install-torch-mlir-snapshot)
- [MLIR](https://mlir.llvm.org/getting_started/)

---

## Generating models from Nasbench

### Dependencies (Python libraries)

- tensorflow
- tqdm
- tf2onnx
- onnx

### Generating the models

The Nasbench has over 423k benchmarks but, due to space constraints, only 100 were included in this repository. The script [NASBenchConvert.py](scripts/NASBenchConvert.py) can generate the remaining models, although it will take some time to run (ONNX files are already included on the .gitignore).