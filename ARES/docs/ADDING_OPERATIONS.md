# Adding Operations

This is the “end-to-end” checklist for adding a new operation so it is supported in:
PyTorch extraction → Python INT8 reference → codegen planning → GAP9 runtime execution → tests.

Before adding a new op, check `docs/SUPPORTED_OPERATIONS.md` to confirm whether a compatible path already exists.

## Workflow

### 1) Python reference implementation

- Create `atomic_ops/<new_op>.py`
- Implement INT8 behavior (use INT32 accumulation for MACs)
- Add a unit test function so you can run `python atomic_ops/<new_op>.py`
- Export the op from `atomic_ops/__init__.py`

### 2) PyTorch/Brevitas extraction support

- Update `tools/pytorch_extractor.py`:
  - Detect the Brevitas/custom module in `BrevitasExtractor._extract_layers()`
  - Extract parameters (weights/biases/shapes/etc.)
  - Extract runtime scales via a forward pass (if required)
  - Insert the layer into `layer_order`

### 3) INT8 inference engine support (golden generation)

- Update `tools/int8_inference.py`:
  - Add a handler in `forward()`
  - Call your atomic op
  - Propagate scales correctly
  - Add fusion detection (if applicable)

### 4) GAP9 C kernel

- Update `codegen/runtime/src/network_kernels.c`:
  - Implement `network_<new_op>_int8(...)`
  - Match Python behavior exactly (bit-exact)
  - Add prototype in `codegen/runtime/inc/network_kernels.h`

### 5) Layer descriptors (data-driven executor)

- Update `codegen/runtime/inc/layer_descriptors.h`:
  - Add a new `LayerType` enum value
  - Extend `LayerSpec` with any required parameters

### 6) Code generation integration

- Update `codegen/generate_c_code.py`:
  - Extend the execution-plan build step (layer specs) for the new layer type
  - Compute shapes, buffers, and memory residency decisions
  - Emit the right `LayerSpec` fields for the runtime executor
- Update tiling, if needed:
  - Add tiling logic to `codegen/gap9_model.py`
  - Return `None` when tiling is not applicable (forces L2 fallback)

### 7) Runtime executor dispatch

- Update `codegen/templates/network.c.mako`:
  - Add a `case LAYER_<NEW_OP>:` in the main execution loop
  - Choose path (L1 tiled vs L2 fallback vs L3 staged/streamed)
  - Wire buffers via arena offsets (avoid per-tensor malloc)

### 8) Test network validation

- Add a new test network in `tests/test_networks/test_N_<new_op>.py`
- Register it in `tests/generate_all_tests.py` (`TestGenerator.NETWORKS`)
- Run end-to-end:
  - `python tests/generate_all_tests.py --test test_N_<new_op> --skip-gvsoc`
  - `python tests/run_gap9_projects.py --tests test_N_<new_op>` (requires GAP SDK)

## Concrete example: adding `sigmoid`

Minimal “shape” of the work:

1. `atomic_ops/sigmoid.py`: implement `sigmoid_int8()` + `test_sigmoid_int8()`
2. `tools/pytorch_extractor.py`: detect `nn.Sigmoid` (or Brevitas equivalent) and add layer type
3. `tools/int8_inference.py`: add `elif layer_type == 'Sigmoid': ...`
4. `codegen/runtime/src/network_kernels.c`: add `network_sigmoid_int8()` (likely LUT-based)
5. `codegen/runtime/inc/layer_descriptors.h`: add `LAYER_SIGMOID`
6. `codegen/generate_c_code.py` + `codegen/templates/network.c.mako`: emit + execute the new layer spec
7. `tests/test_networks/test_N_sigmoid_basic.py`: add small net that exercises sigmoid in-context

## Debugging checklist (before calling it done)

- [ ] `python atomic_ops/<new_op>.py` passes
- [ ] Golden outputs generate without errors
- [ ] C code compiles cleanly
- [ ] GVSOC run completes
- [ ] Layer-by-layer golden comparison passes
