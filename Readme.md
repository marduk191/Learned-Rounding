# Changelog

## [FP16 Version] - 2025-08-15

### Added
- FP16 output support using `torch.float16` as target dtype
- Adaptive scaling strategy optimized for FP16's larger dynamic range
- FP16-specific constants and range information display
- More conservative optimization parameters suitable for higher precision format

### Changed
- **Target Format**: Changed from `torch.float8_e4m3fn` to `torch.float16`
- **Scaling Logic**: 
  - Only applies modest scaling (10x) for very small weights (< 1e-4)
  - No scaling applied for normal weight ranges (vs aggressive scaling in FP8)
  - Leverages FP16's larger representable range (-65504 to +65504)
- **Optimization Parameters**:
  - Reduced default learning rate from `1.0` to `0.1`
  - Tightened convergence tolerance from `1e-8` to `1e-10`
  - Reduced default iterations from `500` to `256`
  - More conservative learning rate increases (`1.1x` vs `2.0x`)
- **Variable Naming**: Updated all references from FP8 to FP16
  - `TARGET_FP8_DTYPE` → `TARGET_FP16_DTYPE`
  - `f8_max_val` → `f16_max_val`
  - `FP8_MIN/MAX` → `FP16_MIN/MAX`
  - `quantized_fp8_tensor` → `quantized_fp16_tensor`
- **Function Names**: 
  - `convert_to_fp8_scaled()` → `convert_to_fp16_scaled()`
  - `get_fp8_constants()` → `get_fp16_constants()`
- **Output Files**: Default naming changed from `*_float8_e4m3fn_*` to `*_float16_*`
- **Marker Tensor**: Changed from `"scaled_fp8"` to `"scaled_fp16"`

### Technical Details
- **Precision**: FP16 provides ~3-4 decimal digits of precision vs FP8's ~2 digits
- **Range**: FP16 range is ±65504 vs FP8 e4m3fn's ±448
- **Memory**: FP16 uses 2 bytes per parameter vs FP8's 1 byte
- **Compatibility**: Better hardware support across different GPU architectures

### Removed
- FP8-specific aggressive scaling logic
- FP8 hardware compatibility checks (FP16 is universally supported)

### Performance Considerations
- Reduced default iteration count may speed up conversion
- Lower learning rates may require more iterations for convergence in some cases
- FP16 operations are generally faster than FP8 on most hardware

### Migration Notes
For users migrating from the FP8 version:
1. Output files will be ~2x larger due to FP16 vs FP8 storage
2. Models should maintain higher accuracy due to increased precision
3. All command-line arguments remain the same
4. Scale tensors and bias correction logic unchanged

### Example Usage
```bash
# Basic conversion
python convert_fp16_scaled_learned_svd_fast.py --input model.safetensors

# T5XXL with custom parameters  
python convert_fp16_scaled_learned_svd_fast.py --input model.safetensors --t5xxl --num_iter 128 --top_k 2

# With distillation layer preservation
python convert_fp16_scaled_learned_svd_fast.py --input model.safetensors --keep_distillation
```

---

## [Original FP8 Version] - Base Implementation

### Features
- FP8 E4M3FN quantization with learned rounding
- SVD-based principal component error correction
- Adaptive bias correction
- T5XXL model compatibility
- Distillation layer handling
- Calibration-based optimization
