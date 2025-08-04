# Learned-Rounding
A repository of Python &amp; PyTorch scripts which (currently) converts .safetensors models into scaled FP8 variants, utilizing gradient descent for optimal rounding.

# TPEC-Quant (Top-Principal Error Correction Quantization) 
- A novel method (Designed by Clybius) which utilizes SVD to calculate error on the top principal component and descends upon more accurate representations, leading to far better results in FP8 precision. Results are very akin to FP16/BF16 precision when scaled with a single scalar value.
- Obtainable via `convert_fp8_scaled_learned_svd.py` in this repository.
- Natively supported in ComfyUI and other UI utilities based off of ComfyUI, thanks to their scaled FP8 implementation!
- Takes ~10 minutes to quantize a large AI image diffusion model (Chroma). Performance may largely vary depending on hardware, and can likely be improved upon with multiprocessing (not yet complete), a faster drive for reading/writing (partially broke), and a faster GPU/processing unit.
- Supports Chroma, FLUX, T5XXL (includes removal of decoder and extra tensors from a full model), and maybe more!

Usage: `python convert_fp8_scaled_learned_svd.py --input /path/to/model.safetensors`

Arguments:
- `--input "/path/to/model.safetensors"`: Input safetensors file path.
- `--output "/path/to/output_model.safetensors"`: Output safetensors file path. If not provided, it will be created based on the input location and name.
- `--keep_distillation`: Exclude distillation layers from quantization. (Likely not helpful because ComfyUI may use Round-to-Nearest in place of this without further modification.) (Default False)
- `--t5xxl`: Exclude certain layers for T5XXL model compatibility. (Default False)
- `--calib_samples INT`: Number of random samples for calibration. Currently only used for bias correction. (Default 1024)
- `--num_iter INT`: Number of optimization iterations per tensor. Increasing will result in higher accuracy, but longer offline quantization times. (Default 250)
