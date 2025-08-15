# FP16 Scaled Learned Quantization with SVD

A high-quality neural network weight quantization tool that converts PyTorch model weights to FP16 format using learned rounding optimization and SVD-based error correction.

## 🚀 Features

- **Learned Rounding Optimization**: Advanced quantization using adaptive rounding inspired by AdaRound
- **SVD Error Correction**: Principal component analysis for minimizing quantization errors  
- **Adaptive Bias Correction**: Automatically adjusts biases to compensate for quantization errors
- **Smart Scaling**: Intelligent scaling strategy optimized for FP16's dynamic range
- **Model Compatibility**: Special handling for T5XXL and distillation layers
- **Memory Efficient**: Processes tensors individually to minimize GPU memory usage
- **Progress Tracking**: Detailed progress bars and logging for long conversions


## 🔧 Installation

```bash
git clone https://github.com/Clybius/Learned-Rounding.git
cd Learned-Rounding
```

## 📚 Usage

### Basic Usage

```bash
python convert_fp16_scaled_learned_svd_fast.py --input model.safetensors
```

### Advanced Usage

```bash
# T5XXL model with custom optimization parameters
python convert_fp16_scaled_learned_svd_fast.py \
    --input t5xxl_model.safetensors \
    --output t5xxl_fp16.safetensors \
    --t5xxl \
    --num_iter 512 \
    --top_k 2 \
    --calib_samples 4096
```

### Command Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--input` | str | **Required** | Input safetensors file path |
| `--output` | str | Auto-generated | Output file path |
| `--t5xxl` | flag | False | Enable T5XXL compatibility mode |
| `--keep_distillation` | flag | False | Preserve distillation layers from quantization |
| `--num_iter` | int | 256 | Optimization iterations per tensor |
| `--top_k` | int | 1 | Number of principal components for SVD |
| `--calib_samples` | int | 3072 | Calibration samples for bias correction |

## 🧠 Algorithm Overview

### 1. Learned Rounding Optimization

The core algorithm implements "TPEC-Quant" (Top-Principal Error Correction Quantization):

```python
# Simplified algorithm flow
W_scaled = W_original * scale_factor
W_quantized = optimize_rounding(W_scaled, calibration_data)
W_final = W_quantized.to(torch.float16)
```

### 2. SVD Error Correction

Uses singular value decomposition to focus optimization on the most important error directions:

```python
U, _, Vh = torch.pca_lowrank(W_original, q=top_k)
projected_error = U_k.T @ error @ Vh_k.T
gradient = U_k @ projected_error @ Vh_k
```

### 3. Adaptive Bias Correction

Automatically corrects biases to compensate for quantization errors:

```python
weight_error = W_original - W_dequantized
output_error = calibration_data @ weight_error.T
bias_correction = output_error.mean(dim=0)
new_bias = original_bias - bias_correction
```

## 📊 Performance Characteristics

| Aspect | FP16 Version |
|--------|-------------|
| **Precision** | ~3-4 decimal digits |
| **Range** | ±65,504 |
| **Memory Usage** | 2 bytes per parameter |
| **Speed** | Fast on modern GPUs |
| **Compatibility** | Universal hardware support |

## 🎯 Model Compatibility

### Supported Models
- ✅ Diffusion models (Stable Diffusion, FLUX, etc.)
- ✅ Language models (T5, BERT, etc.)
- ✅ Vision transformers
- ✅ Any PyTorch model saved as safetensors

### Special Modes
- **T5XXL Mode** (`--t5xxl`): Handles encoder-decoder architectures
- **Distillation Preservation** (`--keep_distillation`): Maintains teacher-student model compatibility

## 📁 Output Format

The converted model includes:
- **Quantized weights**: FP16 format with learned rounding
- **Scale factors**: `{layer}.scale_weight` tensors for dequantization  
- **Corrected biases**: Automatically adjusted bias terms
- **Metadata**: `scaled_fp16` marker tensor

### Loading Converted Models

```python
from safetensors import safe_open
import torch

# Load converted model
with safe_open("model_fp16_scaled.safetensors", framework="pt") as f:
    weight = f.get_tensor("layer.weight")  # FP16 quantized
    scale = f.get_tensor("layer.scale_weight")  # FP32 scale factor
    
    # Dequantize for use
    dequantized_weight = weight.to(torch.float32) * scale
```

## 🔍 Examples

### Example 1: Basic Model Conversion
```bash
python convert_fp16_scaled_learned_svd_fast.py --input stable_diffusion.safetensors
# Output: stable_diffusion_float16_scaled_learned_svd.safetensors
```

### Example 2: High-Quality T5XXL Conversion
```bash
python convert_fp16_scaled_learned_svd_fast.py \
    --input t5xxl_encoder.safetensors \
    --t5xxl \
    --num_iter 512 \
    --top_k 3 \
    --calib_samples 8192
```

### Example 3: Batch Processing Script
```bash
#!/bin/bash
for model in models/*.safetensors; do
    echo "Converting $model"
    python convert_fp16_scaled_learned_svd_fast.py --input "$model" --num_iter 128
done
```

## ⚡ Performance Tips

1. **GPU Memory**: Use `--calib_samples 1024` for very large models on limited GPU memory
2. **Speed vs Quality**: Reduce `--num_iter` to 64-128 for faster conversion with minimal quality loss  
3. **High Quality**: Increase `--top_k` to 2-3 and `--num_iter` to 512+ for maximum quality
4. **Batch Processing**: Process multiple models sequentially to avoid memory issues

## 🐛 Troubleshooting

### Common Issues

**Out of GPU Memory**
```bash
# Solution: Reduce calibration samples
python convert_fp16_scaled_learned_svd_fast.py --input model.safetensors --calib_samples 512
```

**Very Slow Conversion**  
```bash
# Solution: Reduce iterations or use CPU
python convert_fp16_scaled_learned_svd_fast.py --input model.safetensors --num_iter 64
```

**File Size Too Large**
- FP16 models are 2x smaller than FP32 but 2x larger than FP8
- Consider the original FP8 version for maximum compression

## 📈 Quality Metrics

The algorithm optimizes for minimal reconstruction error:
- **Objective**: Minimize `||W_original - W_dequantized||_F`
- **Method**: Learned rounding with principal component focus
- **Validation**: Automatic bias correction ensures output consistency

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 🙏 Acknowledgments

- **AdaRound Paper**: [Adaptive Rounding for Post-Training Quantization](https://arxiv.org/abs/2004.10568)
- **Original Author**: Clybius (FP8 implementation)
- **PyTorch Team**: For excellent quantization primitives
- **Safetensors**: For efficient model serialization

---

**⭐ If this project helped you, please give it a star!**
