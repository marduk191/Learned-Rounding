import argparse
import os
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from typing import Dict, Tuple
from tqdm import tqdm
import gc

# Written by Clybius - Modified for FP16 output

# Keys containing these strings will not be quantized if a given argument is set
AVOID_KEY_NAMES = ["norm", "bias", "embed_tokens", "shared"] #T5XXL, may need to be changed for other TEs.
T5XXL_REMOVE_KEY_NAMES = ["decoder", "lm_head"] # ComfyUI doesn't need decoder tensors or other extraneous tensors that may exist in a full T5XXL.
DISTILL_LAYER_KEYNAMES = ["distilled_guidance_layer", "final_layer", "img_in", "txt_in"]
# Target FP16 format
TARGET_FP16_DTYPE = torch.float16
# Intermediate dtype for calculations
COMPUTE_DTYPE = torch.float32
# Dtype for storing scale factors
SCALE_DTYPE = torch.float32

class LearnedRoundingConverter:
    """
    Implements adaptive rounding for converting a weight to float16.
    Inspired by AdaRound paper (https://arxiv.org/abs/2004.10568).
    "TPEC-Quant" (Top-Principal Error Correction Quantization)
    """
    def __init__(self, num_iter=256, top_k=1):
        self.num_iter = num_iter
        self.top_k = top_k
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # The maximum representable value for float16, used for scaling if needed.
        self.f16_max_val = torch.finfo(TARGET_FP16_DTYPE).max
        print(f"LearnedRoundingConverter initialized on device: {self.device}")

    def convert(self, W_orig: torch.Tensor, X_calib: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Performs the learned rounding conversion for a single weight tensor.
        For FP16, we focus on optimizing the representation within FP16's precision limits.
        """
        W_float32 = W_orig.to(self.device, dtype=COMPUTE_DTYPE)

        # Step 1: Check if scaling is needed (FP16 has much larger range than FP8)
        w_max = W_float32.abs().max()
        if w_max < 1e-12:
            print("  - Tensor is all zeros, skipping optimization.")
            scale = torch.tensor(1.0, device=self.device)
            quantized_tensor = torch.zeros_like(W_float32, dtype=TARGET_FP16_DTYPE)
            return quantized_tensor.cpu(), scale.cpu().reshape(1), torch.zeros_like(W_float32).cpu()

        # For FP16, we typically don't need aggressive scaling like FP8
        # But we can still apply a modest scale if the values are very small
        if w_max < 1e-4:
            scale = 10.0 / w_max  # Modest scaling for very small weights
        else:
            scale = torch.tensor(1.0, device=self.device)  # No scaling needed for normal range
        
        W_scaled = W_float32 * scale

        # Step 2: Initialize with naive FP16 quantization
        W_rounded = W_scaled.to(TARGET_FP16_DTYPE).to(COMPUTE_DTYPE)
        
        k = min(self.top_k, min(W_float32.shape))
        U, _, Vh = torch.pca_lowrank(W_float32, q=k, center=False, niter=16)
        Vh = Vh.T
        U_k = U[:, :k]
        Vh_k = Vh[:k, :]

        W_q_refined = W_rounded.clone()

        # Step 4: The optimization loop
        best_loss = float('inf')
        best_tensor = None
        worse_loss_counter = 0
        lr = 0.1  # Smaller learning rate for FP16 since precision is higher
        curr_lr = lr
        pbar = tqdm(range(self.num_iter), desc="    Optimizing rounding", leave=False)
        
        for i in pbar:
            current_dq = W_q_refined / scale
            error = current_dq - W_float32

            projected_error = U_k.T @ error @ Vh_k.T
            loss = torch.linalg.norm(projected_error)**2

            if loss.abs() < 1e-10:  # Tighter tolerance for FP16
                print(f"Loss {loss.item():.9f} is negligible. Stopping at iteration {i}.")
                break
            
            # Simple learning rate scheduler and early stopping
            if loss.abs() >= best_loss:
                worse_loss_counter += 1
                curr_lr = max(curr_lr / 2, 1e-10)
                if worse_loss_counter >= 40:
                    print(f"Loss ({best_loss}) has only gotten worse over {worse_loss_counter} iterations, keeping best tensor and skipping...")
                    break
            else:
                best_loss = loss.abs().item()
                best_tensor = W_q_refined.clone()
                worse_loss_counter = 0
                curr_lr = min(curr_lr * 1.1, lr)  # More conservative LR increase
            
            grad = U_k @ projected_error @ Vh_k
            W_q_refined = W_q_refined - curr_lr * grad

            pbar.set_postfix({"loss": f"{loss.item():.2e}"})

        final_tensor = best_tensor if best_tensor is not None else W_q_refined

        # Final Hard Quantization to FP16
        with torch.no_grad():
            W_f16 = final_tensor.to(TARGET_FP16_DTYPE)

        # Calculate dequantization scale (reciprocal of the quantization scale)
        dequant_scale = scale.reciprocal().reshape(1)
        
        # Clean up GPU memory
        del W_float32, W_scaled, W_rounded, W_q_refined, error, U, Vh, U_k, Vh_k
        gc.collect()
        if self.device == 'cuda':
            torch.cuda.empty_cache()

        return W_f16.cpu(), dequant_scale.cpu(), (W_f16.to(COMPUTE_DTYPE) * dequant_scale).cpu()

def get_fp16_constants(fp16_dtype: torch.dtype) -> Tuple[float, float, float]:
    """Gets the min, max, and smallest positive normal value for FP16."""
    finfo = torch.finfo(fp16_dtype)
    return float(finfo.min), float(finfo.max), float(finfo.tiny)

# Global FP16 constants
FP16_MIN, FP16_MAX, FP16_MIN_POS = get_fp16_constants(TARGET_FP16_DTYPE)

def convert_to_fp16_scaled(input_file: str, output_file: str, t5xxl: bool, keep_distillation: bool, calib_samples: int, **converter_kwargs):
    """
    Converts a safetensors file to a version with FP16 scaled weights using learned rounding (modified from AdaRound).
    """
    print(f"Processing: {input_file}")
    print(f"Output will be saved to: {output_file}")
    print(f"Using FP16 format: {TARGET_FP16_DTYPE}")
    print(f"FP16 Range: [{FP16_MIN}, {FP16_MAX}]")
    print(f"FP16 Min Precision: [{FP16_MIN_POS}]")

    tensors: Dict[str, torch.Tensor] = {}
    try:
        with safe_open(input_file, framework="pt", device="cpu") as f:
            for key in f.keys():
                tensors[key] = f.get_tensor(key).cpu()
    except Exception as e:
        print(f"Error loading '{input_file}': {e}")
        return

    # Instantiate the converter with hyperparameters from command line
    converter = LearnedRoundingConverter(**converter_kwargs)

    # Pre-generate calibration data for each unique input dimension to be more efficient
    print("\nScanning model for linear layer dimensions...")
    calibration_data_cache = {}
    for key, tensor in tensors.items():
        if key.endswith('.weight') and tensor.ndim == 2:
            in_features = tensor.shape[1]
            if in_features not in calibration_data_cache:
                print(f"  - Found new in_features dimension: {in_features}. Generating calibration data.")
                calibration_data_cache[in_features] = torch.randn(
                    calib_samples, in_features, dtype=COMPUTE_DTYPE
                )
    print("Calibration data generated.\n")

    new_tensors: Dict[str, torch.Tensor] = {}
    weight_keys = sorted([key for key in tensors.keys() if key.endswith('.weight')])
    total_weights = len(weight_keys)
    skipped_count = 0
    processed_count = 0

    print(f"Found {total_weights} weight tensors to potentially process.")

    for i, key in enumerate(weight_keys):
        process_this_key = True

        if t5xxl and any(avoid_name in key for avoid_name in T5XXL_REMOVE_KEY_NAMES):
            print(f"({i+1}/{total_weights}) Removing decoder T5XXL tensor: {key}")
            process_this_key = False
            skipped_count += 1
            continue

        if t5xxl and any(avoid_name in key for avoid_name in AVOID_KEY_NAMES):
            print(f"({i+1}/{total_weights}) Skipping excluded T5XXL tensor: {key}")
            new_tensors[key] = tensors[key]
            process_this_key = False
            skipped_count += 1

        if keep_distillation and any(avoid_name in key for avoid_name in DISTILL_LAYER_KEYNAMES):
            print(f"({i+1}/{total_weights}) Skipping excluded distillation tensor: {key}")
            new_tensors[key] = tensors[key]
            base_name = key[:-len('.weight')]
            scale_weight_key = f"{base_name}.scale_weight"
            new_tensors[scale_weight_key] = torch.tensor([1.0], dtype=SCALE_DTYPE)
            process_this_key = False
            skipped_count += 1

        if not process_this_key:
            continue

        print(f"({i+1}/{total_weights}) Processing tensor: {key}")
        processed_count += 1

        original_tensor = tensors[key]

        if original_tensor.numel() == 0 or original_tensor.ndim != 2:
            print(f"  - Skipping empty or non-2D tensor: {key}")
            new_tensors[key] = tensors[key].to(TARGET_FP16_DTYPE)
            base_name = key[:-len('.weight')]
            scale_weight_key = f"{base_name}.scale_weight"
            new_tensors[scale_weight_key] = torch.tensor([1.0], dtype=SCALE_DTYPE)
            continue

        in_features = original_tensor.shape[1]
        if in_features not in calibration_data_cache:
             print(f"  - WARNING: No calibration data found for in_features={in_features}. Skipping {key}")
             new_tensors[key] = original_tensor
             skipped_count += 1
             processed_count -= 1
             continue

        calibration_data = calibration_data_cache[in_features]

        # Use the learned rounding converter
        quantized_fp16_tensor, dequant_scale, dequantized_weight_tensor = converter.convert(original_tensor, calibration_data)

        # Store the results
        new_tensors[key] = quantized_fp16_tensor
        base_name = key[:-len('.weight')]
        bias_key = f"{base_name}.bias"
        scale_weight_key = f"{base_name}.scale_weight"
        new_tensors[scale_weight_key] = dequant_scale.to(SCALE_DTYPE)

        # --- BIAS CORRECTION ---
        if bias_key in tensors:
            print(f"  - Found and adjusting corresponding bias: {bias_key}")
            with torch.no_grad():
                original_bias = tensors[bias_key]
                
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                # Move tensors to the compute device
                W_orig_dev = original_tensor.to(device, dtype=COMPUTE_DTYPE)
                W_dequant_dev = dequantized_weight_tensor.to(device, dtype=COMPUTE_DTYPE)
                X_calib_dev = calibration_data.to(device, dtype=COMPUTE_DTYPE)
                b_orig_dev = original_bias.to(device, dtype=COMPUTE_DTYPE)

                # Calculate weight error
                weight_error = W_orig_dev - W_dequant_dev
                
                # Propagate error through the linear layer's matrix multiplication
                output_error = X_calib_dev @ weight_error.T
                
                # The bias correction is the mean of this output error across the batch dimension
                bias_correction = output_error.mean(dim=0)
                
                # Apply the correction to the original bias
                b_new = b_orig_dev - bias_correction
                
                # Store the new bias, converting back to original dtype and CPU
                new_tensors[bias_key] = b_new.cpu().to(original_bias.dtype)
                
                print(f"  - Original bias mean: {original_bias.mean().item():.6f}")
                print(f"  - New bias mean     : {new_tensors[bias_key].mean().item():.6f}")
                
                # Clean up GPU memory
                del W_orig_dev, W_dequant_dev, X_calib_dev, b_orig_dev, weight_error, output_error, bias_correction, b_new
                if device == 'cuda':
                    torch.cuda.empty_cache()

        if t5xxl:
            scale_input_key = f"{base_name}.scale_input"
            new_tensors[scale_input_key] = dequant_scale.detach().clone().to(SCALE_DTYPE)

        print(f"  - Dequant Scale  : {dequant_scale.item():.9}")
        print(f"  - Weight  : {quantized_fp16_tensor}")

    # Combine original non-weight tensors with new/modified ones
    for key, tensor in tensors.items():
        if (any(avoid_name in key for avoid_name in T5XXL_REMOVE_KEY_NAMES) and t5xxl):
            print(f"(+) Skipping decoder tensor: {key}")
            continue
        if key not in new_tensors:
            new_tensors[key] = tensor
            print(f"(+) Adding original non-quantized tensor: {key}")

    new_tensors["scaled_fp16"] = torch.empty((2), dtype=TARGET_FP16_DTYPE) if not t5xxl else torch.empty((0), dtype=TARGET_FP16_DTYPE)

    print("-" * 40)
    print(f"Saving {len(new_tensors)} tensors to {output_file}")
    try:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        save_file(new_tensors, output_file)
        print("Conversion complete!")
    except Exception as e:
        print(f"Error saving file '{output_file}': {e}")
        return

    print("-" * 40)
    print("Summary:")
    print(f"  - Original tensor count : {len(tensors)}")
    print(f"  - Weights processed     : {processed_count}")
    print(f"  - Weights skipped       : {skipped_count}")
    print(f"  - Final tensor count    : {len(new_tensors)}")
    print("-" * 40)


def main():
    parser = argparse.ArgumentParser(
        description=f"Convert safetensors weights to Scaled {TARGET_FP16_DTYPE} format using learned rounding, adapted from AdaRound.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # Original arguments
    parser.add_argument("--input", type=str, required=True, help="Input safetensors file path.")
    parser.add_argument("--output", type=str, help="Output safetensors file path. If not provided, generated based on input name.")
    parser.add_argument("--keep_distillation", action='store_true', help="Exclude distillation layers from quantization.")
    parser.add_argument("--t5xxl", action='store_true', help="Exclude certain layers for T5XXL model compatibility.")

    parser.add_argument("--calib_samples", type=int, default=3072, help="Number of random samples for calibration.")
    parser.add_argument("--num_iter", type=int, default=256, help="Number of optimization iterations per tensor.")  # Reduced default for FP16
    parser.add_argument("--top_k", type=int, default=1, help="Number of top principal components to use in optimization.")

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        return

    fp16_type_str = TARGET_FP16_DTYPE.__str__().split('.')[-1]
    distill_str = "_nodistill" if args.keep_distillation else ""
    if not args.output:
        base_name = os.path.splitext(args.input)[0]
        output_file = f"{base_name}_{fp16_type_str}_scaled_learned{distill_str}_svd.safetensors"
    else:
        output_file = args.output

    if os.path.abspath(args.input) == os.path.abspath(output_file):
        print("Error: Output file cannot be the same as the input file.")
        return

    # Pass learned rounding hyperparameters to the conversion function
    converter_kwargs = {
        'num_iter': args.num_iter,
        'top_k': args.top_k,
    }

    convert_to_fp16_scaled(
        args.input,
        output_file,
        args.t5xxl,
        args.keep_distillation,
        args.calib_samples,
        **converter_kwargs
    )

if __name__ == "__main__":
    main()