import argparse
import os
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from typing import Dict, Tuple
from tqdm import tqdm
import gc

# Written by Clybius

# Keys containing these strings will not be quantized if a given argument is set
AVOID_KEY_NAMES = ["norm", "bias", "embed_tokens", "shared"] #T5XXL, may need to be changed for other TEs.
T5XXL_REMOVE_KEY_NAMES = ["decoder", "lm_head"] # ComfyUI doesn't need decoder tensors or other extraneous tensors that may exist in a full T5XXL.
DISTILL_LAYER_KEYNAMES = ["distilled_guidance_layer", "final_layer", "img_in", "txt_in"]
# Target FP8 format
TARGET_FP8_DTYPE = torch.float8_e4m3fn
# Intermediate dtype for calculations
COMPUTE_DTYPE = torch.float32 # Don't think more hurts here since we're working tensor by tensor.
# Dtype for storing scale factors
SCALE_DTYPE = torch.float32

class LearnedRoundingConverter:
    """
    Implements adaptive rounding for converting a weight to float8.
    Inspired by AdaRound paper (https://arxiv.org/abs/2004.10568).
    "TPEC-Quant" (Top-Principal Error Correction Quantization)
    """
    def __init__(self, num_iter=256):
        self.num_iter = num_iter
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # The maximum representable value for e4m3fn, used for scaling.
        self.f8_max_val = torch.finfo(TARGET_FP8_DTYPE).max
        print(f"LearnedRoundingConverter initialized on device: {self.device}")

    def convert(self, W_orig: torch.Tensor, X_calib: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Performs the learned rounding conversion for a single weight tensor.
        """
        W_float32 = W_orig.to(self.device, dtype=COMPUTE_DTYPE)

        # Step 1: Calculate the quantization scale (per-tensor asymmetric)
        w_max = W_float32.abs().max()
        if w_max < 1e-12:
            print("  - Tensor is all zeros, skipping optimization.")
            scale = torch.tensor(1.0, device=self.device)
            quantized_tensor = torch.zeros_like(W_float32, dtype=TARGET_FP8_DTYPE)
            return quantized_tensor.cpu(), scale.reciprocal().cpu().reshape(1), torch.zeros_like(W_float32).cpu()

        scale = self.f8_max_val / w_max # Example: (absmax = 1, fp8 max = +-448 for dtype e4m3_fn)
        W_scaled = W_float32 * scale # absmax now +-448

        # Step 2: Initialize the rounding mask 'h'
        W_rounded = W_scaled.to(TARGET_FP8_DTYPE).to(COMPUTE_DTYPE) # Naive RtN quantization on scaled model
        W_dq_rounded = W_rounded / scale # Scale back down with scalar

        try: # Try PCA for far faster estimation of U and Vh
            U, _, Vh = torch.pca_lowrank(W_float32, q=1, center=False, niter=500) # To my knowledge, LAPACK (or magma or w/e) uses 1k iters by default. Unsure if the default of 2 is good so set it to 1k here.
            Vh = Vh.T
        except: # Fallback to SVD just in case
            U, _, Vh = torch.linalg.svd(W_float32, full_matrices=False)
        U_k = U[:, :1] # Obtain most important low-rank matrices
        Vh_k = Vh[:1, :]

        W_q_refined = W_rounded.clone() # Clone, as this tensor will be the one thats iteratively refined

        # Step 4: The optimization loop
        best_loss = float('inf')
        best_tensor = None
        worse_loss_counter = 0
        lr = 4.0
        curr_lr = lr
        pbar = tqdm(range(self.num_iter), desc="    Optimizing rounding", leave=False)
        for i in pbar:

            current_dq = W_q_refined / scale
            error = current_dq - W_float32

            projected_error = U_k.T @ error @ Vh_k.T

            loss = torch.linalg.norm(projected_error)**2

            if loss.abs() < 1e-8:
                print(f"Loss {loss.item():.9f} is negligible. Stopping at iteration {i}.")
                break
            
            # Simple learning rate scheduler and early stopping
            if loss.abs() >= best_loss:
                worse_loss_counter += 1
                curr_lr = max(curr_lr / 2, 1e-8)
                if worse_loss_counter >= 40: # Reduce LR after 20 worse iterations
                    print(f"Loss ({best_loss}) has only gotten worse over {worse_loss_counter} iterations, keeping best tensor and skipping...")
                    break
            else:
                best_loss = loss.abs().item()
                best_tensor = W_q_refined.clone()
                worse_loss_counter = 0
                curr_lr = min(curr_lr * 2, lr)
            
            grad = U_k @ projected_error @ Vh_k

            W_q_refined = W_q_refined - curr_lr * grad

            pbar.set_postfix({"loss": f"{loss.item():.2e}"})

        final_tensor = best_tensor if best_tensor is not None else W_q_refined

        # Final Hard Quantization
        with torch.no_grad():
            W_f8 = final_tensor.to(TARGET_FP8_DTYPE)

        # Calculate dequantization scale (reciprocal of the quantization scale)
        dequant_scale = scale.reciprocal().reshape(1)
        # Clean up GPU memory
        del W_float32, W_scaled, W_rounded, W_q_refined, W_dq_rounded, error, U, Vh, U_k, Vh_k
        gc.collect()
        if self.device == 'cuda':
            torch.cuda.empty_cache()

        return W_f8.cpu(), dequant_scale.cpu(), (W_f8.to(COMPUTE_DTYPE) * dequant_scale).cpu()

def get_fp8_constants(fp8_dtype: torch.dtype) -> Tuple[float, float, float]:
    """Gets the min, max, and smallest positive normal value for a given FP8 dtype."""
    finfo = torch.finfo(fp8_dtype)
    return float(finfo.min), float(finfo.max), float(finfo.tiny)

# Global FP8 constants
FP8_MIN, FP8_MAX, FP8_MIN_POS = get_fp8_constants(TARGET_FP8_DTYPE)

def convert_to_fp8_scaled(input_file: str, output_file: str, t5xxl: bool, keep_distillation: bool, calib_samples: int, **converter_kwargs):
    """
    Converts a safetensors file to a version with FP8 scaled weights using learned rounding (modified from AdaRound).
    """
    print(f"Processing: {input_file}")
    print(f"Output will be saved to: {output_file}")
    print(f"Using FP8 format: {TARGET_FP8_DTYPE}")
    print(f"FP8 Range: [{FP8_MIN}, {FP8_MAX}]")
    print(f"FP8 Min Precision: [{FP8_MIN_POS}]")

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
                    calib_samples, in_features, dtype=COMPUTE_DTYPE # Use bf16 for realistic inputs, but COMPUTE_DTYPE should work? Unsure if this even matters.
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
            new_tensors[key] = tensors[key].to(TARGET_FP8_DTYPE) # Store as empty FP8
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
        quantized_fp8_tensor, dequant_scale, dequantized_weight_tensor = converter.convert(original_tensor, calibration_data)

        # Store the results
        new_tensors[key] = quantized_fp8_tensor
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
                # Output error: (N, C_out) = (N, C_in) @ (C_in, C_out).T
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
        print(f"  - Weight  : {quantized_fp8_tensor}")

    # Combine original non-weight tensors with new/modified ones
    for key, tensor in tensors.items():
        if (any(avoid_name in key for avoid_name in T5XXL_REMOVE_KEY_NAMES) and t5xxl):
            print(f"(+) Skipping decoder tensor: {key}")
            continue
        if key not in new_tensors:
            new_tensors[key] = tensor
            print(f"(+) Adding original non-quantized tensor: {key}")

    new_tensors["scaled_fp8"] = torch.empty((2), dtype=TARGET_FP8_DTYPE) if not t5xxl else torch.empty((0), dtype=TARGET_FP8_DTYPE)

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
        description=f"Convert safetensors weights to Scaled {TARGET_FP8_DTYPE} format using learned rounding, adapted from AdaRound.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # Original arguments
    parser.add_argument("--input", type=str, required=True, help="Input safetensors file path.")
    parser.add_argument("--output", type=str, help="Output safetensors file path. If not provided, generated based on input name.")
    parser.add_argument("--keep_distillation", action='store_true', help="Exclude distillation layers from quantization. \n(Likely not helpful because ComfyUI may use Round-to-Nearest in place of this, which SUXASS.)")
    parser.add_argument("--t5xxl", action='store_true', help="Exclude certain layers for T5XXL model compatibility.")

    parser.add_argument("--calib_samples", type=int, default=3072, help="Number of random samples for calibration.") # Random calibration samples for bias correction
    parser.add_argument("--num_iter", type=int, default=500, help="Number of optimization iterations per tensor.")

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        return

    # Check for FP8 support
    try:
        _ = torch.zeros(1, dtype=TARGET_FP8_DTYPE)
    except (RuntimeError, TypeError):
        print("Error: This version of PyTorch or this hardware does not support torch.float8_e4m3fn.")
        return

    fp8_type_str = TARGET_FP8_DTYPE.__str__().split('.')[-1]
    distill_str = "_nodistill" if args.keep_distillation else ""
    if not args.output:
        base_name = os.path.splitext(args.input)[0]
        output_file = f"{base_name}_{fp8_type_str}_scaled_learned{distill_str}_svd.safetensors"
    else:
        output_file = args.output

    if os.path.abspath(args.input) == os.path.abspath(output_file):
        print("Error: Output file cannot be the same as the input file.")
        return

    # Pass learned rounding hyperparameters to the conversion function
    converter_kwargs = {
        'num_iter': args.num_iter,
    }

    convert_to_fp8_scaled(
        args.input,
        output_file,
        args.t5xxl,
        args.keep_distillation,
        args.calib_samples,
        **converter_kwargs
    )

if __name__ == "__main__":
    main()