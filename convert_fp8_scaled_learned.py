import argparse
import os
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from typing import Dict, Tuple
from tqdm import tqdm
import gc

#from GOODDOG import GOODDOG

# Keys containing these strings will not be quantized if --t5xxl is set
AVOID_KEY_NAMES = ["norm", "bias", "embed_tokens", "shared"] #T5XXL, may need to be changed for other TEs.
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
    """
    def __init__(self, num_iter=500, lr=1e-3, reg_lambda=0.01, beta_start=20, beta_end=2):
        self.num_iter = num_iter
        self.lr = lr
        self.reg_lambda = reg_lambda
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # The maximum representable value for e4m3fn, used for scaling.
        self.f8_max_val = torch.finfo(TARGET_FP8_DTYPE).max
        print(f"LearnedRoundingConverter initialized on device: {self.device}")

    def convert(self, W_orig: torch.Tensor, X_calib: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Performs the learned rounding conversion for a single weight tensor.
        """
        W_float32 = W_orig.to(self.device, dtype=COMPUTE_DTYPE)
        X_calib = X_calib.to(self.device, dtype=COMPUTE_DTYPE)

        # Step 1: Calculate the quantization scale (per-tensor asymmetric)
        w_max = W_float32.abs().max()
        if w_max < 1e-12:
            print("  - Tensor is all zeros, skipping optimization.")
            scale = torch.tensor(1.0, device=self.device)
            quantized_tensor = torch.zeros_like(W_float32, dtype=TARGET_FP8_DTYPE)
            return quantized_tensor.cpu(), scale.reciprocal().cpu().reshape(1)

        scale = self.f8_max_val / w_max # Example: (absmax = 1, fp8 max = +-448 for dtype e4m3_fn)
        W_scaled = W_float32 * scale # absmax now +-448

        # Step 2: Initialize the rounding mask 'h'
        h_init = W_scaled - (torch.floor(W_scaled / FP8_MIN_POS) * FP8_MIN_POS)
        h = torch.nn.Parameter(h_init)

        # Step 3: Setup optimizer and beta schedule (annealing)
        #optimizer = GOODDOG([h], lr=self.lr, adaptive_muon=False, invariant=True) # https://github.com/Clybius/Personalized-Optimizers
        optimizer = torch.optim.RMSprop([h], lr=self.lr)
        beta_schedule = torch.linspace(self.beta_start, self.beta_end, self.num_iter).to(self.device)

        # Step 4: The optimization loop
        pbar = tqdm(range(self.num_iter), desc="    Optimizing rounding", leave=False)
        for i in pbar:
            beta = beta_schedule[i]
            W_soft_quant = ((torch.floor(W_scaled / FP8_MIN_POS) * FP8_MIN_POS) + h) / scale
            Y_orig = X_calib @ W_float32.T
            Y_quant = X_calib @ W_soft_quant.T
            recon_loss = (Y_orig - Y_quant).pow(2).mean()
            reg_loss = self.reg_lambda * torch.sum(1 - torch.abs(2 * h/FP8_MIN_POS - 1).pow(beta))
            total_loss = recon_loss + reg_loss
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            with torch.no_grad():
                h.clamp_(0, FP8_MIN_POS)
            pbar.set_postfix({"Recon Loss": f"{recon_loss.item():.4e}", "Reg Loss": f"{reg_loss.item():.4e}"})
            if reg_loss.item() < 1e-8:
                break

        # Step 5: Final Hard Quantization
        with torch.no_grad():
            W_quant_final_scaled = (torch.floor(W_scaled / FP8_MIN_POS) * FP8_MIN_POS) + h.data
            W_f8 = W_quant_final_scaled.to(dtype=TARGET_FP8_DTYPE)

        # Calculate dequantization scale (inverse of the quantization scale)
        dequant_scale = scale.reciprocal().reshape(1)
        # Clean up GPU memory
        del W_float32, X_calib, h, optimizer, Y_orig, Y_quant, W_soft_quant, W_scaled
        gc.collect()
        if self.device == 'cuda':
            torch.cuda.empty_cache()

        return W_f8.cpu(), dequant_scale.cpu()

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
                    calib_samples, in_features, dtype=torch.bfloat16 # Use bf16 for realistic inputs, but COMPUTE_DTYPE should work? Unsure if this even matters.
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
        if t5xxl and any(avoid_name in key for avoid_name in AVOID_KEY_NAMES):
            print(f"({i+1}/{total_weights}) Skipping excluded T5XXL tensor: {key}")
            new_tensors[key] = tensors[key]
            process_this_key = False
            skipped_count += 1

        if keep_distillation and any(avoid_name in key for avoid_name in ["distilled_guidance_layer", "final_layer", "img_in", "txt_in"]):
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
        quantized_fp8_tensor, dequant_scale = converter.convert(original_tensor, calibration_data)

        # Store the results
        new_tensors[key] = quantized_fp8_tensor
        base_name = key[:-len('.weight')]
        scale_weight_key = f"{base_name}.scale_weight"
        new_tensors[scale_weight_key] = dequant_scale.to(SCALE_DTYPE)
        if t5xxl:
            scale_input_key = f"{base_name}.scale_input"
            new_tensors[scale_input_key] = dequant_scale.detach().clone().to(SCALE_DTYPE)

        print(f"  - Dequant Scale  : {dequant_scale.item():.9}")
        print(f"  - Weight  : {quantized_fp8_tensor}")

    # Combine original non-weight tensors with new/modified ones
    for key, tensor in tensors.items():
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

    parser.add_argument("--calib_samples", type=int, default=256, help="Number of random samples for calibration.") # Might need modification, I'm unsure if this is too little or too high, but it works fine.
    parser.add_argument("--num_iter", type=int, default=512, help="Number of optimization iterations per tensor.") # Good nuff for GOODDOG optimizer (@ my personalized optims github) or RMSprop @ 1e-2 lr
    parser.add_argument("--lr", type=float, default=1e-2, help="Learning rate for the rounding optimizer.")
    parser.add_argument("--reg_lambda", type=float, default=0.01, help="Regularization strength for the rounding loss.")

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
        output_file = f"{base_name}_{fp8_type_str}_scaled_learned{distill_str}.safetensors"
    else:
        output_file = args.output

    if os.path.abspath(args.input) == os.path.abspath(output_file):
        print("Error: Output file cannot be the same as the input file.")
        return

    # Pass learned rounding hyperparameters to the conversion function
    converter_kwargs = {
        'num_iter': args.num_iter,
        'lr': args.lr,
        'reg_lambda': args.reg_lambda,
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