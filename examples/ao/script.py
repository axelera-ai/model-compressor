# Copyright (c) 2026 Axelera AI
# SPDX-License-Identifier: MIT

import os

import torch
import torchvision.models as models
import modelopt.torch.quantization as mtq
from modelopt.torch._deploy.utils import OnnxBytes, get_onnx_bytes_and_metadata


def export_to_onnx(model, input_shape, onnx_save_path, device, weights_dtype="fp32"):
    """Export the torch model to ONNX format."""
    # Create input tensor with same precision as model's first parameter
    input_dtype = model.parameters().__next__().dtype
    input_tensor = torch.randn(input_shape, dtype=input_dtype).to(device)
    model_name = os.path.basename(onnx_save_path).replace(".onnx", "")

    onnx_bytes, _ = get_onnx_bytes_and_metadata(
        model=model,
        dummy_input=(input_tensor,),
        weights_dtype=weights_dtype,
        model_name=model_name,
    )
    onnx_bytes_obj = OnnxBytes.from_bytes(onnx_bytes)

    # Write the onnx model to the specified directory without cleaning the directory
    onnx_bytes_obj.write_to_disk(os.path.dirname(onnx_save_path), clean_dir=False)


def main():
    print("Initializing model and synthetic data...")
    # 1. Load the pre-trained ShuffleNetV2 model and push to GPU
    model = models.shufflenet_v2_x1_0(pretrained=True).cuda().eval()

    # 2. Define the exact TensorRT PTQ configuration recipe
    quant_cfg = mtq.INT8_DEFAULT_CFG.copy()
    
    # CRITICAL LAYER FUSION RULE: Skip channel manipulation layers to maximize TensorRT throughput
    quant_cfg["quant_cfg"].append({"quantizer_name": "*cat*", "enable": False})
    quant_cfg["quant_cfg"].append({"quantizer_name": "*split*", "enable": False})
    quant_cfg["quant_cfg"].append({"quantizer_name": "*shuffle*", "enable": False})

    # 3. Define a synthetic calibration loop using fake GPU tensors
    # Since accuracy is ignored, random tensors are sufficient to compute valid QDQ scale factors
    def fake_calibrate_loop(quantized_model):
        with torch.no_grad():
            for _ in range(16):  # Stream 16 iterations * batch size 32 = 512 total iterations
                fake_batch = torch.randn(32, 3, 224, 224).cuda()
                quantized_model(fake_batch)

    # 4. Inject QDQ nodes and run the calibration execution 
    print("Running INT8 QDQ calibration with synthetic data...")
    calib_model = mtq.quantize(model, quant_cfg, forward_loop=fake_calibrate_loop)
    print("Calibration complete.")

    # 5. Export the structurally optimized graph to ONNX
    dummy_input = torch.randn(1, 3, 224, 224).cuda()
    onnx_path = "shufflenetv2_lat_test_int8.onnx"
    
    export_to_onnx(
        model=calib_model,
        input_shape=dummy_input.shape,
        onnx_save_path=onnx_path,
        device=dummy_input.device,
    )
    print(f"Successfully generated profiling model at: {onnx_path}")
    print("\nExecute the following terminal command to profile real hardware latency:")
    print(f"trtexec --onnx={onnx_path} --int8 --explicitPrecision --avgRuns=100")

if __name__ == "__main__":
    main()
