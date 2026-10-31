#!/bin/sh
# Run from examples/model-compressor. Builds each ONNX with TensorRT (explicit Q/DQ,
# strongly-typed so the Q/DQ fix types) and exports per-iteration timings + layer info.
D=trt/siglip
for V in strict permissive; do
  trtexec --onnx=$D/${V}_W.onnx --stronglyTyped \
    --dumpProfile=$D/${V}_W_dumpProfile.json \
    --exportLayerInfo=$D/${V}_W_layerInfo.json \
    --exportTimes=$D/${V}_W_times.json \
    --profilingVerbosity=detailed --iterations=512 --warmUp=256
done
