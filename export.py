import torch
from train import Net
import numpy as np
import math

model = Net()
model.load_state_dict(torch.load("weights/model.pth", map_location="cpu"))
model.eval()

def quantize(t):
    max_val = t.abs().max().item()
    scale = 127.0 / max_val if max_val != 0 else 1.0
    q = (t * scale).round().clamp(-128, 127).to(torch.int8)
    return q.numpy(), float(scale)

fc1_w = model.fc1.weight.data
fc1_b = model.fc1.bias.data
fc2_w = model.fc2.weight.data
fc2_b = model.fc2.bias.data

FC1_OUT, INPUT_SIZE = fc1_w.shape
OUTPUT_SIZE, _ = fc2_w.shape

# Quantize weights FIRST so s1, s2 exist
fc1_w_q, s1 = quantize(fc1_w)
fc2_w_q, s2 = quantize(fc2_w)

fc1_b_q = (fc1_b * s1).round().to(torch.int32).numpy()
fc2_b_q = (fc2_b * s2).round().to(torch.int32).numpy()

# NOW compute quant params (s1, s2 are defined)
def compute_cmsis_quant_params(input_scale, weight_scale, output_scale=1.0/127.0):
    real_scale = (float(input_scale) * float(weight_scale)) / float(output_scale)
    shift = 0
    multiplier = real_scale
    while multiplier < 0.5:
        multiplier *= 2
        shift += 1
    while multiplier >= 1.0:
        multiplier /= 2
        shift -= 1
    multiplier_int = int(round(multiplier * (2**31)))
    return multiplier_int, shift

input_scale = 1.0 / 127.0
fc1_quant_mult, fc1_quant_shift = compute_cmsis_quant_params(input_scale, 1.0/s1)
fc2_quant_mult, fc2_quant_shift = compute_cmsis_quant_params(1.0/127.0, 1.0/s2)

print(f"FC1 quant: multiplier={fc1_quant_mult}, shift={fc1_quant_shift}")
print(f"FC2 quant: multiplier={fc2_quant_mult}, shift={fc2_quant_shift}")

with open("model_weights.h", "w") as f:
    f.write("#ifndef MODEL_WEIGHTS_H\n#define MODEL_WEIGHTS_H\n\n")
    f.write("#include <stdint.h>\n\n")

    f.write(f"const int8_t fc1_weights[{FC1_OUT}][{INPUT_SIZE}] = {{\n")
    for row in fc1_w_q:
        f.write("{" + ",".join(map(str, row)) + "},\n")
    f.write("};\n\n")

    f.write(f"const int32_t fc1_bias[{FC1_OUT}] = {{")
    f.write(",".join(map(str, fc1_b_q)))
    f.write("};\n\n")

    f.write(f"const int8_t fc2_weights[{OUTPUT_SIZE}][{FC1_OUT}] = {{\n")
    for row in fc2_w_q:
        f.write("{" + ",".join(map(str, row)) + "},\n")
    f.write("};\n\n")

    f.write(f"const int32_t fc2_bias[{OUTPUT_SIZE}] = {{")
    f.write(",".join(map(str, fc2_b_q)))
    f.write("};\n\n")

    # Write quant params directly into the header so model.c can use them
    f.write(f"#define FC1_QUANT_MULTIPLIER {fc1_quant_mult}\n")
    f.write(f"#define FC1_QUANT_SHIFT      {fc1_quant_shift}\n")
    f.write(f"#define FC2_QUANT_MULTIPLIER {fc2_quant_mult}\n")
    f.write(f"#define FC2_QUANT_SHIFT      {fc2_quant_shift}\n")
    f.write("#endif\n")

print("Export DONE — model_weights.h written")