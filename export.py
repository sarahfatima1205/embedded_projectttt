import torch
from train import Net
import numpy as np

model = Net()
model.load_state_dict(torch.load("weights/model.pth", map_location="cpu"))
model.eval()

def quantize(t):
    max_val = t.abs().max()
    scale = 127.0 / max_val if max_val != 0 else 1.0
    q = (t * scale).round().clamp(-128, 127).to(torch.int8)
    return q.numpy(), scale

fc1_w = model.fc1.weight.data
fc1_b = model.fc1.bias.data
fc2_w = model.fc2.weight.data
fc2_b = model.fc2.bias.data

# 🔥 AUTO dimensions
FC1_OUT, INPUT_SIZE = fc1_w.shape
OUTPUT_SIZE, _ = fc2_w.shape

fc1_w_q, s1 = quantize(fc1_w)
fc2_w_q, s2 = quantize(fc2_w)

fc1_b_q = (fc1_b * s1).round().to(torch.int32).numpy()
fc2_b_q = (fc2_b * s2).round().to(torch.int32).numpy()

with open("model_weights.h", "w") as f:
    f.write("#ifndef MODEL_WEIGHTS_H\n#define MODEL_WEIGHTS_H\n\n")
    f.write("#include <stdint.h>\n\n")

    # FC1
    f.write(f"const int8_t fc1_weights[{FC1_OUT}][{INPUT_SIZE}] = {{\n")
    for row in fc1_w_q:
        f.write("{" + ",".join(map(str, row)) + "},\n")
    f.write("};\n\n")

    f.write(f"const int32_t fc1_bias[{FC1_OUT}] = {{")
    f.write(",".join(map(str, fc1_b_q)))
    f.write("};\n\n")

    # FC2
    f.write(f"const int8_t fc2_weights[{OUTPUT_SIZE}][{FC1_OUT}] = {{\n")
    for row in fc2_w_q:
        f.write("{" + ",".join(map(str, row)) + "},\n")
    f.write("};\n\n")

    f.write(f"const int32_t fc2_bias[{OUTPUT_SIZE}] = {{")
    f.write(",".join(map(str, fc2_b_q)))
    f.write("};\n\n")

    f.write("#endif\n")

print("Export DONE")