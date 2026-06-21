#!/usr/bin/env python3
"""Tiny PyTorch CPU training smoke: prove a 1-layer net trains with no compiler.

Fits a single nn.Linear to a known linear target with SGD and asserts the loss
drops substantially. Pure CPU, no CUDA, no Visual Studio. Independent of Unsloth.
"""
import sys
import torch
import torch.nn as nn

print("torch", torch.__version__, "| cuda?", torch.cuda.is_available(),
      "| device backend OK")
torch.manual_seed(0)

# y = 3x + 2  (the 1-layer net must learn weight~3, bias~2)
x = torch.linspace(-1, 1, 256).unsqueeze(1)
y = 3.0 * x + 2.0

net = nn.Linear(1, 1)            # exactly one layer
opt = torch.optim.SGD(net.parameters(), lr=0.1)
loss_fn = nn.MSELoss()

first = None
for step in range(300):
    opt.zero_grad()
    loss = loss_fn(net(x), y)
    loss.backward()
    opt.step()
    if first is None:
        first = loss.item()
last = loss.item()

w = net.weight.item()
b = net.bias.item()
print(f"loss: {first:.4f} -> {last:.6f} | learned w={w:.3f} (want 3) b={b:.3f} (want 2)")

ok = last < first * 0.01 and abs(w - 3.0) < 0.2 and abs(b - 2.0) < 0.2
if not ok:
    print("::error::CPU training did not converge as expected")
    sys.exit(1)
print("TORCH CPU TRAIN OK: 1-layer net trained on CPU with no Visual Studio.")
