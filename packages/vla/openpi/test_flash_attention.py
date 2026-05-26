#!/usr/bin/env python3
# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Comprehensive flash-attention sanity check for the ROCm Triton-AMD backend."""

import os
import importlib
import pkgutil
import traceback

import torch
import torch.nn.functional as F


def section(title):
    print("\n" + "=" * 8 + f" {title} " + "=" * 8)


# ---------------------------------------------------------------------------
# 1. Environment + package info
# ---------------------------------------------------------------------------
section("env / package")
print("FLASH_ATTENTION_TRITON_AMD_ENABLE =",
      os.environ.get("FLASH_ATTENTION_TRITON_AMD_ENABLE"))
print("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL =",
      os.environ.get("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"))
print("GPU_ARCHS =", os.environ.get("GPU_ARCHS"))
print("torch:", torch.__version__, "cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
    print("hip/cuda runtime:", getattr(torch.version, "hip", None) or torch.version.cuda)

import flash_attn
print("flash_attn version:", flash_attn.__version__)
print("flash_attn path   :", flash_attn.__file__)

# List submodules so we can see what backend modules are actually shipped
subs = [m.name for m in pkgutil.iter_modules(flash_attn.__path__)]
print("flash_attn submodules:", subs)

# ---------------------------------------------------------------------------
# 2. Backend module probe (informational — don't fail on these)
# ---------------------------------------------------------------------------
section("backend probe")
for m in (
    "flash_attn_2_cuda",        # CUDA C++ ext (absent on ROCm)
    "flash_attn_2_rocm",        # legacy ROCm C++ ext (absent on Triton-AMD)
    "flash_attn.flash_attn_triton",
    "flash_attn.flash_attn_triton_amd",
):
    try:
        importlib.import_module(m)
        print("  loaded :", m)
    except Exception as e:
        print("  missing:", m, "->", type(e).__name__)

# ---------------------------------------------------------------------------
# 3. flash_attn_func: forward, backward, parity vs SDPA
# ---------------------------------------------------------------------------
section("flash_attn_func fwd/bwd + SDPA parity")
from flash_attn import flash_attn_func

device = "cuda"
results = []

for dtype in (torch.float16, torch.bfloat16):
    for causal in (False, True):
        B, S, H, D = 2, 256, 8, 64
        q = torch.randn(B, S, H, D, device=device, dtype=dtype, requires_grad=True)
        k = torch.randn(B, S, H, D, device=device, dtype=dtype, requires_grad=True)
        v = torch.randn(B, S, H, D, device=device, dtype=dtype, requires_grad=True)

        out = flash_attn_func(q, k, v, causal=causal)
        finite_fwd = torch.isfinite(out).all().item()

        out.sum().backward()
        finite_bwd = all(torch.isfinite(t.grad).all().item() for t in (q, k, v))

        # SDPA reference in fp32 for a clean comparison
        qh, kh, vh = (t.detach().float().transpose(1, 2) for t in (q, k, v))
        ref = F.scaled_dot_product_attention(qh, kh, vh, is_causal=causal).transpose(1, 2)
        diff = (out.detach().float() - ref).abs().max().item()

        tol = 5e-2  # fp16/bf16 roundoff with this shape
        ok = finite_fwd and finite_bwd and diff < tol
        print(f"  dtype={str(dtype):>15}  causal={causal!s:<5}  "
              f"max|FA-SDPA|={diff:.3e}  fwd_finite={finite_fwd}  "
              f"bwd_finite={finite_bwd}  {'OK' if ok else 'FAIL'}")
        results.append(ok)

# ---------------------------------------------------------------------------
# 4. Varlen path (used by HF when there's padding)
# ---------------------------------------------------------------------------
section("flash_attn_varlen_func")
try:
    from flash_attn import flash_attn_varlen_func

    seqlens = torch.tensor([128, 200, 96, 256], device=device, dtype=torch.int32)
    cu = torch.cat([torch.zeros(1, device=device, dtype=torch.int32),
                    torch.cumsum(seqlens, dim=0, dtype=torch.int32)])
    total = int(seqlens.sum())
    H, D = 8, 64
    q = torch.randn(total, H, D, device=device, dtype=torch.float16, requires_grad=True)
    k = torch.randn(total, H, D, device=device, dtype=torch.float16, requires_grad=True)
    v = torch.randn(total, H, D, device=device, dtype=torch.float16, requires_grad=True)

    out = flash_attn_varlen_func(
        q, k, v,
        cu_seqlens_q=cu, cu_seqlens_k=cu,
        max_seqlen_q=int(seqlens.max()), max_seqlen_k=int(seqlens.max()),
        causal=True,
    )
    out.sum().backward()
    print("  varlen out:", out.shape, "finite:",
          torch.isfinite(out).all().item(),
          "grads finite:",
          all(torch.isfinite(t.grad).all().item() for t in (q, k, v)))
    results.append(True)
except Exception:
    traceback.print_exc()
    results.append(False)

# ---------------------------------------------------------------------------
# 5. HF transformers detection
# ---------------------------------------------------------------------------
section("transformers detection")
try:
    from transformers.utils.import_utils import is_flash_attn_2_available
    print("  is_flash_attn_2_available():", is_flash_attn_2_available())
except Exception as e:
    print("  could not probe transformers:", type(e).__name__, e)

# ---------------------------------------------------------------------------
# 6. Quick microbench: FA vs SDPA on a PaliGemma-ish shape
# ---------------------------------------------------------------------------
section("microbench (PaliGemma-ish: B=1, S=1024, H=8, D=128, fp16)")
B, S, H, D = 1, 1024, 8, 128
q = torch.randn(B, S, H, D, device=device, dtype=torch.float16)
k = torch.randn(B, S, H, D, device=device, dtype=torch.float16)
v = torch.randn(B, S, H, D, device=device, dtype=torch.float16)

def bench(fn, iters=50, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters  # ms

def fa_call():
    return flash_attn_func(q, k, v, causal=True)

def sdpa_call():
    qh, kh, vh = (t.transpose(1, 2) for t in (q, k, v))
    return F.scaled_dot_product_attention(qh, kh, vh, is_causal=True)

try:
    print(f"  FA   : {bench(fa_call):.3f} ms/iter")
    print(f"  SDPA : {bench(sdpa_call):.3f} ms/iter")
except Exception:
    traceback.print_exc()

# ---------------------------------------------------------------------------
# 7. Is PI0 actually using FA?  (optional — only if the policy already loaded)
# ---------------------------------------------------------------------------
section("PI0 attention-impl audit (optional)")
try:
    from lerobot.policies.pi0 import PI0Policy
    policy = PI0Policy.from_pretrained("lerobot/pi0_base").to(device).eval()

    impls = set()
    for _, mod in policy.named_modules():
        impl = (getattr(mod, "_attn_implementation", None)
                or getattr(getattr(mod, "config", None), "_attn_implementation", None))
        if impl:
            impls.add(impl)
    print("  attention impls present:", impls or "<none reported>")

    # Drill into the PaliGemma sub-config if reachable
    for path in (
        "model.paligemma_with_expert.paligemma.config",
        "paligemma_with_expert.paligemma.config",
    ):
        obj = policy
        try:
            for part in path.split("."):
                obj = getattr(obj, part)
            print(f"  {path}._attn_implementation =",
                  getattr(obj, "_attn_implementation", "<unset>"))
            break
        except AttributeError:
            continue
except Exception:
    print("  (skipped — couldn't load PI0)")
    traceback.print_exc()

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
section("summary")
print("kernel checks passed:", sum(results), "/", len(results))
if not all(results):
    raise SystemExit(1)
print("flash-attention looks healthy.")