from typing import Tuple
import json
import os
import sys

import torch
import torch.nn.functional as F
import triton

from vllm.model_executor.layers.fused_moe import fused_moe, get_config_file_name

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# torch._inductor.config.coordinate_descent_tuning = True

# Mixtral-8x7B configs
D_MODEL = 4096
N_EXPERTS = 8
TOP_K = 2
INTERMEDIATE_SIZE = 14336
N_LAYERS = 32


def conditional_linear(x, w, expert_indices):
    if expert_indices.shape[0] <= 2:
        w_weights = w[expert_indices].view(-1, *w.shape[-2:])  # [T, A, O, I]
        return torch.einsum("ti, toi -> to", x, w_weights)
    else:
        dense_out = torch.einsum("ti, eoi -> teo", x, w)
        one_hot_indices = torch.nn.functional.one_hot(
            expert_indices.view(-1),
            num_classes=N_EXPERTS).to(dtype=dense_out.dtype)
        return torch.einsum("teo, te -> to", dense_out, one_hot_indices)


def torch_moe(hidden_states, gating_output, w1, w2, w3, renormalize=True):
    # T = num_tokens, E = num_experts, D = hidden dim, A = activated experts
    # x: [T, D]
    # score: [T, E]
    x = hidden_states
    expert_weights = torch.nn.functional.softmax(gating_output, dim=-1)
    expert_weights, expert_indices = torch.topk(expert_weights, TOP_K,
                                                dim=-1)  # [T, A], [T, A]
    
    if renormalize:
        expert_weights /= expert_weights.sum(dim=-1, keepdim=True)  # [T, A]

    x = x.unsqueeze(1).expand(x.shape[0], expert_indices.shape[-1],
                              x.shape[-1])
    x = x.reshape(-1, x.shape[-1])

    x1 = conditional_linear(x, w1, expert_indices)
    x3 = conditional_linear(x, w3, expert_indices)
    x1 = torch.nn.functional.silu(x1) * x3

    expert_outs = conditional_linear(x1, w2, expert_indices)
    out = torch.einsum('tai,ta -> ti',
                        expert_outs.view(-1, TOP_K, expert_outs.shape[-1]),
                        expert_weights)
    return out


def main():
    method = fused_moe
    # for bs in [
    #         1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 256, 512, 1024, 1536,
    #         2048, 3072, 4096
    # ]:
    #     run_grid(bs, method=method)

    config = {
        "BLOCK_SIZE_M": 16,
        "BLOCK_SIZE_N": 32,
        "BLOCK_SIZE_K": 64,
        "GROUP_SIZE_M": 1,
        "num_warps": 4,
        "num_stages": 4,
    }
    check_output(bs=1, methods=[(fused_moe, True), (torch_moe, False)], config=config, tp_size=1)


def gen_tensors(bs, tp_size):
    shard_intermediate_size = INTERMEDIATE_SIZE // tp_size

    hidden_states = torch.rand(
        (bs, D_MODEL),
        device="cuda:0",
        dtype=torch.bfloat16,
    )

    ws = torch.rand(
        (N_EXPERTS, 2 * shard_intermediate_size, D_MODEL),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    ) / 1e3

    w2s = torch.rand(
        (N_EXPERTS, D_MODEL, shard_intermediate_size),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    ) / 1e3

    gating_output = torch.rand(
        (bs, N_EXPERTS),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )

    return gating_output, hidden_states, ws, w2s


def run_grid(bs,
             method,
             tp_size=1,
             num_calls=100,
             num_warmup_trials=1,
             num_trials=1):
    configs = []
    if bs <= 16:
        BLOCK_SIZES_M = [16]
    elif bs <= 32:
        BLOCK_SIZES_M = [16, 32]
    elif bs <= 64:
        BLOCK_SIZES_M = [16, 32, 64]
    elif bs <= 128:
        BLOCK_SIZES_M = [16, 32, 64, 128]
    else:
        BLOCK_SIZES_M = [16, 32, 64, 128, 256]

    for block_size_n in [32, 64, 128, 256]:
        for block_size_m in BLOCK_SIZES_M:
            for block_size_k in [64, 128, 256]:
                for group_size_m in [1, 16, 32, 64]:
                    for num_warps in [4, 8]:
                        configs.append({
                            "BLOCK_SIZE_M": block_size_m,
                            "BLOCK_SIZE_N": block_size_n,
                            "BLOCK_SIZE_K": block_size_k,
                            "GROUP_SIZE_M": group_size_m,
                            "num_warps": num_warps,
                            "num_stages": 4,
                        })

    best_config = None
    best_time_us = 1e20

    for config in configs:
        print(f"{tp_size=} {bs=}")
        print(f"{config}")

        gating_output, hs, ws, w2s = gen_tensors(bs, tp_size)

        # warmup
        print("warming up")
        try:
            for _ in range(num_warmup_trials):
                run_timing(
                    gating_output,
                    hs,
                    ws,
                    w2s,
                    num_calls=num_calls,
                    method=method,
                    config=config,
                )
        except triton.runtime.autotuner.OutOfResources:
            continue

        # trial
        print("benchmarking")
        for _ in range(num_trials):
            kernel_dur_ms, _ = run_timing(
                gating_output,
                hs,
                ws,
                w2s,
                num_calls=num_calls,
                method=method,
                config=config,
            )
            if kernel_dur_us < best_time_us:
                best_config = config
                best_time_us = kernel_dur_us

            kernel_dur_us = 1000 * kernel_dur_ms
            model_dur_ms = kernel_dur_ms * N_LAYERS

            print(f"{kernel_dur_us=:.1f} {model_dur_ms=:.1f}"
                  f" {bs=} {tp_size=} {TOP_K=} {N_EXPERTS=} "
                  f"{D_MODEL=} {INTERMEDIATE_SIZE=} {N_LAYERS=}")

    print("best_time_us", best_time_us)
    print("best_config", best_config)

    # holds Dict[str, Dict[str, int]]
    filename = get_config_file_name(N_EXPERTS, INTERMEDIATE_SIZE // tp_size)
    print(f"writing config to file {filename}")
    existing_content = {}
    if os.path.exists(filename):
        with open(filename, "r") as f:
            existing_content = json.load(f)
    existing_content[str(bs)] = best_config
    with open(filename, "w") as f:
        json.dump(existing_content, f, indent=4)
        f.write("\n")


def check_output(bs, methods, config, tp_size):
    gating_output, hs, ws, w2s = gen_tensors(bs, tp_size)

    outputs = []
    for method, merged_gated in methods:
        if merged_gated:
            out = method(
                hidden_states=hs,
                w1=ws,
                w2=w2s,
                gating_output=gating_output,
                topk=TOP_K,
                renormalize=True,
                inplace=False,
                override_config=config,
            )
        else:
            w1, w3 = ws.chunk(2, dim=1)
            out = method(
                hidden_states=hs,
                w1=w1,
                w2=w2s,
                w3=w3,
                gating_output=gating_output,
                renormalize=True,
            )
        outputs.append((method.__name__, out))

    ref_method, ref_out = outputs[0]
    print(f"Reference output ({ref_out.shape}) from method {ref_method} with error 1e-2")
    for method_name, out in outputs[1:]:
        err_idxs = torch.nonzero(torch.abs(out - ref_out) > 1e-2).squeeze()
        print(f"{method_name} has the {err_idxs.shape[0]} / {ref_out.numel()} different indices:")
        print(err_idxs)


def run_timing(
    gating_output,
    hidden_states,
    ws,
    w2s,
    num_calls: int,
    method,
    config,
) -> Tuple[float, torch.Tensor]:
    bs = hidden_states.shape[0]

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for i in range(num_calls):
        hidden_states = method(
            hidden_states=hidden_states,
            w1=ws,
            w2=w2s,
            gating_output=gating_output,
            topk=TOP_K,
            renormalize=True,
            inplace=True,
            override_config=config,
        )
    end_event.record()
    end_event.synchronize()

    dur_ms = start_event.elapsed_time(end_event) / num_calls
    return dur_ms, hidden_states


if __name__ == "__main__":
    sys.exit(main())
