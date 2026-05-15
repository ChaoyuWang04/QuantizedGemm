"""
qwen_w8a8_test.py - 端到端测试 Qwen2.5-0.5B + W8A8 集成.

四个 milestone (每个 --milestone N 独立跑):
    1: 加载 fp16 模型, 跑通 forward, 测 prompt 生成
    2: 加载 fp16 + 转 W8A8, 跑通 forward
    3: 同 prompt 下 fp16 vs W8A8 生成对比 (定性)
    4: WikiText-2 子集上 perplexity 对比 (定量)

执行:
    python experiments/qwen_w8a8_test.py --milestone 1     # 先跑 1 确保模型可用
    python experiments/qwen_w8a8_test.py --milestone 2
    python experiments/qwen_w8a8_test.py --milestone 3
    python experiments/qwen_w8a8_test.py --milestone 4
    python experiments/qwen_w8a8_test.py --milestone all   # 全跑 (10-15 分钟)

环境:
    - transformers >= 4.37 (Qwen2 支持)
    - 5090 显存约 1-2GB 即可

Note:
    第一次跑 W8A8 forward 会触发 ~4 次 autotune, 每次 20-40s, 共 2-3 分钟.
    后续调用秒级.
"""

import argparse
import math
import os
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))


# 设置 HF cache 到持久化盘 (RunPod 重启不丢)
DEFAULT_HF_CACHE = "/autodl-fs/hf_cache"
if Path("/autodl-fs").exists():
    os.environ.setdefault("HF_HOME", DEFAULT_HF_CACHE)

MODEL_NAME = "Qwen/Qwen2.5-0.5B"


# ============================================================
# Milestone 1: 加载 fp16 模型, 验证环境
# ============================================================

def milestone_1():
    print("=" * 75)
    print("Milestone 1: Load fp16 Qwen2.5-0.5B, run forward")
    print("=" * 75)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"\nLoading {MODEL_NAME} ...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="cuda",
    )
    print(f"Loaded in {time.time()-t0:.1f}s")
    print(f"Model size: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params")

    # 简单 forward 验证
    prompt = "The capital of France is"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    print(f"\nPrompt: {prompt!r}")
    print(f"Input ids shape: {inputs['input_ids'].shape}")

    with torch.no_grad():
        t0 = time.time()
        outputs = model.generate(
            **inputs,
            max_new_tokens=20,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
        gen_time = time.time() - t0

    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    new_tokens = outputs.shape[1] - inputs["input_ids"].shape[1]
    print(f"Generated ({gen_time:.2f}s, {new_tokens} tokens, "
          f"{new_tokens/gen_time:.1f} tok/s):")
    print(f"  {generated!r}")

    print("\nPASS: fp16 model works")
    return model, tokenizer


# ============================================================
# Milestone 2: 转 W8A8, 跑 forward
# ============================================================

def milestone_2():
    print("=" * 75)
    print("Milestone 2: Convert to W8A8, run forward")
    print("=" * 75)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from integration.convert import convert_model_to_w8a8

    print(f"\nLoading {MODEL_NAME} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="cuda",
    )
    print("Loaded")

    print("\nConverting to W8A8 (verbose=False, ~5s)...")
    t0 = time.time()
    stats = convert_model_to_w8a8(model, verbose=False)
    print(f"Converted in {time.time()-t0:.1f}s")
    print(f"  Converted: {len(stats['converted'])}")
    print(f"  Skipped:   {len(stats['skipped'])}")
    print(f"  Failed:    {len(stats['failed'])}")
    print(f"  Memory saved: {stats['total_params_saved']/1e6:.1f} MB")

    if stats["failed"]:
        print("\nWARN: Some layers failed to convert:")
        for name, err in stats["failed"][:5]:
            print(f"  {name}: {err}")

    # Forward verification
    prompt = "The capital of France is"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    print(f"\nPrompt: {prompt!r}")
    print("First forward (will trigger ~4 autotune calls, ~2 minutes)...")

    with torch.no_grad():
        t0 = time.time()
        # 单独跑 forward (不 generate) 看 logits 是否合理
        outputs = model(**inputs)
        fwd_time = time.time() - t0

    logits = outputs.logits   # [batch, seq, vocab]
    print(f"Forward done in {fwd_time:.1f}s")
    print(f"Output logits shape: {logits.shape}")
    print(f"Logits dtype: {logits.dtype}")
    print(f"Logits stats: mean={logits.mean().item():.4f}, "
          f"std={logits.std().item():.4f}, "
          f"max={logits.max().item():.4f}")

    # Sanity check: logits 应该是有意义的数值, 不该是 NaN/Inf
    has_nan = torch.isnan(logits).any().item()
    has_inf = torch.isinf(logits).any().item()
    if has_nan or has_inf:
        print(f"FAIL: logits contains NaN={has_nan} or Inf={has_inf}")
    else:
        print("PASS: logits are finite, W8A8 forward works")

    return model, tokenizer


# ============================================================
# Milestone 3: 生成对比 (fp16 vs W8A8)
# ============================================================

def milestone_3():
    print("=" * 75)
    print("Milestone 3: Compare generation (fp16 vs W8A8)")
    print("=" * 75)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from integration.convert import convert_model_to_w8a8

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    test_prompts = [
        "The capital of France is",
        "Once upon a time, there was a",
        "Python is a programming language that",
        "The most important invention of the 20th century was",
    ]

    # ---- fp16 model ----
    print("\nLoading fp16 model ...")
    model_fp16 = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="cuda",
    )
    model_fp16.eval()

    print("Generating with fp16...")
    fp16_outputs = []
    for prompt in test_prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = model_fp16.generate(
                **inputs, max_new_tokens=30,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        fp16_outputs.append(tokenizer.decode(out[0], skip_special_tokens=True))

    # 清理 fp16 model 释放显存
    del model_fp16
    torch.cuda.empty_cache()

    # ---- W8A8 model ----
    print("\nLoading W8A8 model (re-load + convert)...")
    model_w8a8 = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="cuda",
    )
    convert_model_to_w8a8(model_w8a8, verbose=False)
    model_w8a8.eval()

    print("Generating with W8A8 (first time triggers autotune)...")
    w8a8_outputs = []
    for i, prompt in enumerate(test_prompts):
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            t0 = time.time()
            out = model_w8a8.generate(
                **inputs, max_new_tokens=30,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
            t = time.time() - t0
        w8a8_outputs.append(tokenizer.decode(out[0], skip_special_tokens=True))
        if i == 0:
            print(f"  First gen (with autotune): {t:.1f}s")

    # ---- 对比 ----
    print("\n" + "=" * 75)
    print("Generation Comparison")
    print("=" * 75)
    matches = 0
    for prompt, fp16_o, w8a8_o in zip(test_prompts, fp16_outputs, w8a8_outputs):
        print(f"\nPrompt: {prompt!r}")
        print(f"  fp16: {fp16_o!r}")
        print(f"  w8a8: {w8a8_o!r}")
        if fp16_o == w8a8_o:
            print(f"  -> IDENTICAL")
            matches += 1
        else:
            # 计算前 N 个 token 一致性
            fp16_toks = tokenizer.encode(fp16_o)
            w8a8_toks = tokenizer.encode(w8a8_o)
            common = 0
            for a, b in zip(fp16_toks, w8a8_toks):
                if a == b:
                    common += 1
                else:
                    break
            print(f"  -> First {common} tokens match (of {len(fp16_toks)})")

    print(f"\n{matches}/{len(test_prompts)} prompts produce identical output")
    print("Note: Even with quantization noise, greedy decoding often agrees")
    print("      on early tokens but diverges later. This is expected.")


# ============================================================
# Milestone 4: Perplexity 对比
# ============================================================

@torch.no_grad()
def compute_perplexity(model, tokenizer, text: str, max_length: int = 512) -> float:
    """
    计算给定文本的 perplexity.
    
    简化版: 用一个连续文本, 按 max_length 截断, 计算每个位置的 cross-entropy.
    PPL = exp(平均 cross-entropy).
    """
    encodings = tokenizer(text, return_tensors="pt").to("cuda")
    input_ids = encodings["input_ids"]
    
    # 限制长度
    if input_ids.shape[1] > max_length:
        input_ids = input_ids[:, :max_length]

    target_ids = input_ids.clone()

    outputs = model(input_ids, labels=target_ids)
    neg_log_likelihood = outputs.loss
    
    # PPL = exp(loss)
    return math.exp(neg_log_likelihood.item())


def milestone_4():
    print("=" * 75)
    print("Milestone 4: Perplexity comparison")
    print("=" * 75)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from integration.convert import convert_model_to_w8a8

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # 用 WikiText 风格的几段文本测 PPL.
    # 不用完整 WikiText-2 (太大), 用嵌入式 sample 简化测试.
    test_texts = [
        # 这些是结构良好的英文段落, PPL 应该比较低
        "Machine learning is a subfield of artificial intelligence that focuses on the development of algorithms and statistical models that enable computer systems to learn from and make predictions on data. The field has grown rapidly in recent decades, driven by advances in computing power, the availability of large datasets, and breakthroughs in deep learning techniques. Modern applications include natural language processing, computer vision, recommendation systems, and autonomous vehicles.",
        
        "The Great Wall of China is one of the most iconic structures in the world. Built over many centuries by various Chinese dynasties, it stretches across mountains, deserts, and grasslands. The wall was constructed primarily for defense against nomadic invasions from the north. Today, it is a UNESCO World Heritage Site and a major tourist attraction, drawing millions of visitors each year.",
        
        "Climate change refers to long-term shifts in global temperatures and weather patterns. While natural factors contribute to climate variations, scientific consensus identifies human activities, particularly the burning of fossil fuels, as the primary driver of recent warming trends. The consequences include rising sea levels, more frequent extreme weather events, and disruptions to ecosystems worldwide.",
    ]

    # ---- fp16 PPL ----
    print("\nLoading fp16 and computing PPL...")
    model_fp16 = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16, device_map="cuda",
    )
    model_fp16.eval()
    
    fp16_ppls = []
    for text in test_texts:
        ppl = compute_perplexity(model_fp16, tokenizer, text)
        fp16_ppls.append(ppl)
        print(f"  fp16 PPL: {ppl:.2f}")
    fp16_avg = sum(fp16_ppls) / len(fp16_ppls)
    print(f"\nfp16 average PPL: {fp16_avg:.2f}")
    
    del model_fp16
    torch.cuda.empty_cache()

    # ---- W8A8 PPL ----
    print("\nLoading W8A8 and computing PPL (first call triggers autotune)...")
    model_w8a8 = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16, device_map="cuda",
    )
    convert_model_to_w8a8(model_w8a8, verbose=False)
    model_w8a8.eval()
    
    w8a8_ppls = []
    for i, text in enumerate(test_texts):
        t0 = time.time()
        ppl = compute_perplexity(model_w8a8, tokenizer, text)
        t = time.time() - t0
        w8a8_ppls.append(ppl)
        print(f"  w8a8 PPL: {ppl:.2f}" + (f" ({t:.1f}s, autotune)" if i == 0 else ""))
    w8a8_avg = sum(w8a8_ppls) / len(w8a8_ppls)
    print(f"\nw8a8 average PPL: {w8a8_avg:.2f}")

    # ---- 对比 ----
    print()
    print("=" * 75)
    print("Perplexity Summary")
    print("=" * 75)
    print(f"{'Text':<10}{'fp16 PPL':<15}{'w8a8 PPL':<15}{'Ratio (w8a8/fp16)':<20}")
    print("-" * 60)
    for i in range(len(test_texts)):
        ratio = w8a8_ppls[i] / fp16_ppls[i]
        print(f"Text {i+1:<4}{fp16_ppls[i]:<15.2f}{w8a8_ppls[i]:<15.2f}{ratio:<20.4f}")
    print(f"{'Avg':<10}{fp16_avg:<15.2f}{w8a8_avg:<15.2f}"
          f"{w8a8_avg/fp16_avg:<20.4f}")

    # 判定
    ratio_avg = w8a8_avg / fp16_avg
    print()
    if ratio_avg < 1.05:
        print(f"PASS: W8A8 PPL only {(ratio_avg-1)*100:.1f}% higher, 量化精度优秀")
    elif ratio_avg < 1.20:
        print(f"OK: W8A8 PPL {(ratio_avg-1)*100:.1f}% higher, 量化精度可接受")
    else:
        print(f"WARN: W8A8 PPL {(ratio_avg-1)*100:.1f}% higher, 量化精度退化严重")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--milestone", default="1",
                        choices=["1", "2", "3", "4", "all"])
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available")
        sys.exit(1)

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"HF cache: {os.environ.get('HF_HOME', '~/.cache/huggingface')}")
    print()

    if args.milestone == "1":
        milestone_1()
    elif args.milestone == "2":
        milestone_2()
    elif args.milestone == "3":
        milestone_3()
    elif args.milestone == "4":
        milestone_4()
    elif args.milestone == "all":
        milestone_1()
        print("\n\n")
        milestone_2()
        print("\n\n")
        milestone_3()
        print("\n\n")
        milestone_4()


if __name__ == "__main__":
    main()