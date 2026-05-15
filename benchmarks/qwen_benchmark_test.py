"""
qwen_benchmark_test.py - 真实任务 benchmark, 对比 W8A8 vs fp16 准确率.

[2026-05 修复说明]
    上一版有两个严重 bug, 导致 W8A8 准确率 (34%) 反而高于 fp16 (15%):
    
    Bug 1: extract_answer_number fallback "最后一个数字" 太弱.
        模型本来输出 "115. Let me check if I can find another way... 
        To solve 91 + 24", 模型答对了, 但 regex 提取最后一个数字得 24.
        
    Bug 2: Raw prompt + chat 模型组合不当.
        Qwen3-0.6B 是指令调过的 chat model. 直接给 raw prompt 
        "What is 23 + 45? The answer is" 会触发它"继续聊天"行为,
        生成长文本干扰 extract.
        
    讽刺结果: W8A8 输出更短(噪声促使早停), 反而让烂 regex 误打误撞.
    
[修复策略]
    A. Extract 防御: 优先取生成的第一个数字 (而不是最后一个).
    B. Prompt 根治: 用 chat template + "Reply with ONLY the number" 指令.
    C. Qwen3 关 thinking: enable_thinking=False 避免长推理.

测试任务:
    Task A: 小学算术 (100 题, 加减乘除 1-2 位数)
    Task B: GSM8K 子集 (30 题, 多步推理)

执行:
    python experiments/qwen_benchmark_test.py
    python experiments/qwen_benchmark_test.py --skip-gsm8k
"""

import argparse
import copy
import random
import re
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from integration.convert import convert_model_to_w8a8


# ============================================================
# Task A: 自造小学算术
# ============================================================

def generate_arithmetic_problems(n: int = 100, seed: int = 42) -> list:
    rng = random.Random(seed)
    problems = []
    for i in range(n):
        op_type = i % 4
        if op_type == 0:
            a, b = rng.randint(10, 99), rng.randint(10, 99)
            problems.append({"question": f"What is {a} + {b}?", "answer": a + b})
        elif op_type == 1:
            a = rng.randint(50, 99)
            b = rng.randint(1, a - 1)
            problems.append({"question": f"What is {a} - {b}?", "answer": a - b})
        elif op_type == 2:
            a, b = rng.randint(2, 12), rng.randint(2, 12)
            problems.append({"question": f"What is {a} * {b}?", "answer": a * b})
        else:
            b = rng.randint(2, 12)
            a = b * rng.randint(2, 12)
            problems.append({"question": f"What is {a} / {b}?", "answer": a // b})
    return problems


# ============================================================
# Task B: GSM8K 子集 (硬编码 30 题)
# ============================================================

GSM8K_SUBSET = [
    {"question": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?", "answer": 72},
    {"question": "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?", "answer": 10},
    {"question": "Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need to buy the wallet?", "answer": 5},
    {"question": "Julie is reading a 120-page book. Yesterday, she was able to read 12 pages and today, she read twice as many pages as yesterday. If she wants to read half of the remaining pages tomorrow, how many pages should she read?", "answer": 42},
    {"question": "James writes a 3-page letter to 2 different friends twice a week. How many pages does he write a year?", "answer": 624},
    {"question": "Albert is wondering how much pizza he can eat in one day. He buys 2 large pizzas and 2 small pizzas. A large pizza has 16 slices and a small pizza has 8 slices. If he eats it all, how many pieces does he eat that day?", "answer": 48},
    {"question": "Ken created a care package to send to his brother. Ken placed a box on a scale, and then he poured into the box enough jelly beans to bring the weight to 2 pounds. Then, he added enough brownies to cause the weight to triple. Next, he added another 2 pounds of jelly beans. And finally, he added enough gummy worms to double the weight once again. What was the final weight of the box of goodies, in pounds?", "answer": 16},
    {"question": "Tina makes $18.00 an hour. If she works more than 8 hours per shift, she is eligible for overtime, paid at 1.5x her hourly wage. If she works 10 hours every day for 5 days, how much money does she make?", "answer": 990},
    {"question": "A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?", "answer": 3},
    {"question": "Josh decides to try flipping a house. He buys a house for $80,000 and then puts in $50,000 in repairs. This increased the value of the house by 150%. How much profit did he make?", "answer": 70000},
    {"question": "James decides to run 3 sprints 3 times a week. He runs 60 meters each sprint. How many total meters does he run a week?", "answer": 540},
    {"question": "Every day, Wendi feeds each of her chickens three cups of mixed chicken feed. In the morning, she gives 15 cups of feed. In the afternoon, she gives another 25 cups. How many cups does she give in the final meal of the day if her flock has 20 chickens?", "answer": 20},
    {"question": "Toulouse has twice as many sheep as Charleston. Charleston has 4 times as many sheep as Seattle. How many sheep do Toulouse, Charleston, and Seattle have together if Seattle has 20 sheep?", "answer": 260},
    {"question": "Eliza's rate per hour for the first 40 hours she works each week is $10. She also receives overtime pay of 1.2 times her regular hourly rate. If Eliza worked for 45 hours this week, how much are her earnings?", "answer": 460},
    {"question": "A new program had 60 downloads in the first month. The second month had three times as many downloads as the first month. The third month was reduced by 30% from the second month. How many downloads total over three months?", "answer": 366},
    {"question": "Toula bought 3 dozen donuts which cost $68 per dozen, 2 dozen mini cupcakes which cost $80 per dozen, and 6 dozen mini cheesecakes for $55 per dozen. How much was the total cost?", "answer": 694},
    {"question": "Lao can sell each chicken for $1.50. A bag of chicken feed weighs 20 pounds and costs $2. Each chicken needs 2 pounds of feed. If he makes $65 profit from selling chickens, how many did he sell?", "answer": 50},
    {"question": "Janet pays $40/hour for 3 hours per week of clarinet lessons and $28/hour for 5 hours a week of piano lessons. How much more does she spend on piano lessons than clarinet lessons in a year?", "answer": 1040},
    {"question": "Mary received 18 new potted plants. She already has 2 potted plants on each of 40 window ledges. She gives 1 potted plant from each ledge to friends and family. How many potted plants will Mary have?", "answer": 58},
    {"question": "A bear needs to gain 1000 pounds. It gained a fifth of the weight from berries during summer, and during autumn it gained twice that amount from acorns. Salmon made up half of the remaining weight. How many pounds did it gain eating small animals?", "answer": 200},
    {"question": "Marcus has 100 water balloons. Each balloon holds 3 ounces of water. He can buy 50 ounces of water for $2.5 a bottle. If he walks into the store with $20, how much change will he have after he buys all the water he needs?", "answer": 5},
    {"question": "Sam bought a dozen boxes, each with 30 highlighter pens inside, for $10 each box. He rearranged five of these boxes into packages of six highlighters each and sold them for $3 per package. He sold the rest at the rate of three pens for $2. How much profit did he make in total, in dollars?", "answer": 115},
    {"question": "Carla is downloading a 200 GB file. She downloads at 2 GB/minute, but 40% through, Windows restarts (taking 20 min) and she must restart the download. How long total in minutes?", "answer": 160},
    {"question": "Lyle wants to buy himself and his friends a sandwich and a pack of juice. A sandwich costs $0.30, juice costs $0.20. If Lyle has $2.50, how many of his friends can have a sandwich and juice (besides Lyle)?", "answer": 4},
    {"question": "One glass costs $5, but every second glass costs only 60% of the price. Kylar wants to buy 16 glasses. How much does he need to pay?", "answer": 64},
    {"question": "Alexis went shopping with $200. She spent $30, $46, $38, $11, and $18 on various items. She also bought shoes but lost the receipt. She has $16 left. How much did Alexis pay for the shoes?", "answer": 41},
    {"question": "John drives 3 hours at 60 mph, then turns around. He spends 2 hours stuck in traffic, then 30 minutes at 30 mph, then the rest of the 4 hours at 80 mph. How far is he from home at the end?", "answer": 45},
    {"question": "Mark planted yellow, purple and green flowers. 10 are yellow. There are 80% more purple than yellow. Green is 25% as many as yellow and purple combined. How many flowers total?", "answer": 35},
    {"question": "In Idaho, 472 people were surveyed. The central angle of the 'Pop' sector of the pie chart was 251 degrees. How many people chose 'Pop'?", "answer": 329},
    {"question": "Susy has 100 followers. She gains 40 in week 1, 20 in week 2, 10 in week 3. Sarah has 50 followers, gains 90 in week 1, 30 in week 2, 10 in week 3. After three weeks, how many followers does the one with more total have?", "answer": 180},
]


# ============================================================
# 改进版: extract 答案 (取生成开头的数字, 不是最后)
# ============================================================

def extract_answer_number(text: str):
    """
    从生成文本里提取数字答案. 
    
    [新策略] 优先取"生成开头附近"的数字, 因为正确答案通常在最前面.
    这避免了上一版"取最后一个数字"被中间步骤数字干扰的 bug.
    
    顺序:
        1. 文本第一行/第一句的数字
        2. "answer is X" 或 "\\boxed{X}" 模式
        3. fallback: 整个文本第一个数字
    """
    # 取第一行 (或第一句, 用 . 截断)
    first_chunk = text.strip().split("\n")[0]
    first_chunk = first_chunk.split(".")[0] + "."   # 保留首句
    
    # 先在第一句里找数字
    numbers = re.findall(r"-?\d+", first_chunk)
    if numbers:
        try:
            return int(numbers[0])    # 第一句的第一个数字
        except ValueError:
            pass
    
    # fallback: 找 explicit 答案标记
    for pattern in [
        r"answer\s*(?:is|:)?\s*\$?(-?\d+)",
        r"\\boxed\{(-?\d+)\}",
        r"=\s*\$?(-?\d+)",
    ]:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            try:
                return int(matches[0])    # 第一个匹配
            except ValueError:
                pass
    
    # 最后 fallback: 整个文本第一个数字
    all_numbers = re.findall(r"-?\d+", text)
    if all_numbers:
        try:
            return int(all_numbers[0])
        except ValueError:
            pass
    
    return None


# ============================================================
# Generation: 用 chat template + enable_thinking=False
# ============================================================

@torch.no_grad()
def generate_with_chat(model, tokenizer, question: str, instruction: str,
                       max_new_tokens: int = 100) -> str:
    """
    用 chat template + 明确指令让模型简洁回答.
    
    instruction 例子:
        "Reply with ONLY the final number, nothing else."
        "Think step by step, then give the final number on the last line."
    """
    messages = [{"role": "user", "content": f"{question}\n{instruction}"}]
    
    # Qwen3 特殊: enable_thinking=False 关 thinking mode
    try:
        prompt_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,   # 关 thinking
        )
    except TypeError:
        # 旧版 tokenizer 不支持 enable_thinking, 退回普通模板
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
    
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    # 只取生成部分 (跳过 prompt)
    generated_ids = outputs[0][inputs.input_ids.shape[1]:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def warmup_model(model, tokenizer):
    inputs = tokenizer("Hello", return_tensors="pt").to(model.device)
    _ = model(**inputs)
    torch.cuda.synchronize()


# ============================================================
# 评估
# ============================================================

def evaluate_task(model, tokenizer, problems: list, label: str,
                  instruction: str, max_new_tokens: int,
                  progress_every: int = 20) -> dict:
    correct = 0
    wrong_examples = []
    
    t_start = time.time()
    for i, prob in enumerate(problems):
        response = generate_with_chat(
            model, tokenizer, prob["question"], instruction, max_new_tokens,
        )
        predicted = extract_answer_number(response)
        
        is_correct = (predicted == prob["answer"])
        if is_correct:
            correct += 1
        elif len(wrong_examples) < 5:
            wrong_examples.append({
                "q": prob["question"][:80],
                "expected": prob["answer"],
                "got": predicted,
                "raw": response[:120].replace("\n", " "),
            })
        
        if (i + 1) % progress_every == 0:
            print(f"    [{label}] {i+1}/{len(problems)}: {correct}/{i+1} correct")
    
    t_elapsed = time.time() - t_start
    return {
        "correct": correct,
        "total": len(problems),
        "accuracy": correct / len(problems),
        "time_s": t_elapsed,
        "wrong_examples": wrong_examples,
    }


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/qwen3-0.6b")
    parser.add_argument("--n-arith", type=int, default=100)
    parser.add_argument("--n-gsm8k", type=int, default=30)
    parser.add_argument("--skip-gsm8k", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available")
        sys.exit(1)
    print(f"GPU:   {torch.cuda.get_device_name(0)}")
    print(f"Model: {args.model}")
    print()

    # ----- Load models -----
    print("=" * 80)
    print("Loading fp16 + creating W8A8 copy")
    print("=" * 80)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model_fp16 = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float16, device_map="cuda",
    )
    model_fp16.eval()
    print(f"  fp16 loaded ({time.time()-t0:.1f}s)")
    
    t0 = time.time()
    model_w8a8 = copy.deepcopy(model_fp16)
    convert_model_to_w8a8(model_w8a8, verbose=False)
    print(f"  W8A8 created ({time.time()-t0:.1f}s)")
    
    print("\nWarmup W8A8 (trigger autotune)...")
    t0 = time.time()
    warmup_model(model_w8a8, tokenizer)
    print(f"  Warmup took {time.time()-t0:.1f}s")
    print()

    # ----- Task A: 小学算术 -----
    print("=" * 80)
    print(f"Task A: 小学算术 ({args.n_arith} 题)")
    print("=" * 80)
    arith_problems = generate_arithmetic_problems(args.n_arith)
    arith_instruction = "Reply with ONLY the final number, nothing else."
    
    print(f"\n  Evaluating fp16...")
    arith_fp16 = evaluate_task(
        model_fp16, tokenizer, arith_problems, "fp16",
        arith_instruction, max_new_tokens=20,
    )
    
    print(f"\n  Evaluating W8A8...")
    arith_w8a8 = evaluate_task(
        model_w8a8, tokenizer, arith_problems, "W8A8",
        arith_instruction, max_new_tokens=20,
    )

    # ----- Task B: GSM8K -----
    gsm8k_fp16 = gsm8k_w8a8 = None
    if not args.skip_gsm8k:
        print()
        print("=" * 80)
        n_gsm = min(args.n_gsm8k, len(GSM8K_SUBSET))
        print(f"Task B: GSM8K subset ({n_gsm} 题)")
        print("=" * 80)
        gsm_problems = GSM8K_SUBSET[:n_gsm]
        # GSM8K 需要推理, 让模型先想再给数字, 但仍然指令简洁
        gsm_instruction = ("Solve this step by step. End your response with "
                           "'The answer is X.' where X is the final number.")
        
        print(f"\n  Evaluating fp16...")
        gsm8k_fp16 = evaluate_task(
            model_fp16, tokenizer, gsm_problems, "fp16",
            gsm_instruction, max_new_tokens=300, progress_every=10,
        )
        
        print(f"\n  Evaluating W8A8...")
        gsm8k_w8a8 = evaluate_task(
            model_w8a8, tokenizer, gsm_problems, "W8A8",
            gsm_instruction, max_new_tokens=300, progress_every=10,
        )

    # ----- 总结 -----
    print()
    print("=" * 80)
    print("最终结果")
    print("=" * 80)
    print()
    print(f"{'Task':<25}{'fp16':<22}{'W8A8':<22}{'Diff':<12}")
    print("-" * 80)
    
    fp_pct = arith_fp16["accuracy"] * 100
    w8_pct = arith_w8a8["accuracy"] * 100
    diff = fp_pct - w8_pct
    fp_str = f"{fp_pct:.1f}% ({arith_fp16['correct']}/{arith_fp16['total']})"
    w8_str = f"{w8_pct:.1f}% ({arith_w8a8['correct']}/{arith_w8a8['total']})"
    diff_str = f"{diff:+.1f}pp"
    print(f"{'Arithmetic':<25}{fp_str:<22}{w8_str:<22}{diff_str:<12}")
    
    if gsm8k_fp16 is not None:
        fp_pct = gsm8k_fp16["accuracy"] * 100
        w8_pct = gsm8k_w8a8["accuracy"] * 100
        diff = fp_pct - w8_pct
        fp_str = f"{fp_pct:.1f}% ({gsm8k_fp16['correct']}/{gsm8k_fp16['total']})"
        w8_str = f"{w8_pct:.1f}% ({gsm8k_w8a8['correct']}/{gsm8k_w8a8['total']})"
        diff_str = f"{diff:+.1f}pp"
        print(f"{'GSM8K subset':<25}{fp_str:<22}{w8_str:<22}{diff_str:<12}")
    
    print()
    print("Speed:")
    print(f"  Arithmetic: fp16 {arith_fp16['time_s']:.1f}s, W8A8 {arith_w8a8['time_s']:.1f}s "
          f"(W8A8 {arith_w8a8['time_s']/arith_fp16['time_s']:.2f}x slower)")
    if gsm8k_fp16:
        print(f"  GSM8K:      fp16 {gsm8k_fp16['time_s']:.1f}s, W8A8 {gsm8k_w8a8['time_s']:.1f}s "
              f"(W8A8 {gsm8k_w8a8['time_s']/gsm8k_fp16['time_s']:.2f}x slower)")
    
    # 错例对比 (打印 fp16 和 W8A8 各自的错例)
    print()
    print("=" * 80)
    print("错例对比 (Arithmetic)")
    print("=" * 80)
    print("\nfp16 wrong:")
    for ex in arith_fp16["wrong_examples"][:3]:
        print(f"  Q: {ex['q']}")
        print(f"    Expected: {ex['expected']}, Got: {ex['got']}")
        print(f"    Raw: {ex['raw']}")
    print("\nW8A8 wrong:")
    for ex in arith_w8a8["wrong_examples"][:3]:
        print(f"  Q: {ex['q']}")
        print(f"    Expected: {ex['expected']}, Got: {ex['got']}")
        print(f"    Raw: {ex['raw']}")

    # Verdict
    print()
    print("=" * 80)
    arith_drop = arith_fp16["accuracy"] - arith_w8a8["accuracy"]
    if abs(arith_drop) < 0.05:
        print(f"VERDICT: W8A8 PASS — 准确率差异 {arith_drop*100:+.1f}pp (< 5pp)")
    elif arith_drop < 0.10:
        print(f"VERDICT: W8A8 OK — 准确率损失 {arith_drop*100:.1f}pp (5-10pp 可接受)")
    elif arith_drop > 0:
        print(f"VERDICT: W8A8 DEGRADED — 准确率损失 {arith_drop*100:.1f}pp (> 10pp)")
    else:
        print(f"VERDICT: 异常 — W8A8 (W8A8 比 fp16 高 {-arith_drop*100:.1f}pp), 可能仍有 benchmark bug")
    print("=" * 80)


if __name__ == "__main__":
    main()