#!/usr/bin/env python3
"""Generate SFT training data for CoDA (Pure Advanced Reasoning).
Creates expert-style trajectories with type-aware patterns:
- Math questions (GSM8K) → Direct CoT reasoning
- QA questions (HotpotQA) → Multi-step CoT reasoning with verification
"""
import pandas as pd
import json
import random

SYSTEM_PROMPT = """You are a helpful assistant that excels at answering questions with step-by-step reasoning.
To answer questions, you must reason carefully through the problem using <think> and </think>.
Break down complex problems into smaller steps and verify each step.
You may use multiple rounds of thinking if needed.
Once you have sufficient confidence, provide a concise final answer using <answer> and </answer>. For example, <answer> Donald Trump </answer>."""


# =============================================================================
# MATH PATTERNS (for GSM8K — direct CoT reasoning)
# =============================================================================

def make_math_direct_cot(question, target, search_query):
    """Math: Direct CoT reasoning → answer (70% of math)"""
    cot = random.choice([
        f'Let me solve this step by step.\n\nQuestion: "{question}"\n\nI need to work through the math carefully:\n- First, identify the key numbers and operations\n- Then calculate step by step\n- The answer should be {target}',
        f'This is a math problem. Let me reason through it:\n\nGiven: "{question}"\n\nStep 1: Identify what we need to find\nStep 2: Set up the calculation\nStep 3: Compute the result\n\nAfter working through the math, the answer is {target}.',
        f'Let me break down this math problem: "{question}"\n\nTo solve this:\n1. Parse the numbers and relationships\n2. Apply the correct operations\n3. Calculate carefully\n\nThe final answer is {target}.',
        f'I can solve this mathematically.\n\nProblem: "{question}"\n\nWorking through the calculations:\n- Analyzing the given information\n- Performing the arithmetic\n- Verifying the result\n\nThe answer is {target}.',
    ])
    
    response = f"""<think>
{cot}
</think>
<answer>{target}</answer>"""
    return response


def make_math_with_verification(question, target, search_query):
    """Math: Think → answer → verify (30% of math)"""
    cot_solve = random.choice([
        f'Let me work through this problem: "{question}"\n\nI need to:\n1. Understand what\'s being asked\n2. Identify the relevant numbers\n3. Perform the calculation step by step',
        f'This math problem requires careful calculation.\n\nQuestion: "{question}"\n\nLet me identify the approach and solve it systematically.',
    ])
    
    cot_verify = random.choice([
        f'Let me verify my answer of {target}.\n\nDouble-checking the calculation:\n- Re-reading the question to make sure I understood correctly\n- Verifying each step of my computation\n- The answer {target} is confirmed correct.',
        f'I should double-check: is {target} really the right answer?\n\nVerification:\n- Going back through each step\n- The math checks out\n- I\'m confident the answer is {target}.',
    ])
    
    response = f"""<think>
{cot_solve}
</think>
<think>
{cot_verify}
</think>
<answer>{target}</answer>"""
    return response


# =============================================================================
# QA PATTERNS (for HotpotQA — pure reasoning, no search)
# =============================================================================

def make_qa_direct_reasoning(question, target, search_query):
    """QA: Direct CoT reasoning → answer (50% of QA)"""
    cot = random.choice([
        f'Let me break down this question: "{question}"\nI need to identify the key entity and recall specific information about it. Let me think through what I know step by step.\n\nBased on my knowledge, the answer is {target}. Let me verify this makes sense in context.',
        f'To answer "{question}", I need to:\n1. Identify what specific information is being asked\n2. Recall relevant facts and knowledge\n3. Reason through to the answer\n\nAfter careful reasoning, I believe the answer is {target}.',
        f'This question asks about "{question}". Let me reason through this carefully by considering what I know about the key subject.\n\nStep 1: Identify the core question\nStep 2: Apply relevant knowledge\nStep 3: Verify the answer\n\nThe answer is {target}.',
        f'Analyzing the question: "{question}"\nThis requires factual knowledge. Let me think through what I know step by step.\n\nAfter careful consideration of the relevant facts, the answer is {target}.',
    ])
    
    response = f"""<think>
{cot}
</think>
<answer>{target}</answer>"""
    return response


def make_qa_multi_step_reasoning(question, target, search_query):
    """QA: Multi-step reasoning with intermediate steps (30% of QA)"""
    cot_step1 = random.choice([
        f'This question seems to require multiple pieces of information: "{question}"\nLet me start by breaking it down into sub-questions that I need to answer.',
        f'To answer "{question}", I may need to connect multiple facts.\nStep 1: Identify the key entities and relationships involved.\nLet me start with the most fundamental aspect.',
        f'Analyzing this question, it likely requires connecting multiple facts.\nI\'ll approach this systematically - first identify the key components, then reason through each one.',
    ])
    
    cot_step2 = random.choice([
        f'Now let me connect the pieces together.\nBased on my reasoning about each sub-question, I can now synthesize the information to arrive at the answer.\n\nConnecting the dots:\n- First piece of information leads to a key insight\n- Second piece confirms the direction\n- Together, they point to {target} as the answer.',
        f'After reasoning through the individual components, I can now put it all together.\n\nThe key insight is that the answer is {target}. This makes sense because the reasoning chain is consistent and each step logically follows from the previous one.',
        f'Excellent! By reasoning through each component separately, I can now combine my findings.\nThe answer {target} is supported by the logical chain of reasoning I\'ve constructed.',
    ])
    
    response = f"""<think>
{cot_step1}
</think>
<think>
{cot_step2}
</think>
<answer>{target}</answer>"""
    return response


def make_qa_with_verification(question, target, search_query):
    """QA: Reason + self-verify answer (20% of QA)"""
    cot_reason = random.choice([
        f'I need to find information about: "{question}"\nLet me reason through this carefully, considering what I know and whether my reasoning is sound.',
        f'Let me think about "{question}". I\'ll need to be careful to distinguish solid reasoning from assumptions.',
    ])
    
    cot_verify = random.choice([
        f'Let me verify my reasoning:\n- Does my answer make sense in context? Yes\n- Am I confident in my reasoning chain? Let me double-check\n- After verification, I\'m confident that {target} is the correct answer.\n\nMy reasoning is sound and the answer {target} is well-supported.',
        f'I need to evaluate the quality of my reasoning:\n- My initial reasoning leads to {target}\n- Let me check for any logical gaps or unsupported assumptions\n- After careful verification, {target} is indeed correct.',
    ])
    
    response = f"""<think>
{cot_reason}
</think>
<think>
{cot_verify}
</think>
<answer>{target}</answer>"""
    return response


# =============================================================================
# Main
# =============================================================================

MATH_PATTERNS = [
    (make_math_direct_cot,        0.70),  # 70% direct CoT
    (make_math_with_verification, 0.30),  # 30% CoT + verify
]

QA_PATTERNS = [
    (make_qa_direct_reasoning,     0.50),  # 50% direct reasoning
    (make_qa_multi_step_reasoning, 0.30),  # 30% multi-step
    (make_qa_with_verification,    0.20),  # 20% reason + verify
]


def detect_question_type(row):
    """Detect if question is math (GSM8K) or QA (HotpotQA)."""
    qid = str(row.get('id', row.get('extra_info', {}).get('id', '')))
    if 'gsm8k' in qid.lower():
        return 'math'
    # Check if answer looks numeric
    reward_model = row.get('reward_model', {})
    if isinstance(reward_model, str):
        try:
            reward_model = json.loads(reward_model)
        except:
            reward_model = {}
    if isinstance(reward_model, dict):
        gt = reward_model.get('ground_truth', {})
        if isinstance(gt, dict):
            t = gt.get('target', gt.get('answer', ''))
            if isinstance(t, list):
                t = t[0] if t else ''
            t = str(t).strip()
            # Numeric answers are likely math
            try:
                float(t.replace(',', '').replace('$', '').replace('%', ''))
                return 'math'
            except:
                pass
    return 'qa'


def select_pattern(patterns):
    """Weighted random selection from pattern list."""
    r = random.random()
    cumulative = 0
    for fn, weight in patterns:
        cumulative += weight
        if r <= cumulative:
            return fn
    return patterns[0][0]


def load_typhoon_data(path='dataset/typhoon_data.jsonl'):
    """Load and format Typhoon advanced reasoning data."""
    try:
        data = []
        with open(path, 'r') as f:
            for line in f:
                item = json.loads(line)
                prompt = item.get('prompt', '')
                response = item.get('response', '')
                
                # Extract parts for <think> and <answer>
                # Pattern: [Actor - Initial Reasoning/Planning] ... [Initial Code/Answer] ...
                think_content = ""
                answer_content = ""
                
                # Simple parsing logic based on observation
                if "[Actor - Initial Reasoning/Planning]" in response:
                    parts = response.split("[Initial Code/Answer:")
                    if len(parts) > 1:
                        think_part = parts[0].replace("[Actor - Initial Reasoning/Planning]", "").strip()
                        answer_part = parts[1].strip()
                        # Remove trailing Critic/Refined sections if present (keep first answer for simplicity or refine logic)
                        if "[Critic - Evaluation]" in answer_part:
                            answer_part = answer_part.split("[Critic - Evaluation]")[0].strip()
                        
                        think_content = think_part
                        answer_content = answer_part
                
                if not think_content or not answer_content:
                    # Fallback: treat whole response as answer if parsing fails
                    answer_content = response
                
                formatted_response = f"<think>\n{think_content}\n</think>\n<answer>\n{answer_content}\n</answer>"
                
                full_prompt = f"{SYSTEM_PROMPT}\n{prompt}\nAssistant: "
                data.append({
                    'prompt': full_prompt,
                    'response': formatted_response
                })
        print(f"✅ Loaded {len(data)} examples from {path}")
        return data
    except Exception as e:
        print(f"⚠️ Warning: Check if {path} exists. Error: {e}")
        return []

def create_sft_examples_from_train(train_path='data/train.parquet', output_path='data/sft_train.parquet', n_samples=1500):
    """Create SFT examples with mixed data: 50% Synthetic (Math/QA) + 50% Typhoon (Advanced)."""
    
    # 1. Load Typhoon Data (Advanced Reasoning)
    typhoon_data = load_typhoon_data()
    n_typhoon = len(typhoon_data)
    
    # 2. Load Synthetic Source Data
    if os.path.exists(train_path):
        df = pd.read_parquet(train_path)
        print(f"Loaded {len(df)} rows from {train_path}")
    else:
        print(f"⚠️ Warning: {train_path} not found. Skipping synthetic data generation.")
        df = pd.DataFrame()
    
    # Balance: Aim for 50/50 split if possible
    # If we have N typhoon samples, try to get N synthetic samples
    n_synthetic = n_typhoon if n_typhoon > 0 else n_samples
    
    # Cap total samples if needed, or sample with replace if not enough source data
    if len(df) > 0:
        replace = len(df) < n_synthetic
        df = df.sample(n=n_synthetic, replace=replace, random_state=42)
    
    synthetic_data = []
    type_counts = {'math': 0, 'qa': 0}
    pattern_counts = {}
    
    for idx, row in df.iterrows():
        # ... (Existing extraction logic) ...
        # Extract question
        prompt_data = row.get('prompt', '')
        if isinstance(prompt_data, list):
            question = prompt_data[0]['content'] if prompt_data else ''
        elif isinstance(prompt_data, str):
            question = prompt_data
        else:
            question = str(prompt_data)
        
        if not question:
            continue
        
        # Extract target answer
        reward_model = row.get('reward_model', {})
        if isinstance(reward_model, str):
            try:
                reward_model = json.loads(reward_model)
            except:
                reward_model = {}
        
        target = ''
        if isinstance(reward_model, dict):
            # Check for direct 'target' key
            if 'target' in reward_model:
                t = reward_model['target']
                if isinstance(t, list):
                    target = t[0] if t else ''
                else:
                    target = str(t)
            # Check for 'ground_truth' key
            elif 'ground_truth' in reward_model:
                gt = reward_model['ground_truth']
                if isinstance(gt, dict):
                    t = gt.get('target', gt.get('answer', ''))
                    if isinstance(t, list):
                        target = t[0] if t else ''
                    else:
                        target = str(t)
                elif isinstance(gt, str):
                    target = gt
                elif isinstance(gt, list):
                    target = gt[0] if gt else ''
        
        # If still no target, try golden_answers
        if not target:
            golden = row.get('golden_answers', [])
            if isinstance(golden, list) and golden:
                target = golden[0]
            elif isinstance(golden, str):
                target = golden
        
        if not target:
            continue
        
        # Detect type and select appropriate pattern
        q_type = detect_question_type(row)
        type_counts[q_type] += 1
        
        if q_type == 'math':
            selected_fn = select_pattern(MATH_PATTERNS)
        else:
            selected_fn = select_pattern(QA_PATTERNS)
        
        fn_name = selected_fn.__name__
        pattern_counts[fn_name] = pattern_counts.get(fn_name, 0) + 1
        
        search_query = question[:80]
        full_prompt = f"{SYSTEM_PROMPT}\nQuestion: {question}\nAssistant: "
        response = selected_fn(question, target, search_query)
        
        synthetic_data.append({
            'prompt': full_prompt,
            'response': response,
        })
    
    # Combine datasets
    all_data = typhoon_data + synthetic_data
    random.shuffle(all_data)
    
    sft_df = pd.DataFrame(all_data)
    sft_df.to_parquet(output_path, index=False)
    
    print(f"\n✅ Created {len(sft_df)} SFT examples → {output_path}")
    print(f"   - Typhoon Data: {len(typhoon_data)}")
    print(f"   - Synthetic Data: {len(synthetic_data)}")
    
    print(f"\nSynthetic Type distribution:")
    for t, count in type_counts.items():
        pct = count / len(synthetic_data) * 100 if len(synthetic_data) > 0 else 0
        print(f"  {t}: {count} ({pct:.0f}%)")

if __name__ == '__main__':
    random.seed(42)
    create_sft_examples_from_train()
