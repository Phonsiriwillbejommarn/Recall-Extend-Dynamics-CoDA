import os
import json
import requests
from tqdm import tqdm

# --- CONFIGURATION ---
TYPHOON_API_KEY = os.getenv("TYPHOON_API_KEY", "sk-vh0x2ClqhzKOvspZpT5CcmofWuhSbQwtRfedpZWrrji5jSPy")
TYPHOON_BASE_URL = "https://api.opentyphoon.ai/v1" 
MODEL_NAME = "typhoon-v2.5-30b-a3b-instruct" 

OUTPUT_FILE = "dataset/typhoon_data.jsonl"
TARGET_COUNT = 5000
BATCH_SIZE = 10
DELAY_SECONDS = 5

# --- PROMPTS ---
QUESTION_GENERATION_PROMPT = """You are a Professor of Computer Science and Mathematics.
Generate {n} diverse problems requiring multi-step reasoning or coding.
Please include a mix of:
1. Advanced Math (Calculus, Linear Algebra, Probability)
2. Applied Mathematics (Physics, Engineering, Business Optimization)
3. Daily Life Math (Pattern recognition, Logic puzzles, Financial planning, Ratios)
4. Programming & Algorithms (Python, C++, Data Structures, Debugging, System Design)
Format: Return ONLY a Python list of strings. Example:
[
  "Problem 1...",
  "Problem 2..."
]
"""

# ... (SYSTEM_PROMPT remains the same as before, simplified for this block) ...
SYSTEM_PROMPT = """You are an AI assistant capable of complex reasoning and coding.
Solve the problem using this EXACT format.

<think>
[Actor - Initial Reasoning/Planning]
Step 1: ...
Step 2: ...
Initial Code/Answer: ...

[Critic - Evaluation]
Step 1 Feedback: ... (Check for bugs, edge cases, of logic errors)
Overall Assessment: ...

[Actor - Refined Reasoning/Correction]
Step 1 (Revised): ... (Fix bugs or logic)
Final Answer/Code: ...
</think>
<answer>Put the final answer or code here</answer>
"""

import time
import ast

def get_current_count():
    if not os.path.exists(OUTPUT_FILE):
        return 0
    with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
        return sum(1 for line in f)

def generate_typhoon(prompt):
    headers = {
        "Authorization": f"Bearer {TYPHOON_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7,
        "max_tokens": 2048,
        "top_p": 0.9,
        "repetition_penalty": 1.05,
    }
    
    try:
        response = requests.post(f"{TYPHOON_BASE_URL}/chat/completions", headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
        return data['choices'][0]['message']['content']
    except Exception as e:
        print(f"Error calling Typhoon: {e}")
        time.sleep(10) # Backoff
        return None

def generate_questions(n=10):
    print(f"Generating {n} new questions...")
    headers = {
        "Authorization": f"Bearer {TYPHOON_API_KEY}",
        "Content-Type": "application/json"
    }
    
    prompt = QUESTION_GENERATION_PROMPT.format(n=n)
    
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.9,
        "max_tokens": 2048
    }
    
    try:
        response = requests.post(f"{TYPHOON_BASE_URL}/chat/completions", headers=headers, json=payload)
        response.raise_for_status()
        content = response.json()['choices'][0]['message']['content']
        
        try:
            start = content.find('[')
            end = content.rfind(']') + 1
            if start != -1 and end != -1:
                return ast.literal_eval(content[start:end])
        except:
             pass
        return [line.strip().strip('- "') for line in content.split('\n') if line.strip() and len(line) > 10][:n]
    except Exception as e:
        print(f"Error generating questions: {e}")
        return []

def main():
    current_count = get_current_count()
    print(f"Current dataset size: {current_count}/{TARGET_COUNT}")
    
    pbar = tqdm(total=TARGET_COUNT, initial=current_count)
    
    while current_count < TARGET_COUNT:
        questions = generate_questions(n=BATCH_SIZE)
        
        if not questions:
            print("Retrying generation...")
            time.sleep(5)
            continue
            
        for q in questions:
            response_text = generate_typhoon(q)
            time.sleep(DELAY_SECONDS) 
            
            if response_text:
                entry = {
                    "id": f"typhoon_gen_{current_count}",
                    "prompt": f"Question: {q}\nAnswer:",
                    "response": response_text
                }
                with open(OUTPUT_FILE, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                current_count += 1
                pbar.update(1)
            
            if current_count >= TARGET_COUNT:
                break
                
    pbar.close()
    print(f"✅ Reached target of {TARGET_COUNT} samples!")

if __name__ == "__main__":
    main()
