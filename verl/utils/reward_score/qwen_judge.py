import re
import requests
import time

def compute_qwen_score(responses_str, ground_truth, credentials=None):
    """
    Computes a reward score using Qwen-7B-Instruct as a judge via vLLM.
    
    Args:
        responses_str: The model's response text.
        ground_truth: The correct answer (string or dict).
        credentials: Optional dict containing 'server_url' (default: http://localhost:8000/v1)
    
    Returns:
        float: A score between 0.0 and 1.0 (or scaled).
    """
    
    server_url = "http://localhost:8000/v1/chat/completions"
    if credentials and 'server_url' in credentials:
        server_url = credentials['server_url']
        
    truth_text = ground_truth
    if isinstance(ground_truth, dict):
        truth_text = ground_truth.get('target', str(ground_truth))

    # Construct the Judge Prompt (Strict & Thorough)
    judge_prompt = f"""You are a strict Logic and Math Auditor.
Your task is to evaluate the student's solution with extreme thoroughness.
Check every single step for logical consistency, accuracy, and hallucination.

[Problem]
(Implicit in the solution)

[Ground Truth Answer]
{truth_text}

[Student Solution]
{responses_str}

[Auditing Rules]
1. **Accuracy is Paramount:** If the final answer is wrong -> Score < 3. No exceptions.
2. **Logic Must Be Sound:** If the answer is correct but the reasoning is flawed or hallucinates steps -> Score < 6. (We do not reward lucky guesses).
3. **Thoroughness:** Did the student verify their result? Did they skip important steps? Deduct points for laziness.
4. **Perfection (10/10):** Reserved for solutions that are undeniable, clear, and mathematically rigorous.

Provide a critique of the logic. BE STRICT.
Final score format: [[Score: X]]
"""

    payload = {
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "messages": [
            {"role": "system", "content": "You are a helpful and strict assistant."},
            {"role": "user", "content": judge_prompt}
        ],
        "temperature": 0.1,
        "max_tokens": 512
    }

    try:
        response = requests.post(server_url, json=payload, timeout=30)
        response.raise_for_status()
        result = response.json()
        content = result['choices'][0]['message']['content']
        
        # Extract Score
        match = re.search(r'\[\[Score:\s*(\d+(\.\d+)?)\]\]', content)
        if match:
            score = float(match.group(1))
            # Normalize to 0.0 - 1.0 range usually preferred for RL (or keep 0-10 if PPO configured for it)
            # User wants to avoid harsh penalty, so 0-1 range is standard. 
            # Let's return raw score for now, but usually rewards are small.
            # GRPO often works well with 0-1 indicators, but scalar is fine.
            # Let's normalize to 0.0 - 1.0
            return score / 10.0
        else:
            print(f"Judge Parse Error: Could not find [[Score: X]] in \n{content}")
            return 0.1 # Fallback for parse error
            
    except Exception as e:
        print(f"Judge Request Error: {e}")
        return 0.0
