import re
import requests
import time

def compute_qwen_score(responses_str, ground_truth, credentials=None, prompts_str=None):
    """
    Computes a reward score using Qwen-7B-Instruct as a judge via vLLM.
    
    Args:
        responses_str: The model's response text.
        ground_truth: The correct answer (string or dict).
        credentials: Optional dict containing 'server_url' (default: http://localhost:8000/v1)
        prompts_str: Optional prompt/question text.
    
    Returns:
        float: A score between 0.0 and 1.0 (or scaled).
    """
    
    server_url = "http://localhost:8000/v1/chat/completions"
    if credentials and 'server_url' in credentials:
        server_url = credentials['server_url']
        
    truth_text = ground_truth
    if isinstance(ground_truth, dict):
        truth_text = ground_truth.get('target', str(ground_truth))

    problem_text = prompts_str if prompts_str else "(Implicit in the solution)"

    # Construct the Judge Prompt (Detailed Rubric)
    judge_prompt = f"""You are a strict Logic and Math Auditor.
Your job is to evaluate the provided [Student Solution] against the [Problem] and [Ground Truth Answer].

[Problem]
{problem_text}

[Ground Truth Answer]
{truth_text}

[Student Solution]
{responses_str}

---
### **Evaluation Rubric (Total: 10 Points)**

**1. Final Answer Accuracy (0-5 Points):**
- **5 pts:** The final answer is EXACTLY correct and matches the Ground Truth.
- **0 pts:** The final answer is wrong. (No partial credit for the answer itself).

**2. Reasoning Process (0-4 Points):**
- **+4 pts:** Logic is flawless, step-by-step, and explicitly derives the answer.
- **+2 pts:** Logic is mostly correct but skips steps or is vague.
- **+1 pts:** Contains minor logical errors but arrives at the right answer (Lucky guess).
- **-2 pts:** **PENALTY:** Hallucinated facts or mathematical operations that are impossible.
- **+1 pts:** **BONUS:** The student performed a "Verification" or "Double Check" step.

**3. Format & Clarity (0-1 Points):**
- **+1 pts:** Used <think> and <answer> tags correctly and reasoning was readable.
- **0 pts:** Messy format or missing tags.

---
### **Auditor's Decision**
Provide a brief critique of the logic, then calculate the final score.
- If the Final Answer is WRONG, the Max Score is 3/10 (for good reasoning only).
- If the Final Answer is CORRECT but Logic is WRONG, the Max Score is 5/10 (Lucky guess penalty).

Final Format: [[Score: X]] (where X is 0-10)
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
