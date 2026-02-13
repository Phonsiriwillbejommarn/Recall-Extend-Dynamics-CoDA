import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# --- Configuration ---
MODEL_NAME = "google/gemma-2b-it" # Or your trained checkpoint path
MAX_NEW_TOKENS = 512
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- System Prompts (The "Role Switching" Magic) ---
ACTOR_SYSTEM_PROMPT = """You are an AI assistant capable of complex reasoning.
To solve problems, you MUST use the following PROTOCOL:

<think>
[Actor - Initial Reasoning]
...step-by-step reasoning...

WHEN you need to verify your logic, CALL THE CRITIC:
<task>
Context: ...summary of reasoning...
Question: Is this correct?
</task>

The System will return the Critic's feedback in <result>...</result>.

[Actor - Refined Reasoning]
...correction based on feedback...
</think>

<answer>Final Answer</answer>"""

CRITIC_SYSTEM_PROMPT = """You are a **Strict Logic Critic**.
Your job is to verify the reasoning provided in the <task>.
CRITICAL: You do NOT see the full history. You only see the specific text the Actor wants you to check.

[Evaluation Rules]
1. Check for logical fallacies.
2. Verify mathematical derivations.
3. Check if the reasoning aligns with the problem statement (if provided).
4. Be skeptical. Do not assume the Actor is correct.

Output your evaluation in this format:
[Critic - Evaluation]
Step 1: [Correct/Incorrect] - Reason...
Overall: [Pass/Fail/Needs Refinement]"""

# --- Helper Functions ---

def generate_response(model, tokenizer, messages, stop_tokens=None):
    """Generic generation function."""
    input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True).to(DEVICE)
    
    outputs = model.generate(
        input_ids,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=True,
        temperature=0.7,
        pad_token_id=tokenizer.eos_token_id
        # In real implementation, add stopping criteria for </task> here
    )
    
    response = tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True)
    return response

def extract_task(text):
    """Finds the LAST <task>...</task> block."""
    pattern = r'<task>(.*?)</task>'
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        return matches[-1].strip() # Return the content of the last task
    return None

def run_coda_inference(user_query, model, tokenizer):
    print(f"\nUser Query: {user_query}\n" + "="*50)
    
    # --- 1. Actor Phase (Start) ---
    actor_messages = [
        {"role": "system", "content": ACTOR_SYSTEM_PROMPT},
        {"role": "user", "content": user_query}
    ]
    
    print("🤖 ACTOR: Thinking...")
    actor_output = generate_response(model, tokenizer, actor_messages)
    print(f"--- Actor Output ---\n{actor_output}\n--------------------")
    
    # --- 2. Controller Logic (The Interceptor) ---
    task_content = extract_task(actor_output)
    
    if task_content:
        print("\n🛑 SYSTEM: <task> detected! Pausing Actor. Summoning Critic...")
        print(f"📝 Task Content: '{task_content}'")
        
        # --- 3. Critic Phase (Fresh Context) ---
        # Notice: We do NOT append actor_messages. We start FRESH.
        critic_messages = [
            {"role": "system", "content": CRITIC_SYSTEM_PROMPT},
            {"role": "user", "content": f"Please verify this reasoning:\n{task_content}"}
        ]
        
        print("\n🕵️ CRITIC: Reviewing (in Fresh Context)...")
        critic_output = generate_response(model, tokenizer, critic_messages)
        print(f"--- Critic Output ---\n{critic_output}\n---------------------")
        
        # --- 4. Resume Actor (Inject Result) ---
        print("\n🔄 SYSTEM: Injecting <result> back to Actor...")
        
        # Construct the "Resume" message
        # In a real app, you'd cache the KV-cache. Here we just append to text history.
        resume_content = f"{actor_output}\n\n<result>\n{critic_output}\n</result>\n\n[Actor - Refined Reasoning]"
        
        actor_messages.append({"role": "assistant", "content": resume_content})
        
        print("🤖 ACTOR: Resuming with feedback...")
        final_output = generate_response(model, tokenizer, actor_messages) # This usually completes the answer
        print(f"--- Final Actor Output ---\n{final_output}\n--------------------------")
        
        return final_output
    else:
        print("\n✅ SYSTEM: No <task> detected. Actor is confident.")
        return actor_output

# --- Main Application ---
if __name__ == "__main__":
    print("Loading Model... (This is a demo script)")
    # tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    # model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE)
    
    # Example usage (Commented out to prevent running without GPU)
    # run_coda_inference("Solve if x + 5 = 10, what is x?", model, tokenizer)
    
    print("✅ inference_demo.py created successfully. Uncomment the load lines to run with a real model.")
