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

Whenever you need to verify your logic, pause and switch to Critic mode:
[Critic - Evaluation]
...strict verification of the reasoning above...

[Actor - Refined Reasoning]
...correction based on your own critique...
</think>

<answer>Final Answer</answer>"""

CRITIC_SYSTEM_PROMPT = """You are a **Strict Logic Critic**.
Your job is to verify the reasoning provided below.

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
    )
    
    response = tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True)
    return response

def run_coda_inference(user_query, model, tokenizer):
    print(f"\nUser Query: {user_query}\n" + "="*50)
    
    # --- 1. Actor Phase (Generate reasoning with self-critique) ---
    actor_messages = [
        {"role": "system", "content": ACTOR_SYSTEM_PROMPT},
        {"role": "user", "content": user_query}
    ]
    
    print("🤖 ACTOR: Thinking with self-critique...")
    actor_output = generate_response(model, tokenizer, actor_messages)
    print(f"--- Actor Output ---\n{actor_output}\n--------------------")
    
    # --- 2. Check if answer is present ---
    answer_pattern = re.compile(r'<answer>(.*?)</answer>', re.DOTALL)
    answer_match = answer_pattern.search(actor_output)
    
    if answer_match:
        final_answer = answer_match.group(1).strip()
        print(f"\n✅ Answer found: {final_answer}")
        return actor_output
    
    # --- 3. If no answer, run Critic for external verification ---
    print("\n🕵️ CRITIC: No answer found. Running external verification...")
    critic_messages = [
        {"role": "system", "content": CRITIC_SYSTEM_PROMPT},
        {"role": "user", "content": f"Please verify this reasoning:\n{actor_output}"}
    ]
    
    critic_output = generate_response(model, tokenizer, critic_messages)
    print(f"--- Critic Output ---\n{critic_output}\n---------------------")
    
    # --- 4. Resume Actor with feedback ---
    print("\n🔄 SYSTEM: Providing feedback back to Actor...")
    resume_content = f"{actor_output}\n\n[Critic Feedback]\n{critic_output}\n\n[Actor - Refined Reasoning]"
    actor_messages.append({"role": "assistant", "content": resume_content})
    
    print("🤖 ACTOR: Resuming with feedback...")
    final_output = generate_response(model, tokenizer, actor_messages)
    print(f"--- Final Actor Output ---\n{final_output}\n--------------------------")
    
    return final_output

# --- Main Application ---
if __name__ == "__main__":
    print("Loading Model... (This is a demo script)")
    # tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    # model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE)
    
    # Example usage (Commented out to prevent running without GPU)
    # run_coda_inference("Solve if x + 5 = 10, what is x?", model, tokenizer)
    
    print("✅ inference_demo.py created successfully. Uncomment the load lines to run with a real model.")
