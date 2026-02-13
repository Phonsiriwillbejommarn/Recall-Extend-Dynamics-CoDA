import transformers
import torch
import re

question_list = [
    "Who was born first out of Cameron Mitchell (Singer) and Léopold De Saussure?", # Ground Truth: "Léopold De Saussure"
    "The Clavivox was invented by an American composer who was born Harry Warnow in what year?", # Ground Truth: "1908"
    "Which movie did Disney produce first, The Many Adventures of Winnie the Pooh or Ride a Wild Pony?", # Ground Truth: "Ride a Wild Pony"
    "Who is the sibling of the author of Kapalkundala?", # Ground Truth: "Sanjib Chandra" or "Sanjib Chandra Chattopadhyay"
]

# Model ID and device setup
model_id = ""
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

curr_eos = [151645, 151643] # for Qwen2.5 series models

# Initialize the tokenizer and model
tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
model = transformers.AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto")

# Define the custom stopping criterion
class StopOnSequence(transformers.StoppingCriteria):
    def __init__(self, target_sequences, tokenizer):
        # Encode the string so we have the exact token-IDs pattern
        self.target_ids = [tokenizer.encode(target_sequence, add_special_tokens=False) for target_sequence in target_sequences]
        self.target_lengths = [len(target_id) for target_id in self.target_ids]
        self._tokenizer = tokenizer

    def __call__(self, input_ids, scores, **kwargs):
        # Make sure the target IDs are on the same device
        targets = [torch.as_tensor(target_id, device=input_ids.device) for target_id in self.target_ids]

        if input_ids.shape[1] < min(self.target_lengths):
            return False

        # Compare the tail of input_ids with our target_ids
        for i, target in enumerate(targets):
            if torch.equal(input_ids[0, -self.target_lengths[i]:], target):
                return True

        return False


# Initialize the stopping criteria — stop when model outputs </answer>
target_sequences = ["</answer>", " </answer>", "</answer>\n", " </answer>\n"]
stopping_criteria = transformers.StoppingCriteriaList([StopOnSequence(target_sequences, tokenizer)])


def run_reasoning(question):
    """Run pure advanced reasoning (Chain-of-Thought) inference."""
    question = question.strip()
    
    # Prepare the prompt — pure reasoning, no search
    prompt = f"""You are a helpful assistant that excels at answering questions with step-by-step reasoning. \
To answer questions, you must reason carefully through the problem using <think> and </think>. \
Break down complex problems into smaller steps and verify each step. \
You may use multiple rounds of thinking if needed. \
Once you have sufficient confidence, provide a concise final answer using <answer> and </answer>. For example, <answer> Donald Trump </answer>. Question: {question}\n"""

    if tokenizer.chat_template:
        prompt = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=False)

    print(prompt)
    
    # Generate reasoning — single pass until </answer>
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    attention_mask = torch.ones_like(input_ids)
    
    outputs = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_new_tokens=2048,
        stopping_criteria=stopping_criteria,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        temperature=0.7
    )

    generated_tokens = outputs[0][input_ids.shape[1]:]
    output_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    print(output_text)
    
    # Extract final answer
    answer_pattern = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
    answer_match = answer_pattern.search(output_text)
    if answer_match:
        final_answer = answer_match.group(1).strip()
        print(f"Final answer found: {final_answer}")
    else:
        print("No final answer found in the output.")
        final_answer = "No final answer found."
    
    return output_text, final_answer

if __name__ == "__main__":
    output_text, final_answer = run_reasoning(question_list[0])
    print(f"Output: {output_text}")
    print(f"Final answer: {final_answer}")