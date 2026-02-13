import re

def postprocess_predictions(prediction):
    # Logic copied from search_r1/llm_agent/generation.py
    # Outer loop tracing
    pattern = r'<(task|answer)>(.*?)</\1>'
    match = re.search(pattern, prediction, re.DOTALL)
    if match:
        content = match.group(2).strip()
        action = match.group(1)
        return action, content
    return None, None

def test_coda_extraction():
    print("Testing CoDA Decoupled Tag Extraction...")
    
    # Case 1: Simple Task
    pred1 = "<task>Check this logic</task>"
    action, content = postprocess_predictions(pred1)
    assert action == "task"
    assert content == "Check this logic"
    print("✅ Case 1 Passed: Simple <task>")

    # Case 2: Think + Task
    pred2 = "<think>Reasoning...</think><task>Verify step 1</task>"
    action, content = postprocess_predictions(pred2)
    assert action == "task"
    assert content == "Verify step 1"
    print("✅ Case 2 Passed: <think> + <task>")

    # Case 3: Answer
    pred3 = "<answer>42</answer>"
    action, content = postprocess_predictions(pred3)
    assert action == "answer"
    assert content == "42"
    print("✅ Case 3 Passed: <answer>")

    # Case 4: Invalid/No Tag
    pred4 = "<think>Just thinking</think>"
    action, content = postprocess_predictions(pred4)
    assert action is None
    print("✅ Case 4 Passed: No Tag (None)")

    print("\n🎉 All tests passed! Regex logic supports Decoupled CoDA.")

if __name__ == "__main__":
    test_coda_extraction()
