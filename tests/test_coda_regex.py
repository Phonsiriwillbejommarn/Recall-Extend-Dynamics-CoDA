import re

def postprocess_predictions_outer(prediction):
    """Logic for outer loop (Planner) — supports task and answer actions."""
    pattern = r'<(task|answer)>(.*?)</\1>'
    match = re.search(pattern, prediction, re.DOTALL)
    if match:
        content = match.group(2).strip()
        action = match.group(1)
        return action, content
    return None, None

def postprocess_predictions_inner(prediction):
    """Logic for inner loop (Executor) — pure reasoning, only answer action."""
    pattern = r'<(answer)>(.*?)</\1>'
    match = re.search(pattern, prediction, re.DOTALL)
    if match:
        content = match.group(2).strip()
        action = match.group(1)
        return action, content
    return None, None

def test_coda_extraction():
    print("Testing CoDA Pure Reasoning Tag Extraction...")
    
    # === Outer Loop Tests (Planner) ===
    
    # Case 1: Simple Task delegation
    pred1 = "<task>Check this logic</task>"
    action, content = postprocess_predictions_outer(pred1)
    assert action == "task"
    assert content == "Check this logic"
    print("✅ Case 1 Passed: Simple <task>")

    # Case 2: Think + Task
    pred2 = "<think>Reasoning...</think><task>Verify step 1</task>"
    action, content = postprocess_predictions_outer(pred2)
    assert action == "task"
    assert content == "Verify step 1"
    print("✅ Case 2 Passed: <think> + <task>")

    # Case 3: Answer
    pred3 = "<answer>42</answer>"
    action, content = postprocess_predictions_outer(pred3)
    assert action == "answer"
    assert content == "42"
    print("✅ Case 3 Passed: <answer>")

    # Case 4: Invalid/No Tag
    pred4 = "<think>Just thinking</think>"
    action, content = postprocess_predictions_outer(pred4)
    assert action is None
    print("✅ Case 4 Passed: No Action Tag (None)")
    
    # === Inner Loop Tests (Executor — Pure Reasoning) ===
    
    # Case 5: Inner loop answer
    pred5 = "<think>Step by step reasoning...</think><answer>The answer is 42</answer>"
    action, content = postprocess_predictions_inner(pred5)
    assert action == "answer"
    assert content == "The answer is 42"
    print("✅ Case 5 Passed: Inner loop <think> + <answer>")
    
    # Case 6: Inner loop only think (no answer yet)
    pred6 = "<think>Still reasoning...</think>"
    action, content = postprocess_predictions_inner(pred6)
    assert action is None
    print("✅ Case 6 Passed: Inner loop no <answer> (None)")
    
    # Case 7: Multi-step reasoning with answer
    pred7 = "<think>[Actor - Initial Reasoning]\nStep 1: analyze\n[Critic - Evaluation]\nStep 1: correct\n[Actor - Refined]\nFinal: 30</think><answer>30</answer>"
    action, content = postprocess_predictions_inner(pred7)
    assert action == "answer"
    assert content == "30"
    print("✅ Case 7 Passed: Multi-step reasoning + <answer>")

    print("\n🎉 All tests passed! Regex logic supports Pure Reasoning CoDA.")

if __name__ == "__main__":
    test_coda_extraction()
