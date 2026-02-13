# CoDA Dataset Format Guide

To train the CoDA model (Actor-Critic Mode), data must be prepared in **JSONL** or **Parquet** format with the following fields:

## 1. Data Structure (JSON Object)

Each line (JSONL) or row (Parquet) must have the following keys:

```json
{
  "id": "unique_id_001",
  "prompt": "Question: <Question>\nAnswer:",
  "response": "<think>\n[Actor - Initial Reasoning]\n...\n[Critic - Evaluation]\n...\n[Actor - Refined Reasoning]\n...\n</think>\n<answer><Final Answer></answer>"
}
```

## 2. Response Format Details (Critical)

The `response` field must strictly follow this XML and text structure:

```xml
<think>
[Actor - Initial Reasoning]
Step 1: ... (Actor starts thinking)
Step 2: ...
Initial Answer: ...

[Critic - Evaluation]
Step 1 Feedback: ... (Critic iterates)
Overall Assessment: ...

[Actor - Refined Reasoning]
Step 1 (Revised): ... (Actor modifies based on critique)
Final Answer: ...
</think>
<answer>Final Answer</answer>
```

## 3. Usage Example

### Question: "What is 20 + 5 * 2?"

**Prompt:**
`Question: What is 20 + 5 * 2?\nAnswer:`

**Response:**
```xml
<think>
[Actor - Initial Reasoning]
Step 1: I will add 20 and 5 to get 25.
Step 2: Then multiply 25 by 2 to get 50.
Initial Answer: 50

[Critic - Evaluation]
Step 1 Feedback: Incorrect order of operations. Multiplication must be done before addition (PEMDAS).
Overall Assessment: Fail, recalculation needed.

[Actor - Refined Reasoning]
Step 1 (Revised): Calculate 5 * 2 first, which is 10.
Step 2 (Revised): Add 20 + 10 to get 30.
Final Answer: 30
</think>
<answer>30</answer>
```
