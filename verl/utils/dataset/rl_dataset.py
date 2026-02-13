# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from omegaconf import ListConfig
import os
from typing import List, Union

import pandas as pd

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, PreTrainedTokenizer
from verl.utils.fs import copy_local_path_from_hdfs

from verl.utils.model import compute_position_id_with_mask
import verl.utils.torch_functional as verl_F


def collate_fn(data_list: list[dict]) -> dict:
    tensors = {}
    non_tensors = {}

    for data in data_list:
        for key, val in data.items():
            if isinstance(val, torch.Tensor):
                if key not in tensors:
                    tensors[key] = []
                tensors[key].append(val)
            else:
                if key not in non_tensors:
                    non_tensors[key] = []
                non_tensors[key].append(val)

    for key, val in tensors.items():
        tensors[key] = torch.stack(val, dim=0)

    for key, val in non_tensors.items():
        non_tensors[key] = np.array(val, dtype=object)

    output = {}
    output.update(tensors)
    output.update(non_tensors)
    return output


class RLHFDataset(Dataset):
    """
    We assume the dataset contains a column that contains prompts and other information
    """

    def __init__(self,
                 parquet_files: Union[str, List[str]],
                 tokenizer: PreTrainedTokenizer,
                 prompt_key='prompt',
                 max_prompt_length=1024,
                 filter_prompts=True,
                 cache_dir='~/.cache/verl/rlhf',
                 chat_template_func=None,
                 return_raw_chat=False,
                 truncation='error',
                 model_type='base'):
        if not isinstance(parquet_files, (List, ListConfig)):
            parquet_files = [parquet_files]

        self.parquet_files = parquet_files
        self.cache_dir = os.path.expanduser(cache_dir)
        self.tokenizer = tokenizer

        self.prompt_key = prompt_key
        self.max_prompt_length = max_prompt_length
        self.filter_prompts = filter_prompts

        self.return_raw_chat = return_raw_chat
        self.chat_template_func = chat_template_func
        self.truncation = truncation
        self.model_type = model_type

        self._download()
        self._read_files_and_tokenize()

    def _download(self):
        from verl.utils.fs import copy_local_path_from_hdfs
        for i, parquet_file in enumerate(self.parquet_files):
            self.parquet_files[i] = copy_local_path_from_hdfs(src=parquet_file, cache_dir=self.cache_dir)

    def _read_files_and_tokenize(self):
        dataframes = []
        for parquet_file in self.parquet_files:
            # read parquet files and cache
            dataframe = pd.read_parquet(parquet_file)
            dataframes.append(dataframe)
        self.dataframe = pd.concat(dataframes)

        print(f'original dataset len: {len(self.dataframe)}')

        # filter out too long prompts
        tokenizer = self.tokenizer
        prompt_key = self.prompt_key

        # nvm if prompt is too long
        # self.dataframe = self.dataframe[self.dataframe.apply(lambda doc: len(
        #     tokenizer.apply_chat_template(doc[prompt_key], add_generation_prompt=True)) <= self.max_prompt_length,
        #                                                      axis=1)]

        print(f'filter dataset len: {len(self.dataframe)}')

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, item):
        """
        Note that we also return the raw_input_ids so that it can be combined with other chat template
        """
        row_dict = self.dataframe.iloc[item].to_dict()

        # Handle both plain string prompts and chat-format (list of dicts) prompts
        chat = row_dict.pop(self.prompt_key)
        if isinstance(chat, str):
            question = chat
        else:
            assert chat[0]["role"] == "user", "role must be user"
            question = chat[0]["content"]

        if self.model_type == "base":
            # Check if prompt already contains system prompt (from generate_sft_data.py)
            if isinstance(question, str) and "You are a helpful assistant" in question:
                 prompt_with_chat_template = question + "\n" + "Answer:"
            else:
                # Decoupled CoDA System Prompt (True CoDA) - with One-Shot Example
                system_prompt = """You are an AI assistant capable of complex reasoning.
To solve problems, you MUST use the following PROTOCOL:

<think>
[Actor - Reasoning]
...step-by-step logic...
</think>
<answer>Final Answer</answer>

Here is an example:
Question: Given a continuous function f(x) = x³ - 6x² + 9x + 1, find the exact coordinates of all local extrema and inflection points using derivatives, then verify your results numerically by plotting the function over [0, 5] with matplotlib.
<think>
[Actor - Reasoning]  
Step 1: To find local extrema, compute the first derivative f'(x) and solve f'(x) = 0 to locate critical points. Then use the second derivative test (f''(x)) to classify each critical point as a local maximum or minimum.  
Step 2: To find inflection points, compute the second derivative f''(x), solve f''(x) = 0, and check where concavity changes (i.e., sign change in f''(x)).  
Step 3: Use Python with NumPy and Matplotlib to define f(x), compute derivatives analytically, evaluate critical and inflection points, and plot f(x) over [0, 5] to visually verify findings.  

Initial Code/Answer:  
```python
import numpy as np
import matplotlib.pyplot as plt

# Define the function
def f(x):
    return x**3 - 6*x**2 + 9*x + 1

# First derivative
def f_prime(x):
    return 3*x**2 - 12*x + 9

# Second derivative
def f_double_prime(x):
    return 6*x - 12

# Find critical points (f'(x) = 0)
critical_points = np.roots([3, -12, 9])  # roots of 3x^2 - 12x + 9 = 0

# Evaluate second derivative at critical points to classify
local_extrema = []
for cp in critical_points:
    if np.isreal(cp):
        cp_val = np.real(cp)
        fpp_val = f_double_prime(cp_val)
        if fpp_val > 0:
            ext_type = "local minimum"
        elif fpp_val < 0:
            ext_type = "local maximum"
        else:
            ext_type = "inflection-like (undetermined)"
        local_extrema.append((cp_val, f(cp_val), ext_type))

# Inflection points: f''(x) = 0
inflection_x = np.roots([6, -12])[0]  # solution to 6x - 12 = 0
inflection_y = f(inflection_x)

# Plotting
x_vals = np.linspace(0, 5, 400)
y_vals = f(x_vals)

plt.figure(figsize=(10, 6))
plt.plot(x_vals, y_vals, label='f(x) = x³ - 6x² + 9x + 1', color='blue')
plt.scatter(*zip(*[(cp, f(cp)) for cp, _, _ in local_extrema]), color='red', zorder=5, label='Local Extrema')
plt.scatter(inflection_x, inflection_y, color='green', zorder=5, label='Inflection Point')
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.grid(True)
plt.legend()
plt.title('Graph of f(x) with Local Extrema and Inflection Point')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.xlim(0, 5)
plt.show()

# Output results
print("Local Extrema:")
for cp, fy, typ in local_extrema:
    print(f"x = {cp:.3f}, f(x) = {fy:.3f} ({typ})")

print(f"Inflection Point: x = {inflection_x:.3f}, f(x) = {inflection_y:.3f}")
```
</think>
<answer>
Local Extrema:
x = 1.000, f(x) = 5.000 (local maximum)
x = 3.000, f(x) = 1.000 (local minimum)
Inflection Point: x = 2.000, f(x) = 3.000
</answer>

Question: """
                prompt_with_chat_template = system_prompt + question + "\n" + "Answer:"

        else:
            # Default case - use chat template
            # Check if prompt already contains system prompt
            if isinstance(question, str) and "You are a helpful assistant" in question:
                 prompt_with_chat_template = question
            else:
                # CoDA System Prompt (Matching generate_typhoon.py)
                system_prompt = """You are an AI assistant capable of complex reasoning and coding.
Solve the problem using this EXACT format.

<think>
[Actor - Initial Reasoning/Planning]
Step 1: ...
Step 2: ...
Initial Code/Answer: ...

[Critic - Evaluation]
Step 1 Feedback: ...
Overall Assessment: ...

[Actor - Refined Reasoning/Correction]
Step 1 (Revised): ...
Final Answer/Code: ...
</think>
<answer>Put the final answer or code here</answer>
"""
                chat = [{"role": "system", "content": system_prompt}, {"role": "user", "content": question}]

                if self.tokenizer.chat_template:
                    prompt_with_chat_template = self.tokenizer.apply_chat_template(chat, add_generation_prompt=True, tokenize=False)
                else:
                    prompt_with_chat_template = chat[0]['content']

        input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(prompt=prompt_with_chat_template,
                                                                         tokenizer=self.tokenizer,
                                                                         max_length=self.max_prompt_length,
                                                                         pad_token_id=self.tokenizer.pad_token_id,
                                                                         left_pad=True,
                                                                         truncation=self.truncation)

        position_ids = compute_position_id_with_mask(attention_mask)

        row_dict['input_ids'] = input_ids[0]
        row_dict['attention_mask'] = attention_mask[0]
        row_dict['position_ids'] = position_ids[0]

        # encode prompts without chat template
        if self.return_raw_chat:
            row_dict['raw_prompt'] = chat.tolist()

        # add index for each prompt
        index = row_dict.get("extra_info", {}).get("index", 0)
        row_dict["index"] = index

        return row_dict
