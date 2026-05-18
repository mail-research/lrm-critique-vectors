# Adapted from https://github.com/IAAR-Shanghai/xVerify/tree/main/src/xVerify/prompts

BASE_TEMPLATE = {
    'xVerify-1B-I': '<|begin_of_text|><|start_header_id|>system<|end_header_id|>\\n\\nYou are a helpful AI assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>\\n\\n{query}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\\n\n',
    'xVerify-3B-Ia': '<|begin_of_text|><|start_header_id|>system<|end_header_id|>\\n\\nYou are a helpful AI assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>\\n\\n{query}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\\n\n',
    'xVerify-8B-I': '<|begin_of_text|><|start_header_id|>system<|end_header_id|>\\n\\nYou are a helpful AI assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>\\n\\n{query}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\\n\n',
    'xVerify-70B-I': '<|begin_of_text|><|start_header_id|>system<|end_header_id|>\\n\\nYou are a helpful assistant<|eot_id|><|start_header_id|>user<|end_header_id|>\\n\\n{query}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\\n\\n\n',
    'xVerify-0.5B-I': '<|im_start|>system\\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\\n<|im_start|>user\\n{query}<|im_end|>\\n<|im_start|>assistant\\n\n',
    'xVerify-1.5B-I': '<|im_start|>system\\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\\n<|im_start|>user\\n{query}<|im_end|>\\n<|im_start|>assistant\\n\n',
    'xVerify-3B-Ib': '<|im_start|>system\\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\\n<|im_start|>user\\n{query}<|im_end|>\\n<|im_start|>assistant\\n\n',
    'xVerify-7B-I': '<|im_start|>system\\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\\n<|im_start|>user\\n{query}<|im_end|>\\n<|im_start|>assistant\\n\n',
    'xVerify-14B-I': '<|im_start|>system\\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\\n<|im_start|>user\\n{query}<|im_end|>\\n<|im_start|>assistant\\n\n',
    'xVerify-32B-I': '<|im_start|>system\\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\\n<|im_start|>user\\n{query}<|im_end|>\\n<|im_start|>assistant\\n\n',
    'xVerify-72B-I': '<|im_start|>system\\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\\n<|im_start|>user\\n{query}<|im_end|>\\n<|im_start|>assistant\\n\n',
    'xVerify-9B-C': "<|system|>\\nYou are ChatGLM3, a large language model trained by Zhipu.AI. Follow the user's instructions carefully. Respond using markdown.\\n<|user|>\\n{query}\\n<|assistant|>\\n\n",
    'xVerify-2B-I': '<bos><start_of_turn>user\\n{query}<end_of_turn>\\n<start_of_turn>model\\n\n',
    'xVerify-9B-I': '<bos><start_of_turn>user\\n{query}<end_of_turn>\\n<start_of_turn>model\\n\n',
    'xVerify-27B-I': '<bos><start_of_turn>user\\n{query}<end_of_turn>\\n<start_of_turn>model\\n\n',
    'xVerify-14B-Ib': '<|im_start|>system<|im_sep|>\\nYou are a medieval knight and must provide explanations to modern people.<|im_end|>\\n<|im_start|>user<|im_sep|>\\n{query}<|im_end|>\\n<|im_start|>assistant<|im_sep|>\\n'
}

JUDGE_PROMPT = '''You are a diligent and precise assistant tasked with evaluating the correctness of responses. You will receive a question, an output sentence, and the correct answer. Your task is to determine if the output sentence accurately answers the question based on the provided correct answer. Respond with either [Correct] or [Incorrect].
-
Special considerations:

1. **Multiple Answers**: If the output contains multiple answers, evaluate whether later answers modify or correct earlier ones. In such cases, compare the final answer with the correct answer. If the final answer is unclear or incorrect, respond with [Incorrect].

2. **Mathematical Problems**: If the formats differ but the answers are mathematically equivalent, respond with [Correct].

3. **Explicit Options**: If the question provides explicit candidate answers, the output will be considered correct if it clearly indicates the correct option's code or the correct option's content.

4. **No Explicit Options**: If the question does not provide explicit options, the output must align with the correct answer in content and meaning to be considered [Correct].
-

Question: """{question}"""

Output sentence: """{output}"""

Correct answer: {answer}

Judgement:
'''