# Adapted from https://github.com/QwenLM/ProcessBench/blob/main/code/run_eval.py

from datasets import load_dataset

# Embedded template to remove file dependency
CRITIQUE_TEMPLATE = """The following is a math problem and a solution (split into paragraphs, enclosed with tags and indexed from 0):

[Math Problem]

{problem}

[Solution]

{tagged_response}

Your task is to review and critique the solution paragraph by paragraph. Once you identify an error in a paragraph, return the index of the paragraph where the earliest error occurs. Otherwise, return the index of -1 (which typically denotes "not found").

Please put your final answer (i.e., the index) in \boxed{{}}."""


def apply_chat_template(toker, messages):
    """Applies the chat template and tokenizes the input."""
    return toker.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, add_special_tokens=False)


def prepare_input_boxed(template, input_d):
    """Prepares the input prompt for ProcessBench by formatting the problem and steps."""
    problem = input_d['problem']
    steps = input_d['steps']
    tagged_response = ''
    for sdx, step in enumerate(steps):
        tagged_response += f'''<paragraph_{sdx}>
{step}
</paragraph_{sdx}>

'''
    tagged_response = tagged_response.strip()
    prompt = template.format(problem=problem, tagged_response=tagged_response)
    messages = [{'role': 'user', 'content': prompt}]
    return messages


def process_data_processbench(split, tokenizer, template):
    """Loads ProcessBench dataset and prepares tokenized prompts."""
    print(f"INFO: Processing data for {"Qwen/ProcessBench"}/{split}...")
    input_data = load_dataset("Qwen/ProcessBench", split=split)
    prompt_token_ids = [apply_chat_template(tokenizer, prepare_input_boxed(template, e)) 
                        for e in input_data]
    return input_data, prompt_token_ids
