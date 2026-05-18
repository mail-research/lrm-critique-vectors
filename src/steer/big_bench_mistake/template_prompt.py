"""
Prompt templates for BIG-Bench Mistake datasets.
Error detection in step-by-step reasoning.
"""

INTRO = "Given a problem and its solution steps below, identify whether there is any incorrect reasoning step. Do NOT solve the problem.\n\n"

ANSWER_FORMAT = "\n\nReturn \\boxed{{k}} where k is the step number of the incorrect step, or \\boxed{{-1}} if all steps are correct."

# === Multistep Arithmetic ===
MULTISTEP_ARITHMETIC_TEMPLATE = (
    INTRO +
    "Task: Solve the arithmetic expression.\n\n"
    "Problem: {input}\n\n"
    "Reasoning:\n{steps}"
    + ANSWER_FORMAT
)

# === Word Sorting ===
WORD_SORTING_TEMPLATE = (
    INTRO +
    "Task: Sort the following words alphabetically.\n\n"
    "Problem: {input}\n\n"
    "Reasoning:\n{steps}"
    + ANSWER_FORMAT
)

# === Dyck Languages ===
DYCK_LANGUAGES_TEMPLATE = (
    INTRO +
    "Task: Complete the rest of the sequence, making sure that the parentheses are closed properly.\n\n"
    "Problem: {input}\n\n"
    "Reasoning:\n{steps}"
    + ANSWER_FORMAT
)

# === Logical Deduction ===
LOGICAL_DEDUCTION_TEMPLATE = (
    INTRO +
    "Task: Logical deduction\n\n"
    "Problem:\n{input}\n\n"
    "Reasoning:\n{steps}"
    + ANSWER_FORMAT
)

# === Tracking Shuffled Objects ===
TRACKING_SHUFFLED_OBJECTS_TEMPLATE = (
    INTRO +
    "Task: Track shuffled objects\n\n"
    "Problem:\n{input}\n\n"
    "Reasoning:\n{steps}"
    + ANSWER_FORMAT
)

BIGBENCH_TEMPLATES = {
    "multistep_arithmetic": MULTISTEP_ARITHMETIC_TEMPLATE,
    "word_sorting": WORD_SORTING_TEMPLATE,
    "dyck_languages": DYCK_LANGUAGES_TEMPLATE,
    "logical_deduction": LOGICAL_DEDUCTION_TEMPLATE,
    "tracking_shuffled_objects": TRACKING_SHUFFLED_OBJECTS_TEMPLATE,
}

BIGBENCH_DATASETS = list(BIGBENCH_TEMPLATES.keys())
