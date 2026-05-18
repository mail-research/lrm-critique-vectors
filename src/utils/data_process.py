import string


# === Dataset Processing: extract the question and correct answer from different dataset formats ===
def _process_mmlu(doc):
    question_text = doc.get("question", "").strip()
    choices = doc.get("choices")
    question_text += "\n" + "\n".join(f"{string.ascii_uppercase[j]}. {choice}" for j, choice in enumerate(choices))
    answer = doc.get("answer")
    answer_idx = answer if isinstance(answer, int) else next((i for i, c in enumerate(choices) if str(c) == str(answer)), -1)
    correct_answer = f"{string.ascii_uppercase[answer_idx]}. {choices[answer_idx]}" if answer_idx != -1 else "N/A"
    return question_text, correct_answer

def _process_gpqa(doc):
    question_text = doc.get("Question").strip()
    choices = [doc.get(f"choice{i}") for i in range(1, 5)]
    question_text += "\n" + "\n".join(f"({string.ascii_uppercase[j]}) {choice}" for j, choice in enumerate(choices) if choice)
    correct_answer = doc.get("Correct Answer")
    return question_text, correct_answer

def _process_gsm8k(doc):
    question_text = doc.get("question").strip()
    correct_answer = doc.get("answer").split("####")[-1].strip()
    return question_text, correct_answer

def _process_arc(doc):
    question_text = doc.get("question").strip()
    choices = doc.get("choices", {})
    choice_texts = choices.get("text")
    choice_labels = choices.get("label")
    question_text += "\n" + "\n".join(f"({label}) {text}" for label, text in zip(choice_labels, choice_texts))
    correct_answer = doc.get("answerKey")
    return question_text, correct_answer

def _process_math_500(doc):
    return doc.get("problem").strip(), doc.get("answer")

def _process_aime_2024(doc):
    return doc.get("Problem").strip(), doc.get("Answer")

def _process_aime_2025(doc):
    return doc.get("problem").strip(), doc.get("answer")

def _process_intervened_dataset(doc):
    return str(doc.get("question")).strip(), str(doc.get("ground_truth")).strip()

def _process_steered_dataset(doc):
    inner = (doc.get("doc") or {})
    q = str(inner.get("question") or "").strip()
    a = str(inner.get("ground_truth") or inner.get("answer") or "").split("####", 1)[-1].strip()
    return q, a

# === Registry mapping dataset names to their process functions ===
DATASET_PROCESSORS = {
    "mmlu": _process_mmlu, "gpqa": _process_gpqa, "gsm8k": _process_gsm8k,
    "arc": _process_arc, "math_500": _process_math_500, "aime_2024": _process_aime_2024,
    "aime_2025": _process_aime_2025, "intervened_local": _process_intervened_dataset,
    "tts": _process_intervened_dataset, "steered": _process_steered_dataset,
}