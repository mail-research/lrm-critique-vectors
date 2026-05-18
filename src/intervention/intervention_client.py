import time
from pydantic import BaseModel, Field
from openai import OpenAI

#----------------------------------------------------------------------------------------
# === System prompts ===
INTERVENE_PROMPT = """
Given a question and its ground-truth answer, generate a natural reasoning trace that contains a calculation error.

PURPOSE:
- This trace is a deliberate error variant used to test a reasoning model's ability to recover from arithmetic mistakes.

INSTRUCTIONS:
- Write the trace as a sequence of short, natural reasoning steps, phrased as if someone is thinking aloud.
- Use natural, anthropomorphic openings such as "First, I need to...", "Okay, so...", "Let me...", or other common starter phrases typically found in the thinking process of reasoning models.
- Separate steps with two newlines (`\\n\\n`).
- Do NOT mention or hint that an error occurred (no phrases like "mistake", "should be", or "wrong").
- Introduce an arithmetic error at the earliest plausible calculation step.
- Stop the reasoning immediately after the mistaken calculation step — do not continue solving or produce an answer. The goal is to isolate the reasoning up to the error only.
- The trace must be self-consistent and match the question up to the step before the error.
- Keep the reasoning coherent, natural, and plausible.

EXAMPLE:
- Question: Olivia buys 3 packs of markers, each pack containing 12 markers. She gives 7 markers to her friend. How many markers does Olivia have left?
- Ground-truth answer: 29

Trace: "Okay, so I need to figure out how many markers Olivia got in total. She bought 3 packs, each with 12 markers.\\n\\nNext, I'll multiply 3 × 12 = 38."

Return only the interventions using the configured JSON schema.

Question:
{question}

Ground-truth answer:
{ground_truth}
"""
#----------------------------------------------------------------------------------------

class GeneratedTrace(BaseModel):
    trace: str = Field(..., description="Reasoning trace with a calculation error.")


class GPTIntervener:
    """OpenAI GPT client to generate reasoning traces."""

    def __init__(self, api_key: str, model_name: str):
        if not api_key: raise ValueError("An OpenAI API key is required.")
        self.api_key = api_key
        self.model_name = model_name
        self.client = OpenAI(api_key=self.api_key)
        
    def build_prompt(self, question: str, ground_truth: str) -> str:
        return INTERVENE_PROMPT.format(
            question=question,
            ground_truth=ground_truth,
        )

    def intervene_reasoning_chunks(self, question: str, ground_truth: str):
        """Generates an intervened trace with a calculation error."""
        
        prompt = self.build_prompt(question, ground_truth)

        for attempt in range(5):
            try:
                response = self.client.responses.parse(
                    model=self.model_name,
                    input=[
                        {"role": "system", "content": "You are a helpful assistant that crafts targeted reasoning traces with calculation errors."}, 
                        {"role": "user", "content": prompt}
                    ],
                    text_format=GeneratedTrace,
                )                                
                parsed: GeneratedTrace = response.output[-1].content[0].parsed

                return {"content": parsed.trace}
            
            except Exception as e:
                print(f"Attempt {attempt + 1}/5 failed: {e}")
                if attempt < 4: time.sleep(5 * (2 ** attempt))
        
        print("All retries exhausted. Returning None.")
        return None