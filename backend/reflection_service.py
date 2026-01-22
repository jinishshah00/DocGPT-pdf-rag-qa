import json
from openai import OpenAI

MODEL_NAME = "gpt-3.5-turbo"

SYSTEM_PROMPT = (
    "You are an assistant that evaluates the quality of AI responses. "
    "Given a question and an AI-generated answer, you will judge its quality on a scale of 1 to 5. "
    "Also provide a short justification and whether a retry is needed."
)

EVAL_PROMPT_TEMPLATE = (
    "Question: {question}\n"
    "Answer: {answer}\n"
    "\n"
    "Evaluate the answer. Respond in JSON format: \n"
    "{{\n  \"score\": 1-5, \"justification\": \"...\", \"retry_needed\": true/false \n}}"
)


def evaluate_answer(question: str, answer: str):
    client = OpenAI()
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": EVAL_PROMPT_TEMPLATE.format(question=question, answer=answer)},
        ],
        temperature=0,
    )
    content = response.choices[0].message.content
    try:
        data = json.loads(content)
        score = int(data.get("score", 0))
        justification = data.get("justification", "")
        retry_needed = bool(data.get("retry_needed", False))
        return score, justification, retry_needed
    except Exception as e:
        return 1, f"Parsing failed: {e}", True


def improve_answer(question: str, answer: str, justification: str):
    client = OpenAI()
    retry_prompt = (
        f"Improve the following answer based on this feedback: '{justification}'.\n"
        f"Original Question: {question}\nOriginal Answer: {answer}"
    )
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": retry_prompt}],
        temperature=0.7,
    )
    return response.choices[0].message.content
