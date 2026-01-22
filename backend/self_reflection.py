from typing import Tuple, Optional, cast
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam
from rag_pipeline import RAGPipeline
from langfuse_utils import langfuse_trace_span

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

def evaluate_answer(client: OpenAI, question: str, answer: str) -> Tuple[int, str, bool]:
    user_prompt = EVAL_PROMPT_TEMPLATE.format(question=question, answer=answer)

    messages = cast(
        list[ChatCompletionMessageParam],
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
    )

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=0
    )

    content = response.choices[0].message.content

    import json
    try:
        data = json.loads(content)
        score = int(data.get("score", 0))
        justification = data.get("justification", "")
        retry_needed = bool(data.get("retry_needed", False))
        return score, justification, retry_needed
    except Exception as e:
        return 1, f"Parsing failed: {e}", True

def self_reflect_and_retry(client: OpenAI, question: str, max_retries: int = 1) -> Tuple[str, Optional[str]]:
    rag = RAGPipeline()
    response = rag.ask(question)
    score, justification, retry = evaluate_answer(client, question, response)

    langfuse_trace_span(
        user_id="reflection",
        session_id="reflection",
        query=question,
        chain_type="reflection",
        use_compression=False,
        result={"result": response},
        sources=[],
        extra={
            "reflection_score": score,
            "reflection_retry": retry,
            "reflection_justification": justification
        }
    )

    if retry and max_retries > 0:
        retry_prompt = (
            f"Improve the following answer based on this feedback: '{justification}'.\n"
            f"Original Question: {question}\nOriginal Answer: {response}"
        )

        messages = cast(
            list[ChatCompletionMessageParam],
            [
                {"role": "user", "content": retry_prompt},
            ]
        )

        retry_response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=0.7
        )
        return retry_response.choices[0].message.content, justification

    return response, None
