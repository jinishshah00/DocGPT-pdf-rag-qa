import csv
import json
import os
from typing import List, Tuple
from langchain_openai import ChatOpenAI

EVAL_RESULTS_PATH = os.getenv("EVAL_RESULTS_PATH", "eval_results.csv")
LLM_METRICS_PATH = os.getenv("LLM_METRICS_PATH", "llm_eval_metrics.csv")

CUSTOM_PROMPT = """
Context (fragments of retrieved documents):
{reference}

User question:
{query}

Agent's answer:
{result}

Analyze whether the agent's answer correctly addresses the user question based on the given context.
If the answer is semantically aligned with the information in the context (even if not quoted verbatim), consider it faithful.

Reply in this format:
faithful: <Yes/No>
explanation: <short reasoning>
"""

LLM_EVAL_PROMPT = """
Question: {question}
Answer: {answer}
Context: {context}

Evaluate the answer with respect to the given context.
Respond in JSON with: faithfulness (1-5), relevance (1-5), conciseness (1-5), justification
"""


def _ensure_header(path: str, fieldnames: List[str]) -> None:
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()


def save_eval_row(question: str, answer: str, sources: List[dict], faithful: bool, feedback: str, filename: str = EVAL_RESULTS_PATH) -> None:
    fieldnames = [
        "question", "answer", "num_sources",
        "answer_based_on_sources", "faithful", "llm_feedback"
    ]
    sources_text = [s.get("snippet", "")[:200].replace("\n", " ") for s in (sources or [])]
    answer_based = any(chunk and chunk in answer for chunk in sources_text)
    row = {
        "question": question,
        "answer": answer,
        "num_sources": len(sources or []),
        "answer_based_on_sources": answer_based,
        "faithful": faithful,
        "llm_feedback": feedback,
    }
    _ensure_header(filename, fieldnames)
    with open(filename, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writerow(row)


def save_llm_metrics_row(question: str, answer: str, faithfulness: float, relevance: float, conciseness: float, justification: str, filename: str = LLM_METRICS_PATH) -> None:
    row = {
        "question": question,
        "answer": answer,
        "faithfulness": faithfulness,
        "relevance": relevance,
        "conciseness": conciseness,
        "justification": justification,
    }
    fieldnames = list(row.keys())
    _ensure_header(filename, fieldnames)
    with open(filename, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writerow(row)


def compute_custom_self_eval(question: str, answer: str, sources_text: List[str]) -> Tuple[bool, str]:
    context = "\n".join([s or "" for s in sources_text])
    prompt = CUSTOM_PROMPT.format(reference=context, query=question, result=answer)
    llm = ChatOpenAI(model="gpt-3.5-turbo")
    response = llm.invoke(prompt)
    faithful = False
    explanation = ""
    if "faithful:" in response.content.lower():
        for line in response.content.splitlines():
            if line.lower().startswith("faithful:"):
                ans = line.split(":", 1)[1].strip().lower()
                faithful = ans.startswith("y")
            elif line.lower().startswith("explanation:"):
                explanation = line.split(":", 1)[1].strip()
    return faithful, (explanation or response.content)


def compute_llm_metrics(question: str, answer: str, sources_text: List[str]) -> Tuple[float, float, float, str]:
    context = "\n\n".join([s or "" for s in sources_text])
    prompt = LLM_EVAL_PROMPT.format(question=question, answer=answer, context=context)
    llm = ChatOpenAI(model="gpt-3.5-turbo")
    response = llm.invoke(prompt)
    try:
        data = json.loads(response.content)
    except Exception as e:
        data = {
            "faithfulness": 1,
            "relevance": 1,
            "conciseness": 1,
            "justification": f"Parsing error: {e}",
        }
    return (
        float(data.get("faithfulness", 1)),
        float(data.get("relevance", 1)),
        float(data.get("conciseness", 1)),
        str(data.get("justification", "")),
    )
