import csv
from langchain_openai import ChatOpenAI

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

def is_answer_based_on_sources(answer, sources, min_overlap=10):
    for s in sources:
        s = s[:200]
        if len(s) < min_overlap:
            continue
        for i in range(len(s) - min_overlap + 1):
            chunk = s[i:i+min_overlap]
            if chunk in answer:
                return True
    return False

def custom_self_eval(question, answer, source_documents, llm=None):
    context = "\n".join([doc.page_content for doc in source_documents])
    prompt = CUSTOM_PROMPT.format(reference=context, query=question, result=answer)
    llm = llm or ChatOpenAI(model="gpt-3.5-turbo")
    response = llm.invoke(prompt)
    faithful = None
    explanation = ""
    if "faithful:" in response.content.lower():
        for line in response.content.splitlines():
            if line.lower().startswith("faithful:"):
                ans = line.split(":", 1)[1].strip().lower()
                faithful = ans.startswith("y")
            elif line.lower().startswith("explanation:"):
                explanation = line.split(":", 1)[1].strip()
    return faithful, explanation or response.content

def save_eval_row(question, answer, sources, faithful, feedback, filename="eval_results.csv"):
    fieldnames = [
        "question", "answer", "num_sources",
        "answer_based_on_sources", "faithful", "llm_feedback"
    ]
    sources_text = [doc.page_content[:200].replace('\n', ' ') for doc in sources]
    answer_based = is_answer_based_on_sources(answer, sources_text)
    row = {
        "question": question,
        "answer": answer,
        "num_sources": len(sources),
        "answer_based_on_sources": answer_based,
        "faithful": faithful,
        "llm_feedback": feedback
    }
    with open(filename, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if f.tell() == 0:
            writer.writeheader()
        writer.writerow(row)

def save_llm_metrics(question, answer, docs, llm=None, filename="llm_eval_metrics.csv"):
    from json import loads
    llm = llm or ChatOpenAI(model="gpt-3.5-turbo")
    context = "\n\n".join([doc.page_content[:1000] for doc in docs])
    prompt = LLM_EVAL_PROMPT.format(question=question, answer=answer, context=context)
    response = llm.invoke(prompt)
    try:
        data = loads(response.content)
    except Exception as e:
        data = {"faithfulness": 1, "relevance": 1, "conciseness": 1, "justification": f"Parsing error: {e}"}

    row = {
        "question": question,
        "answer": answer,
        "faithfulness": data.get("faithfulness", 1),
        "relevance": data.get("relevance", 1),
        "conciseness": data.get("conciseness", 1),
        "justification": data.get("justification", "")
    }
    fieldnames = list(row.keys())
    with open(filename, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if f.tell() == 0:
            writer.writeheader()
        writer.writerow(row)
