import requests
from config import LANGFUSE_HOST, AUTH

def langfuse_trace_span(user_id, session_id, query, chain_type, use_compression, result, sources, extra=None):
    trace_payload = {
        "name": "PDF QA Chain",
        "userId": user_id,
        "sessionId": session_id,
        "input": {"query": query},
        "metadata": {
            "chain_type": chain_type,
            "component": "web_ask",
            "compression": use_compression,
            **(extra or {})
        }
    }
    try:
        resp = requests.post(f"{LANGFUSE_HOST}/api/public/traces", auth=AUTH, json=trace_payload)
        trace_id = resp.json().get("id")
        span_payload = {
            "traceId": trace_id,
            "name": "RetrievalQA",
            "input": {"query": query},
            "output": {
                "result": result["result"],
                "sources": sources,
                "num_source_docs": len(sources),
            },
        }
        requests.post(f"{LANGFUSE_HOST}/api/public/spans", auth=AUTH, json=span_payload)
    except Exception as e:
        print(f"Langfuse trace error: {e}")
