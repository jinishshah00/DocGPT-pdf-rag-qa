from langchain.tools import Tool

def rag_query_tool(pipeline):
    def _rag_tool_func(input_text: str):
        try:
            result = pipeline.ask(input_text)
            return result.get("answer") or result.get("result") or ""
        except Exception as e:
            # Don't raise - return a helpful message so agents can continue
            return f"RAGSearch failed: {str(e)}"
    return Tool.from_function(
        name="RAGSearch",
        description="Use this tool ONLY to search for answers IN THE PDF knowledge base. If the question is about content in the uploaded document, use this tool.",
        func=_rag_tool_func
    )
