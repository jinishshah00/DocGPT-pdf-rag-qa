from langchain_community.tools.tavily_search.tool import TavilySearchResults

def tavily_search_tool():
    return TavilySearchResults(
        description="Use this tool ONLY for questions about recent events or general world knowledge. Do NOT use for questions about the PDF content."
    )
