from langchain.agents import initialize_agent, AgentType
from backend.rag_pipeline import RAGPipeline
from backend.rag_tools import rag_query_tool
from backend.web_tools import tavily_search_tool


def invoke_agent(query: str) -> str:
    pipeline = RAGPipeline()
    rag_tool = rag_query_tool(pipeline)
    web_tool = tavily_search_tool()

    agent = initialize_agent(
        tools=[rag_tool, web_tool],
        llm=pipeline.llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        agent_kwargs={
            "system_message": "Always use RAGSearch for any questions about the uploaded PDF. Use WebSearch only if PDF does not contain the answer."
        },
        verbose=False,
        handle_parsing_errors=True,
    )

    result = agent.invoke(query)
    return result.get("output", "")
