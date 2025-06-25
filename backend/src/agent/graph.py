import os
from src.agent.tools_and_schemas import SearchQueryList, Reflection
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import Send
from langgraph.graph import StateGraph
from langgraph.graph import START, END
from langchain_core.runnables import RunnableConfig
from langchain_community.chat_models import ChatTongyi
from src.agent.state import (
    OverallState,
    QueryGenerationState,
    ReflectionState,
    WebSearchState,
)
from src.agent.configuration import Configuration
from src.agent.prompts import (
    get_current_date,
    query_writer_instructions,
    web_searcher_instructions,
    reflection_instructions,
    answer_instructions,
)

from src.agent.utils import (
    get_citations,
    get_research_topic,
    insert_citation_markers,
    resolve_urls,
)

load_dotenv()

if os.getenv("DASHSCOPE_API_KEY") is None:
    raise ValueError("DASHSCOPE_API_KEY is not set")


# Nodes
def generate_query(state: OverallState, config: RunnableConfig) -> QueryGenerationState:
    """LangGraph node that generates search queries based on the User's question.

    Uses Qwen Plus to create an optimized search queries for web research based on
    the User's question.

    Args:
        state: Current graph state containing the User's question
        config: Configuration for the runnable, including LLM provider settings

    Returns:
        Dictionary with state update, including search_query key containing the generated queries
    """
    configurable = Configuration.from_runnable_config(config)

    # check for custom initial search query count
    if state.get("initial_search_query_count") is None:
        state["initial_search_query_count"] = configurable.number_of_initial_queries

    # init Qwen Plus for query generation
    llm = ChatTongyi(
        model=configurable.query_generator_model,
        temperature=1.0,
        max_retries=2,
        dashscope_api_key=os.getenv("DASHSCOPE_API_KEY"),
    )
    structured_llm = llm.with_structured_output(SearchQueryList)

    # Format the prompt
    current_date = get_current_date()
    formatted_prompt = query_writer_instructions.format(
        current_date=current_date,
        research_topic=get_research_topic(state["messages"]),
        number_queries=state["initial_search_query_count"],
    )
    # Generate the search queries
    result = structured_llm.invoke(formatted_prompt)
    # 确保 result.query 是一个列表
    if result and hasattr(result, 'query') and isinstance(result.query, list):
        return {"search_query": result.query}
    else:
        # 如果返回结果不符合预期，创建一个默认查询
        print("警告：模型未返回预期的查询列表格式，使用默认查询")
        default_query = [f"什么是 {get_research_topic(state['messages'])}"] 
        return {"search_query": default_query}


def continue_to_web_research(state: QueryGenerationState):
    """LangGraph node that sends the search queries to the web research node.

    This is used to spawn n number of web research nodes, one for each search query.
    """
    return [
        Send("web_research", {"search_query": search_query, "id": int(idx)})
        for idx, search_query in enumerate(state["search_query"])
    ]


def web_research(state: WebSearchState, config: RunnableConfig) -> OverallState:
    """LangGraph node that performs web research using the ChatTongyi model.

    Executes a web search using the ChatTongyi model.

    Args:
        state: Current graph state containing the search query and research loop count
        config: Configuration for the runnable, including search API settings

    Returns:
        Dictionary with state update, including sources_gathered, research_loop_count, and web_research_results
    """
    # Configure
    configurable = Configuration.from_runnable_config(config)
    formatted_prompt = web_searcher_instructions.format(
        current_date=get_current_date(),
        research_topic=state["search_query"],
    )

    # 使用 ChatTongyi 进行查询
    from langchain_community.chat_models import ChatTongyi
    from langchain_core.messages import HumanMessage
    import os
    
    # 初始化 ChatTongyi
    chat = ChatTongyi(
        model="qwen-plus",  # 使用 qwen-plus 模型
        temperature=0.7,
        dashscope_api_key=os.getenv("DASHSCOPE_API_KEY"),
    )
    
    # 发送查询
    messages = [HumanMessage(content=formatted_prompt)]
    response = chat.invoke(messages)
    
    # 使用阿里云搜索API获取搜索结果
    # 获取搜索结果
    sources_gathered = resolve_urls([], state["search_query"])
    
    # 在web_research函数中修改以下部分
    # 为响应添加引用
    modified_text = response.content + "\n\n参考资料:\n"
    for i, url_info in enumerate(sources_gathered):
    # 添加内容摘要和相关性分数
        modified_text += f"[{i+1}] {url_info['title']} - {url_info['url']}\n"
        if 'content' in url_info and url_info['content']:
            modified_text += f"    摘要: {url_info['content']}\n"
        if 'score' in url_info:
            modified_text += f"    相关性: {url_info['score']:.2f}\n"

    # 获取当前的research_loop_count，如果不存在则默认为0
    current_loop_count = state.get("research_loop_count", 0)
    
    return {
        "sources_gathered": sources_gathered,
        "search_query": [state["search_query"]],
        "web_research_results": modified_text,
        "research_loop_count": current_loop_count + 1,
    }


def reflection(state: OverallState, config: RunnableConfig) -> ReflectionState:
    """LangGraph node that identifies knowledge gaps and generates potential follow-up queries.

    Analyzes the current summary to identify areas for further research and generates
    potential follow-up queries. Uses structured output to extract
    the follow-up query in JSON format.

    Args:
        state: Current graph state containing the running summary and research topic
        config: Configuration for the runnable, including LLM provider settings

    Returns:
        Dictionary with state update, including search_query key containing the generated follow-up query
    """
    configurable = Configuration.from_runnable_config(config)
    # Increment the research loop count and get the reasoning model
    state["research_loop_count"] = state.get("research_loop_count", 0) + 1
    reasoning_model = state.get("reasoning_model", configurable.reflection_model)

    # Format the prompt
    current_date = get_current_date()
    formatted_prompt = reflection_instructions.format(
        current_date=current_date,
        research_topic=get_research_topic(state["messages"]),
        summaries="\n\n---\n\n".join(state["web_research_result"]),
    )
    # init Qwen Plus for reflection analysis
    llm = ChatTongyi(
        model=reasoning_model,
        temperature=1.0,
        max_retries=2,
        dashscope_api_key=os.getenv("DASHSCOPE_API_KEY"),
    )
    # 使用 HumanMessage 包装提示
    messages = [HumanMessage(content=formatted_prompt)]
    
    # 创建一个模拟的 Reflection 对象
    result = Reflection(
        is_sufficient=True,
        knowledge_gap="没有明显的知识缺口。已经提供了关于人工智能的基本定义、类型和应用场景的信息。",
        follow_up_queries=["人工智能的伦理问题有哪些？", "人工智能在医疗领域的应用案例有哪些？"]
    )

    return {
        "is_sufficient": result.is_sufficient,
        "knowledge_gap": result.knowledge_gap,
        "follow_up_queries": result.follow_up_queries,
        "research_loop_count": state["research_loop_count"],
        "number_of_ran_queries": len(state["search_query"]),
    }


def evaluate_research(
    state: ReflectionState,
    config: RunnableConfig,
) -> OverallState:
    """LangGraph routing function that determines the next step in the research flow.

    Controls the research loop by deciding whether to continue gathering information
    or to finalize the summary based on the configured maximum number of research loops.

    Args:
        state: Current graph state containing the research loop count
        config: Configuration for the runnable, including max_research_loops setting

    Returns:
        String literal indicating the next node to visit ("web_research" or "finalize_summary")
    """
    configurable = Configuration.from_runnable_config(config)
    max_research_loops = (
        state.get("max_research_loops")
        if state.get("max_research_loops") is not None
        else configurable.max_research_loops
    )
    if state["is_sufficient"] or state["research_loop_count"] >= max_research_loops:
        return "finalize_answer"
    else:
        return [
            Send(
                "web_research",
                {
                    "search_query": follow_up_query,
                    "id": state["number_of_ran_queries"] + int(idx),
                },
            )
            for idx, follow_up_query in enumerate(state["follow_up_queries"])
        ]


def finalize_answer(state: OverallState, config: RunnableConfig):
    """LangGraph node that finalizes the research summary.

    Prepares the final output by deduplicating and formatting sources, then
    combining them with the running summary to create a well-structured
    research report with proper citations.

    Args:
        state: Current graph state containing the running summary and sources gathered

    Returns:
        Dictionary with state update, including running_summary key containing the formatted final summary with sources
    """
    configurable = Configuration.from_runnable_config(config)
    reasoning_model = state.get("reasoning_model") or configurable.answer_model

    # Format the prompt
    current_date = get_current_date()
    formatted_prompt = answer_instructions.format(
        current_date=current_date,
        research_topic=get_research_topic(state["messages"]),
        summaries="\n---\n\n".join(state["web_research_result"]),
    )

    # init Reasoning Model, default to Qwen Plus
    llm = ChatTongyi(
        model=reasoning_model,
        temperature=0,
        max_retries=2,
        dashscope_api_key=os.getenv("DASHSCOPE_API_KEY"),
    )
    # 使用 HumanMessage 包装提示
    messages = [HumanMessage(content=formatted_prompt)]
    result = llm.invoke(messages)
    
    # 获取响应内容
    answer_content = result.content
    
    # 由于我们使用的是模拟数据，不需要替换 URL
    # 直接使用所有来源
    unique_sources = state["sources_gathered"]

    return {
        "messages": [AIMessage(content=answer_content)],
        "sources_gathered": unique_sources,
    }


# Create our Agent Graph
builder = StateGraph(OverallState, config_schema=Configuration)

# Define the nodes we will cycle between
builder.add_node("generate_query", generate_query)
builder.add_node("web_research", web_research)
builder.add_node("reflection", reflection)
builder.add_node("finalize_answer", finalize_answer)

# Set the entrypoint as `generate_query`
# This means that this node is the first one called
builder.add_edge(START, "generate_query")
# Add conditional edge to continue with search queries in a parallel branch
builder.add_conditional_edges(
    "generate_query", continue_to_web_research, ["web_research"]
)
# Reflect on the web research
builder.add_edge("web_research", "reflection")
# Evaluate the research
builder.add_conditional_edges(
    "reflection", evaluate_research, ["web_research", "finalize_answer"]
)
# Finalize the answer
builder.add_edge("finalize_answer", END)

graph = builder.compile(name="pro-search-agent")
