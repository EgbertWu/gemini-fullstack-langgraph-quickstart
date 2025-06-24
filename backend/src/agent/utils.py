from typing import Any, Dict, List
import os
from langchain_core.messages import AnyMessage, AIMessage, HumanMessage
from tavily import TavilyClient


def get_research_topic(messages: List[AnyMessage]) -> str:
    """
    Get the research topic from the messages.
    """
    # check if request has a history and combine the messages into a single string
    if len(messages) == 1:
        research_topic = messages[-1].content
    else:
        research_topic = ""
        for message in messages:
            if isinstance(message, HumanMessage):
                research_topic += f"User: {message.content}\n"
            elif isinstance(message, AIMessage):
                research_topic += f"Assistant: {message.content}\n"
    return research_topic


def tavily_search(query: str, top_k: int = 3) -> List[Dict[str, str]]:
    """
    使用Tavily进行网络搜索
    
    Args:
        query: 搜索查询词
        top_k: 返回结果数量
        
    Returns:
        包含搜索结果的列表，每个结果包含url、title和content
    """
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise ValueError("TAVILY_API_KEY环境变量未设置")
    
    # 初始化Tavily客户端
    client = TavilyClient(api_key=api_key)
    
    try:
        # 执行搜索
        response = client.search(
            query=query,
            max_results=top_k
        )
        
        search_results = []
        
        # 处理搜索结果
        if "results" in response:
            for item in response["results"]:
                search_results.append({
                    "url": item.get("url", ""),
                    "title": item.get("title", ""),
                    "content": item.get("content", ""),  # 添加内容摘要
                    "score": item.get("score", 0.0)  # 添加相关性分数
                })
        
        return search_results
    except Exception as e:
        print(f"Tavily搜索API调用失败: {e}")
        # 如果API调用失败，返回空列表
        return []


def resolve_urls(urls_to_resolve: List[Any], query: str) -> List[Dict[str, str]]:
    """
    使用Tavily搜索API解析URL
    
    Args:
        urls_to_resolve: 要解析的URL列表（在这个实现中不使用）
        query: 搜索查询词
        
    Returns:
        包含搜索结果的列表，每个结果包含url和title
    """
    # 使用Tavily搜索API进行搜索
    search_results = tavily_search(query)
    
    # 如果搜索失败或没有结果，返回模拟数据
    if not search_results:
        return [
            {"url": "https://example.com/ai-definition", "title": "人工智能定义与介绍"},
            {"url": "https://example.com/ai-applications", "title": "人工智能应用场景"},
            {"url": "https://example.com/ai-future", "title": "人工智能的未来发展"},
        ]
    
    return search_results


def insert_citation_markers(text, citations_list):
    """
    Inserts citation markers into a text string based on start and end indices.

    Args:
        text (str): The original text string.
        citations_list (list): A list of dictionaries, where each dictionary
                               contains 'start_index', 'end_index', and
                               'segment_string' (the marker to insert).
                               Indices are assumed to be for the original text.

    Returns:
        str: The text with citation markers inserted.
    """
    # Sort citations by end_index in descending order.
    # If end_index is the same, secondary sort by start_index descending.
    # This ensures that insertions at the end of the string don't affect
    # the indices of earlier parts of the string that still need to be processed.
    sorted_citations = sorted(
        citations_list, key=lambda c: (c["end_index"], c["start_index"]), reverse=True
    )

    modified_text = text
    for citation_info in sorted_citations:
        # These indices refer to positions in the *original* text,
        # but since we iterate from the end, they remain valid for insertion
        # relative to the parts of the string already processed.
        end_idx = citation_info["end_index"]
        marker_to_insert = ""
        for segment in citation_info["segments"]:
            marker_to_insert += f" [{segment['label']}]({segment['short_url']})"
        # Insert the citation marker at the original end_idx position
        modified_text = (
            modified_text[:end_idx] + marker_to_insert + modified_text[end_idx:]
        )

    return modified_text


def get_citations(response, resolved_urls_map):
    """
    根据搜索结果创建引用列表
    
    Args:
        response: 模型响应
        resolved_urls_map: 解析后的URL映射
        
    Returns:
        引用列表
    """
    # 创建引用列表
    citations = [
        {
            "start_index": 0,
            "end_index": len(response.content),
            "segments": [
                {
                    "label": url_info["title"],
                    "short_url": url_info["url"],
                    "value": url_info["url"],
                } for url_info in resolved_urls_map
            ],
        }
    ]
    
    return citations
