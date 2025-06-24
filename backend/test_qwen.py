#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试 ChatTongyi 的基本功能
"""

import os
from dotenv import load_dotenv
from langchain_community.chat_models import ChatTongyi
from langchain_core.messages import HumanMessage

# 加载环境变量
load_dotenv()

# 检查 API 密钥
api_key = os.getenv("DASHSCOPE_API_KEY")
if not api_key:
    print("❌ 错误: DASHSCOPE_API_KEY 环境变量未设置")
    exit(1)
else:
    print(f"✅ DASHSCOPE_API_KEY 已设置: {api_key[:5]}...")

# 初始化 ChatTongyi
try:
    chat = ChatTongyi(
        model="qwen-plus",  # 使用 qwen-plus 模型
        temperature=0.7,
        dashscope_api_key=api_key,
    )
    print("✅ ChatTongyi 初始化成功")
    
    # 测试简单对话
    messages = [HumanMessage(content="你好，请简单介绍一下什么是人工智能？")]
    print("\n发送消息: '你好，请简单介绍一下什么是人工智能？'")
    print("\n等待回复...\n")
    
    response = chat.invoke(messages)
    print(f"回复: {response.content}")
    
    print("\n✅ 测试完成!")
except Exception as e:
    print(f"❌ 错误: {e}")