#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试 dashscope 是否正确安装
"""

try:
    import dashscope
    print("✅ dashscope 安装成功")
except ImportError as e:
    print(f"❌ dashscope 安装失败: {e}")

try:
    from langchain_community.chat_models import ChatTongyi
    print("✅ ChatTongyi 导入成功")
except ImportError as e:
    print(f"❌ ChatTongyi 导入失败: {e}")