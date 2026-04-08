#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
提示词构建工具
用于构建和测试各种提示词模板
"""

import json
from typing import Dict, List, Any


class PromptBuilder:
    """提示词构建器"""

    def __init__(self):
        self.system_prompt = ""
        self.user_template = ""
        self.assistant_template = ""

    def build_system_prompt(self) -> str:
        """构建系统提示词"""
        return """你是一个专业的认知状态分析师，专门分析社交媒体评论中的认知状态。

## 认知维度说明

### 1. 情感维度 (Emotion)
- **平静**: 情绪平稳，没有明显的情感波动
- **厌恶**: 对某事物感到反感、讨厌
- **信任**: 对某人或某事抱有信心和依赖
- **喜悦**: 感到高兴、快乐
- **惊讶**: 对意外事物感到震惊
- **愤怒**: 情绪激动，感到生气
- **期待**: 对未来事件抱有期望
- **其他**: 其他不明确或复杂的情感
- **悲伤**: 感到伤心、难过
- **恐惧**: 感到害怕、担心

### 2. 思维维度 (Thinking)
- **直觉型思维**（基于直觉和感觉）:
  - 主观评价: 过度自信效应的表现
  - 认同驱动的顺应: 社会认同理论的体现
  - 情绪化判断: 将情绪状态作为信息来源
  - 基于经验: 依赖脑海中直接浮现的例子
- **分析型思维**（基于逻辑和分析）:
  - 逻辑: 根据前提和规则进行系统化推理
  - 权衡: 分析比较各种选择的利益与效用
  - 循证: 有意识投入认知资源评估信息
  - 批判: 反思性、系统性地分析评估思维

### 3. 意图维度 (Intent)
- **表达主张**: 表达观点或明确立场
- **信息分享**: 提供事实或个人经历
- **分歧与冲突**: 表达对他人观点的不认同
- **情感表达**: 直接表露情感状态
- **寻求信息**: 通过提问获取信息或观点
- **认同与联结**: 表达对他人观点的积极态度
- **号召行动**: 指令或倡议促使他人行动
- **意图不明**: 无法明确分类
- **情绪化判断**: 基于情绪做出的判断

### 4. 立场维度 (Stance)
- **不明确**: 仅描述事件或引用数据，无明确倾向
- **支持美方**: 明确表达对美国关税政策的支持
- **支持中方**: 明确表达对中国关税政策的支持

## 输出格式
请严格按照以下格式输出，不要添加任何额外文字：
[EMO]{情感标签}
[THK]{思维标签}
[INT]{意图标签}
[STN]{立场标签}

## 注意事项
1. 思维标签需要根据具体的thinking_value判断是"直觉"还是"分析"
2. 每个标签必须从上述选项中选择
3. 不要添加解释或理由
4. 确保格式完全一致"""

    def build_user_template(self) -> str:
        """构建用户提示词模板"""
        return """[背景]{context}
[评论]{target}
[指令]请分析该评论的认知状态。"""

    def format_user_prompt(self, context: str, target: str) -> str:
        """格式化用户提示词"""
        template = self.build_user_template()
        return template.format(context=context, target=target)

    def build_assistant_template(self) -> str:
        """构建助手响应模板"""
        return "[EMO]{emotion}\n[THK]{thinking}\n[INT]{intent}\n[STN]{stance}"

    def format_assistant_response(self, emotion: str, thinking: str, intent: str, stance: str) -> str:
        """格式化助手响应"""
        template = self.build_assistant_template()
        return template.format(
            emotion=emotion,
            thinking=thinking,
            intent=intent,
            stance=stance
        )

    def analyze_thinking_type(self, thinking_value: str) -> str:
        """根据thinking_value判断思维类型"""
        intuitive_thinking = ['主观评价', '认同驱动的顺应', '情绪化判断', '基于经验']
        analytical_thinking = ['逻辑', '权衡', '循证', '批判']

        if thinking_value in intuitive_thinking:
            return '直觉'
        elif thinking_value in analytical_thinking:
            return '分析'
        else:
            return thinking_value  # 返回原始值或默认值

    def build_training_example(self, data_item: Dict[str, Any]) -> Dict[str, Any]:
        """构建训练样本"""
        # 提取认知标签
        labels = data_item.get('cognitive_labels', {})

        # 判断思维类型
        thinking_type = self.analyze_thinking_type(labels.get('thinking_value', ''))

        # 构建对话
        messages = [
            {
                "role": "system",
                "content": self.build_system_prompt()
            },
            {
                "role": "user",
                "content": self.format_user_prompt(
                    context=data_item.get('context_post', ''),
                    target=data_item.get('target_post', '')
                )
            },
            {
                "role": "assistant",
                "content": self.format_assistant_response(
                    emotion=labels.get('emotion', ''),
                    thinking=thinking_type,
                    intent=labels.get('intent', ''),
                    stance=labels.get('stance', '')
                )
            }
        ]

        return {
            "messages": messages,
            "source": "us_china_tariff_war",
            "conversation_id": data_item.get('conversation_id', -1),
            "user_id": data_item.get('user_id', -1),
            "original_labels": labels
        }


def test_prompt_builder():
    """测试提示词构建器"""
    builder = PromptBuilder()

    # 测试数据
    test_data = {
        "context_post": "印证了中国说的: 以斗争求和平则和平存，以妥协求和平则和平亡！",
        "target_post": "现在一个个都喊超预期，其实这都在川普计划之内，预期管理的高手",
        "cognitive_labels": {
            "emotion": "厌恶",
            "stance": "不明确",
            "thinking_type": "直觉",
            "thinking_value": "主观评价",
            "intent": "表达主张"
        }
    }

    # 构建训练样本
    example = builder.build_training_example(test_data)

    print("=" * 50)
    print("测试提示词构建器")
    print("=" * 50)
    print("\nSystem Prompt:")
    print(example['messages'][0]['content'])

    print("\nUser Prompt:")
    print(example['messages'][1]['content'])

    print("\nAssistant Response:")
    print(example['messages'][2]['content'])

    print("\n解析成功!")


if __name__ == "__main__":
    test_prompt_builder()