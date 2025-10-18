import asyncio
import time
from typing import Dict, List, Any
from datetime import datetime
from .base_agent import BaseAgent, AgentType, ActionType, AgentDecision
import dashscope
from dashscope import Generation
import logging

logger = logging.getLogger(__name__)

class ClassifierAgent(BaseAgent):
    """分类智能体 - 负责内容的初步分类和毒性检测"""
    
    def __init__(self, agent_id: str, config: Dict[str, Any]):
        super().__init__(agent_id, AgentType.CLASSIFIER, config)
        self.api_key = config.get('dashscope_api_key')
        dashscope.api_key = self.api_key
        
        # 分类相关配置
        self.toxicity_categories = [
            "仇恨言论", "暴力威胁", "骚扰辱骂", "色情内容", 
            "歧视言论", "虚假信息", "垃圾广告", "政治敏感"
        ]
        
        # 分类提示词模板
        self.classification_prompt = """
你是一个专业的中文内容审核分类智能体。请分析以下内容并进行毒性分类。

内容：{content}

请按照以下步骤进行分析：

1. 内容理解：
   - 识别内容的主要话题和情感倾向
   - 分析是否包含隐含或明显的有害信息
   - 考虑上下文和文化背景

2. 毒性检测：
   - 检查是否包含以下类型的有害内容：
     * 仇恨言论：针对特定群体的仇恨或偏见
     * 暴力威胁：威胁、恐吓或煽动暴力
     * 骚扰辱骂：人身攻击、辱骂或恶意骚扰
     * 色情内容：露骨的性内容或性暗示
     * 歧视言论：基于种族、性别、宗教等的歧视
     * 虚假信息：明显的谣言或误导性信息
     * 垃圾广告：不当的商业推广或诈骗
     * 政治敏感：可能引起政治争议的内容

3. 严重程度评估：
   - 1级：完全安全，无任何问题
   - 2级：轻微问题，但可以接受
   - 3级：中等问题，需要注意
   - 4级：严重问题，应当拒绝
   - 5级：极其严重，必须立即阻止

请按以下JSON格式输出结果：
{{
    "classification": "safe" 或 "toxic",
    "confidence": 0.0-1.0之间的置信度,
    "toxicity_categories": ["检测到的毒性类别列表"],
    "severity_level": 1-5的严重程度,
    "reasoning": "详细的分析推理过程",
    "evidence": ["支持分类决策的具体证据"]
}}
"""
    
    async def process(self, content: str, context: Dict[str, Any]) -> AgentDecision:
        """处理内容并进行分类"""
        start_time = time.time()
        
        # 添加初始思考
        self.add_thought(
            thought="开始对内容进行毒性分类分析",
            reasoning="作为分类智能体，需要首先识别内容中的潜在风险",
            confidence=0.9,
            evidence=[f"输入内容长度: {len(content)}字符"]
        )
        
        try:
            # 调用大模型进行分类
            classification_result = await self._classify_content(content)
            processing_time = time.time() - start_time
            
            # 记录分类动作
            self.add_action(
                action_type=ActionType.CLASSIFY,
                description="使用大模型对内容进行毒性分类",
                input_data={"content": content[:100] + "..." if len(content) > 100 else content},
                output_data=classification_result,
                success=True,
                execution_time=processing_time
            )
            
            # 添加分类后的思考
            self.add_thought(
                thought=f"完成内容分类，结果为: {classification_result['classification']}",
                reasoning=classification_result['reasoning'],
                confidence=classification_result['confidence'],
                evidence=classification_result['evidence']
            )
            
            # 构建决策结果
            decision = AgentDecision(
                agent_id=self.agent_id,
                decision=classification_result['classification'],
                confidence=classification_result['confidence'],
                reasoning=classification_result['reasoning'],
                supporting_evidence=classification_result['evidence'],
                timestamp=datetime.utcnow()
            )
            
            return decision
            
        except Exception as e:
            error_time = time.time() - start_time
            logger.error(f"分类智能体 {self.agent_id} 处理失败: {str(e)}")
            
            # 记录失败动作
            self.add_action(
                action_type=ActionType.CLASSIFY,
                description="内容分类失败",
                input_data={"content": content[:100] + "..." if len(content) > 100 else content},
                output_data={},
                success=False,
                execution_time=error_time,
                error_message=str(e)
            )
            
            # 返回默认安全决策
            return AgentDecision(
                agent_id=self.agent_id,
                decision="safe",
                confidence=0.3,
                reasoning=f"分类过程中出现错误，默认判断为安全: {str(e)}",
                supporting_evidence=["系统错误，采用保守策略"],
                timestamp=datetime.utcnow()
            )
    
    async def _classify_content(self, content: str) -> Dict[str, Any]:
        """调用大模型API进行内容分类"""
        try:
            response = Generation.call(
                model='qwen-plus',
                prompt=self.classification_prompt.format(content=content),
                max_tokens=1000,
                temperature=0.1
            )
            
            if response.status_code == 200:
                result_text = response.output.text
                
                # 尝试解析JSON响应
                try:
                    import json
                    # 提取JSON部分
                    start_idx = result_text.find('{')
                    end_idx = result_text.rfind('}') + 1
                    if start_idx != -1 and end_idx > start_idx:
                        json_str = result_text[start_idx:end_idx]
                        result = json.loads(json_str)
                        
                        # 验证结果格式
                        required_keys = ['classification', 'confidence', 'toxicity_categories', 'severity_level', 'reasoning', 'evidence']
                        if all(key in result for key in required_keys):
                            return result
                        else:
                            logger.warning(f"API响应缺少必要字段: {result}")
                    
                except json.JSONDecodeError as e:
                    logger.warning(f"无法解析API响应为JSON: {e}")
                
                # 如果JSON解析失败，进行简单的文本分析
                return self._fallback_classification(content, result_text)
            else:
                logger.error(f"API调用失败: {response.status_code}")
                return self._fallback_classification(content, "API调用失败")
                
        except Exception as e:
            logger.error(f"调用大模型API时出错: {str(e)}")
            return self._fallback_classification(content, f"API错误: {str(e)}")
    
    def _fallback_classification(self, content: str, error_info: str) -> Dict[str, Any]:
        """备用分类方法（基于关键词）"""
        # 简单的关键词检测
        toxic_keywords = [
            '死', '杀', '打死', '傻逼', '操你', '滚', '垃圾', '废物',
            '脑残', '白痴', '智障', '婊子', '贱', '蠢', '妈的'
        ]
        
        detected_toxic = []
        for keyword in toxic_keywords:
            if keyword in content:
                detected_toxic.append(keyword)
        
        is_toxic = len(detected_toxic) > 0
        confidence = min(0.8, 0.3 + len(detected_toxic) * 0.1)
        
        return {
            "classification": "toxic" if is_toxic else "safe",
            "confidence": confidence,
            "toxicity_categories": ["骚扰辱骂"] if is_toxic else [],
            "severity_level": min(5, 2 + len(detected_toxic)) if is_toxic else 1,
            "reasoning": f"基于关键词检测的备用分类。检测到敏感词：{detected_toxic}。原始错误：{error_info}",
            "evidence": detected_toxic if detected_toxic else ["未检测到明显的有害内容"]
        }
    
    async def collaborate(self, other_agents: List[BaseAgent], shared_context: Dict[str, Any]) -> Dict[str, Any]:
        """与其他智能体协作"""
        self.add_thought(
            thought="准备与其他智能体协作",
            reasoning="分类结果需要与推理智能体和协调智能体的分析结果进行比较",
            confidence=0.8,
            evidence=["已完成初步分类", "等待其他智能体的分析结果"]
        )
        
        # 提供自己的分类结果供其他智能体参考
        collaboration_data = {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type.value,
            "classification_result": shared_context.get("my_decision"),
            "thought_chain_summary": self.get_thought_chain_summary(),
            "confidence_level": shared_context.get("my_decision", {}).get("confidence", 0.0)
        }
        
        return collaboration_data
