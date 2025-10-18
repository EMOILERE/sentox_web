import asyncio
import time
from typing import Dict, List, Any
from datetime import datetime
from .base_agent import BaseAgent, AgentType, ActionType, AgentDecision
import dashscope
from dashscope import Generation
import logging

logger = logging.getLogger(__name__)

class ReasonerAgent(BaseAgent):
    """推理智能体 - 负责深度推理分析和上下文理解"""
    
    def __init__(self, agent_id: str, config: Dict[str, Any]):
        super().__init__(agent_id, AgentType.REASONER, config)
        self.api_key = config.get('dashscope_api_key')
        dashscope.api_key = self.api_key
        
        # 推理模板
        self.reasoning_prompt = """
你是一个专业的内容推理分析智能体。你的任务是进行深度推理分析，理解内容的潜在含义、上下文和可能的影响。

原始内容：{content}

已有的初步分类结果：{initial_classification}

请进行以下深度推理分析：

1. 语义分析：
   - 分析文本的深层语义和隐含意义
   - 识别反讽、暗示、双关等修辞手法
   - 理解文化背景和特定语境

2. 上下文推理：
   - 考虑可能的发布背景和目标受众
   - 分析内容可能产生的社会影响
   - 评估在不同平台环境下的适宜性

3. 意图识别：
   - 分析作者的真实意图和动机
   - 识别是否存在恶意诱导或误导
   - 判断是否有隐藏的有害目的

4. 风险评估：
   - 评估内容传播可能带来的风险
   - 分析对特定群体的潜在伤害
   - 考虑长期社会影响

5. 综合判断：
   - 综合多维度分析结果
   - 给出最终的推理结论
   - 提供改进建议（如适用）

请按以下JSON格式输出结果：
{{
    "reasoning_conclusion": "safe" 或 "risky" 或 "needs_review",
    "confidence": 0.0-1.0之间的置信度,
    "risk_factors": ["识别到的风险因素列表"],
    "context_analysis": "上下文分析结果",
    "intent_analysis": "意图分析结果",
    "social_impact": "社会影响评估",
    "detailed_reasoning": "详细的推理过程",
    "supporting_evidence": ["支持推理结论的证据"],
    "recommendations": ["改进建议或处理建议"]
}}
"""
    
    async def process(self, content: str, context: Dict[str, Any]) -> AgentDecision:
        """进行深度推理分析"""
        start_time = time.time()
        
        # 获取初步分类结果
        initial_classification = context.get('initial_classification', {})
        
        # 添加推理开始的思考
        self.add_thought(
            thought="开始深度推理分析",
            reasoning="需要在初步分类基础上进行更深层次的语义和上下文分析",
            confidence=0.9,
            evidence=[f"已获得初步分类: {initial_classification.get('classification', 'unknown')}"]
        )
        
        try:
            # 进行推理分析
            reasoning_result = await self._deep_reasoning(content, initial_classification)
            processing_time = time.time() - start_time
            
            # 记录推理动作
            self.add_action(
                action_type=ActionType.REASON,
                description="进行深度推理和上下文分析",
                input_data={
                    "content": content[:100] + "..." if len(content) > 100 else content,
                    "initial_classification": initial_classification
                },
                output_data=reasoning_result,
                success=True,
                execution_time=processing_time
            )
            
            # 添加推理完成后的思考
            self.add_thought(
                thought=f"推理分析完成，结论: {reasoning_result['reasoning_conclusion']}",
                reasoning=reasoning_result['detailed_reasoning'],
                confidence=reasoning_result['confidence'],
                evidence=reasoning_result['supporting_evidence']
            )
            
            # 构建决策结果
            decision = AgentDecision(
                agent_id=self.agent_id,
                decision=reasoning_result['reasoning_conclusion'],
                confidence=reasoning_result['confidence'],
                reasoning=reasoning_result['detailed_reasoning'],
                supporting_evidence=reasoning_result['supporting_evidence'],
                timestamp=datetime.utcnow()
            )
            
            return decision
            
        except Exception as e:
            error_time = time.time() - start_time
            logger.error(f"推理智能体 {self.agent_id} 处理失败: {str(e)}")
            
            # 记录失败动作
            self.add_action(
                action_type=ActionType.REASON,
                description="推理分析失败",
                input_data={"content": content[:100] + "..." if len(content) > 100 else content},
                output_data={},
                success=False,
                execution_time=error_time,
                error_message=str(e)
            )
            
            # 返回保守的决策
            return AgentDecision(
                agent_id=self.agent_id,
                decision="needs_review",
                confidence=0.3,
                reasoning=f"推理过程中出现错误，建议人工审核: {str(e)}",
                supporting_evidence=["系统错误，需要人工介入"],
                timestamp=datetime.utcnow()
            )
    
    async def _deep_reasoning(self, content: str, initial_classification: Dict[str, Any]) -> Dict[str, Any]:
        """进行深度推理分析"""
        try:
            # 构建推理提示词
            classification_info = {
                "classification": initial_classification.get("classification", "unknown"),
                "confidence": initial_classification.get("confidence", 0.0),
                "categories": initial_classification.get("toxicity_categories", []),
                "severity": initial_classification.get("severity_level", 1)
            }
            
            response = Generation.call(
                model='qwen-plus',
                prompt=self.reasoning_prompt.format(
                    content=content,
                    initial_classification=str(classification_info)
                ),
                max_tokens=1500,
                temperature=0.2
            )
            
            if response.status_code == 200:
                result_text = response.output.text
                
                # 尝试解析JSON响应
                try:
                    import json
                    start_idx = result_text.find('{')
                    end_idx = result_text.rfind('}') + 1
                    if start_idx != -1 and end_idx > start_idx:
                        json_str = result_text[start_idx:end_idx]
                        result = json.loads(json_str)
                        
                        # 验证结果格式
                        required_keys = ['reasoning_conclusion', 'confidence', 'detailed_reasoning', 'supporting_evidence']
                        if all(key in result for key in required_keys):
                            return result
                        else:
                            logger.warning(f"推理API响应缺少必要字段: {result}")
                    
                except json.JSONDecodeError as e:
                    logger.warning(f"无法解析推理API响应为JSON: {e}")
                
                # 如果JSON解析失败，进行基于规则的推理
                return self._fallback_reasoning(content, initial_classification, result_text)
            else:
                logger.error(f"推理API调用失败: {response.status_code}")
                return self._fallback_reasoning(content, initial_classification, "API调用失败")
                
        except Exception as e:
            logger.error(f"调用推理API时出错: {str(e)}")
            return self._fallback_reasoning(content, initial_classification, f"API错误: {str(e)}")
    
    def _fallback_reasoning(self, content: str, initial_classification: Dict[str, Any], error_info: str) -> Dict[str, Any]:
        """备用推理方法（基于规则的推理）"""
        # 分析内容长度和复杂度
        content_length = len(content)
        word_count = len(content.split())
        
        # 分析初步分类结果
        initial_decision = initial_classification.get("classification", "safe")
        initial_confidence = initial_classification.get("confidence", 0.5)
        initial_severity = initial_classification.get("severity_level", 1)
        
        # 基于规则的推理逻辑
        if initial_decision == "toxic":
            if initial_confidence > 0.8 and initial_severity >= 4:
                conclusion = "risky"
                confidence = 0.8
                reasoning = "初步分类显示高置信度的严重毒性内容，推理确认为高风险"
            elif initial_confidence > 0.6:
                conclusion = "needs_review"
                confidence = 0.6
                reasoning = "初步分类显示可能的毒性内容，建议人工审核确认"
            else:
                conclusion = "needs_review"
                confidence = 0.4
                reasoning = "初步分类不确定，需要进一步审核"
        else:
            # 对于初步分类为安全的内容，进行额外检查
            if content_length > 500 and any(char in content for char in '！？。'):
                conclusion = "safe"
                confidence = 0.7
                reasoning = "内容较长且结构完整，推理确认为安全内容"
            else:
                conclusion = "safe"
                confidence = 0.6
                reasoning = "初步分类为安全，推理分析未发现额外风险"
        
        # 简单的风险因素识别
        risk_factors = []
        if content_length < 10:
            risk_factors.append("内容过短，可能缺乏上下文")
        if "http" in content.lower():
            risk_factors.append("包含链接，需要注意钓鱼风险")
        if content.count('!') > 3 or content.count('？') > 3:
            risk_factors.append("过度使用标点符号，可能存在煽动性")
        
        return {
            "reasoning_conclusion": conclusion,
            "confidence": confidence,
            "risk_factors": risk_factors,
            "context_analysis": f"内容长度{content_length}字符，词数约{word_count}个",
            "intent_analysis": "基于规则分析，无法确定明确意图",
            "social_impact": "影响评估需要更多上下文信息",
            "detailed_reasoning": reasoning + f"。备用推理启用原因：{error_info}",
            "supporting_evidence": [
                f"初步分类: {initial_decision}",
                f"初步置信度: {initial_confidence}",
                f"内容特征: 长度{content_length}字符"
            ],
            "recommendations": ["建议结合人工审核" if conclusion == "needs_review" else "可按标准流程处理"]
        }
    
    async def collaborate(self, other_agents: List[BaseAgent], shared_context: Dict[str, Any]) -> Dict[str, Any]:
        """与其他智能体协作"""
        self.add_thought(
            thought="准备协作分享推理结果",
            reasoning="推理分析可以为协调智能体提供深度见解",
            confidence=0.8,
            evidence=["已完成深度推理", "准备分享分析结果"]
        )
        
        # 分析其他智能体的结果
        classifier_result = None
        for agent_data in shared_context.get("agent_results", []):
            if agent_data.get("agent_type") == "classifier":
                classifier_result = agent_data
                break
        
        # 比较分析结果
        my_decision = shared_context.get("my_decision")
        consistency_analysis = "无法比较"
        
        if classifier_result and my_decision:
            classifier_decision = classifier_result.get("classification_result", {}).get("decision")
            my_conclusion = my_decision.decision
            
            if classifier_decision == "safe" and my_conclusion == "safe":
                consistency_analysis = "与分类智能体结果一致，均判断为安全"
            elif classifier_decision == "toxic" and my_conclusion == "risky":
                consistency_analysis = "与分类智能体结果一致，均识别出风险"
            else:
                consistency_analysis = f"与分类智能体存在分歧：分类={classifier_decision}, 推理={my_conclusion}"
        
        collaboration_data = {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type.value,
            "reasoning_result": my_decision.to_dict() if my_decision else {},
            "consistency_analysis": consistency_analysis,
            "thought_chain_summary": self.get_thought_chain_summary(),
            "recommendation_for_coordination": self._get_coordination_recommendation(shared_context)
        }
        
        return collaboration_data
    
    def _get_coordination_recommendation(self, shared_context: Dict[str, Any]) -> str:
        """为协调智能体提供建议"""
        my_decision = shared_context.get("my_decision")
        if not my_decision:
            return "推理分析未完成，无法提供建议"
        
        confidence = my_decision.confidence
        decision = my_decision.decision
        
        if decision == "safe" and confidence > 0.8:
            return "推理分析确认内容安全，建议批准"
        elif decision == "risky" and confidence > 0.8:
            return "推理分析确认存在风险，建议拒绝"
        elif decision == "needs_review":
            return "推理分析发现不确定因素，强烈建议人工审核"
        else:
            return f"推理结果为{decision}，置信度{confidence:.2f}，建议综合其他智能体意见"
