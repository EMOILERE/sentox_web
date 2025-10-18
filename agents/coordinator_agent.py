import asyncio
import time
from typing import Dict, List, Any
from datetime import datetime
from .base_agent import BaseAgent, AgentType, ActionType, AgentDecision
import dashscope
from dashscope import Generation
import logging

logger = logging.getLogger(__name__)

class CoordinatorAgent(BaseAgent):
    """协调智能体 - 负责综合各智能体的决策，达成最终共识"""
    
    def __init__(self, agent_id: str, config: Dict[str, Any]):
        super().__init__(agent_id, AgentType.COORDINATOR, config)
        self.api_key = config.get('dashscope_api_key')
        dashscope.api_key = self.api_key
        self.consensus_threshold = config.get('consensus_threshold', 0.7)
        
        # 协调决策模板
        self.coordination_prompt = """
你是一个多智能体协调系统的核心协调器。你需要综合分析各个智能体的决策结果，并达成最终的共识决策。

原始内容：{content}

各智能体的分析结果：
{agent_results}

协调任务：
1. 结果对比分析：
   - 比较各智能体的决策结果
   - 识别一致性和分歧点
   - 分析置信度差异

2. 证据权重评估：
   - 评估各智能体提供的证据质量
   - 考虑各智能体的专业领域优势
   - 权衡不同类型的分析结果

3. 风险综合评估：
   - 综合考虑所有识别的风险因素
   - 评估误判的潜在后果
   - 考虑平台政策和用户体验

4. 共识决策：
   - 基于多维度分析达成最终决策
   - 确保决策的合理性和可解释性
   - 提供明确的执行建议

请按以下JSON格式输出最终协调结果：
{{
    "final_decision": "approved" 或 "rejected" 或 "escalated",
    "confidence": 0.0-1.0之间的最终置信度,
    "consensus_level": 0.0-1.0之间的共识程度,
    "decision_rationale": "详细的决策依据",
    "agent_agreement_analysis": "各智能体的一致性分析",
    "risk_assessment": "综合风险评估",
    "evidence_summary": ["综合证据总结"],
    "action_recommendations": ["具体执行建议"],
    "escalation_reason": "如果需要升级的原因（仅在escalated时）"
}}
"""
    
    async def process(self, content: str, context: Dict[str, Any]) -> AgentDecision:
        """协调各智能体的决策并达成最终共识"""
        start_time = time.time()
        
        # 获取各智能体的决策结果
        agent_decisions = context.get('agent_decisions', [])
        
        # 添加协调开始的思考
        self.add_thought(
            thought="开始协调各智能体的决策结果",
            reasoning="需要综合分析各智能体的不同视角，达成最优决策",
            confidence=0.9,
            evidence=[f"收到{len(agent_decisions)}个智能体的决策结果"]
        )
        
        try:
            # 进行协调分析
            coordination_result = await self._coordinate_decisions(content, agent_decisions)
            processing_time = time.time() - start_time
            
            # 记录协调动作
            self.add_action(
                action_type=ActionType.COORDINATE,
                description="协调各智能体决策达成最终共识",
                input_data={
                    "content": content[:100] + "..." if len(content) > 100 else content,
                    "agent_count": len(agent_decisions)
                },
                output_data=coordination_result,
                success=True,
                execution_time=processing_time
            )
            
            # 添加协调完成后的思考
            self.add_thought(
                thought=f"协调完成，最终决策: {coordination_result['final_decision']}",
                reasoning=coordination_result['decision_rationale'],
                confidence=coordination_result['confidence'],
                evidence=coordination_result['evidence_summary']
            )
            
            # 构建最终决策结果
            decision = AgentDecision(
                agent_id=self.agent_id,
                decision=coordination_result['final_decision'],
                confidence=coordination_result['confidence'],
                reasoning=coordination_result['decision_rationale'],
                supporting_evidence=coordination_result['evidence_summary'],
                timestamp=datetime.utcnow()
            )
            
            return decision
            
        except Exception as e:
            error_time = time.time() - start_time
            logger.error(f"协调智能体 {self.agent_id} 处理失败: {str(e)}")
            
            # 记录失败动作
            self.add_action(
                action_type=ActionType.COORDINATE,
                description="协调决策失败",
                input_data={"content": content[:100] + "..." if len(content) > 100 else content},
                output_data={},
                success=False,
                execution_time=error_time,
                error_message=str(e)
            )
            
            # 返回升级决策
            return AgentDecision(
                agent_id=self.agent_id,
                decision="escalated",
                confidence=0.3,
                reasoning=f"协调过程中出现错误，建议人工介入: {str(e)}",
                supporting_evidence=["系统错误，需要人工审核"],
                timestamp=datetime.utcnow()
            )
    
    async def _coordinate_decisions(self, content: str, agent_decisions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """协调各智能体的决策"""
        try:
            # 格式化智能体结果用于提示词
            agent_results_text = self._format_agent_results(agent_decisions)
            
            response = Generation.call(
                model='qwen-plus',
                prompt=self.coordination_prompt.format(
                    content=content,
                    agent_results=agent_results_text
                ),
                max_tokens=1500,
                temperature=0.1
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
                        required_keys = ['final_decision', 'confidence', 'decision_rationale', 'evidence_summary']
                        if all(key in result for key in required_keys):
                            return result
                        else:
                            logger.warning(f"协调API响应缺少必要字段: {result}")
                    
                except json.JSONDecodeError as e:
                    logger.warning(f"无法解析协调API响应为JSON: {e}")
                
                # 如果JSON解析失败，使用备用协调逻辑
                return self._fallback_coordination(content, agent_decisions, result_text)
            else:
                logger.error(f"协调API调用失败: {response.status_code}")
                return self._fallback_coordination(content, agent_decisions, "API调用失败")
                
        except Exception as e:
            logger.error(f"调用协调API时出错: {str(e)}")
            return self._fallback_coordination(content, agent_decisions, f"API错误: {str(e)}")
    
    def _format_agent_results(self, agent_decisions: List[Dict[str, Any]]) -> str:
        """格式化智能体结果用于提示词"""
        results_text = ""
        for i, decision in enumerate(agent_decisions, 1):
            agent_id = decision.get('agent_id', f'Agent_{i}')
            agent_type = decision.get('agent_type', 'unknown')
            decision_text = decision.get('decision', 'unknown')
            confidence = decision.get('confidence', 0.0)
            reasoning = decision.get('reasoning', '无推理信息')
            evidence = decision.get('supporting_evidence', [])
            
            results_text += f"""
智能体 {i} ({agent_id} - {agent_type}):
- 决策: {decision_text}
- 置信度: {confidence:.2f}
- 推理: {reasoning}
- 证据: {'; '.join(evidence) if evidence else '无'}

"""
        return results_text
    
    def _fallback_coordination(self, content: str, agent_decisions: List[Dict[str, Any]], error_info: str) -> Dict[str, Any]:
        """备用协调逻辑（基于投票和规则）"""
        if not agent_decisions:
            return {
                "final_decision": "escalated",
                "confidence": 0.3,
                "consensus_level": 0.0,
                "decision_rationale": f"无智能体决策结果可供协调。{error_info}",
                "agent_agreement_analysis": "无数据",
                "risk_assessment": "无法评估",
                "evidence_summary": ["缺少智能体决策数据"],
                "action_recommendations": ["需要重新处理"],
                "escalation_reason": "缺少必要的决策数据"
            }
        
        # 统计各智能体的决策
        decisions = [d.get('decision', 'unknown') for d in agent_decisions]
        confidences = [d.get('confidence', 0.0) for d in agent_decisions]
        
        # 决策映射：safe->approved, toxic/risky->rejected, needs_review->escalated
        normalized_decisions = []
        for decision in decisions:
            if decision in ['safe']:
                normalized_decisions.append('approved')
            elif decision in ['toxic', 'risky']:
                normalized_decisions.append('rejected')
            else:
                normalized_decisions.append('escalated')
        
        # 计算共识程度
        decision_counts = {}
        for decision in normalized_decisions:
            decision_counts[decision] = decision_counts.get(decision, 0) + 1
        
        # 找出主要决策
        majority_decision = max(decision_counts.items(), key=lambda x: x[1])
        consensus_level = majority_decision[1] / len(normalized_decisions)
        
        # 决定最终决策
        if consensus_level >= self.consensus_threshold:
            final_decision = majority_decision[0]
            confidence = sum(confidences) / len(confidences) * consensus_level
        else:
            final_decision = "escalated"
            confidence = 0.5
        
        # 生成分析和建议
        agreement_analysis = f"共识程度: {consensus_level:.2f}, 决策分布: {decision_counts}"
        
        risk_assessment = "中等风险"
        if any(d in ['toxic', 'risky'] for d in decisions):
            risk_assessment = "存在风险因素"
        elif all(d == 'safe' for d in decisions):
            risk_assessment = "风险较低"
        
        evidence_summary = []
        for decision in agent_decisions:
            evidence = decision.get('supporting_evidence', [])
            evidence_summary.extend(evidence)
        
        # 去重并限制数量
        evidence_summary = list(set(evidence_summary))[:5]
        
        return {
            "final_decision": final_decision,
            "confidence": min(0.9, confidence),
            "consensus_level": consensus_level,
            "decision_rationale": f"基于{len(agent_decisions)}个智能体的投票结果，{majority_decision[0]}获得{majority_decision[1]}票。备用协调启用原因：{error_info}",
            "agent_agreement_analysis": agreement_analysis,
            "risk_assessment": risk_assessment,
            "evidence_summary": evidence_summary or ["无明确证据"],
            "action_recommendations": self._get_action_recommendations(final_decision, consensus_level),
            "escalation_reason": "智能体决策存在分歧" if final_decision == "escalated" else None
        }
    
    def _get_action_recommendations(self, decision: str, consensus_level: float) -> List[str]:
        """生成执行建议"""
        recommendations = []
        
        if decision == "approved":
            recommendations.append("内容可以发布")
            if consensus_level < 0.8:
                recommendations.append("建议加强后续监控")
        elif decision == "rejected":
            recommendations.append("拒绝发布此内容")
            recommendations.append("向用户提供修改建议")
            if consensus_level < 0.8:
                recommendations.append("考虑提供申诉渠道")
        else:  # escalated
            recommendations.append("提交人工审核")
            recommendations.append("标记为需要专家意见")
            recommendations.append("设置优先级处理")
        
        return recommendations
    
    async def collaborate(self, other_agents: List[BaseAgent], shared_context: Dict[str, Any]) -> Dict[str, Any]:
        """作为协调者，主要是收集和整合其他智能体的信息"""
        self.add_thought(
            thought="作为协调者收集智能体协作信息",
            reasoning="协调者需要全面了解各智能体的协作状态",
            confidence=0.9,
            evidence=["协调任务即将完成"]
        )
        
        final_decision = shared_context.get("my_decision")
        
        collaboration_data = {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type.value,
            "coordination_complete": True,
            "final_decision": final_decision.to_dict() if final_decision else {},
            "coordination_summary": self.get_action_chain_summary(),
            "system_status": "协调完成" if final_decision else "协调进行中"
        }
        
        return collaboration_data
