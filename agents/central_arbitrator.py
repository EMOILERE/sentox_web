import asyncio
import logging
import time
import json
from typing import Dict, List, Any, Optional
from datetime import datetime
from .communication_hub import AgentCommunicationHub, MessageType
import dashscope
from dashscope import Generation

logger = logging.getLogger(__name__)

class CentralArbitratorAgent:
    """SenTox-GLDA 升级版中心化仲裁智能体
    
    这是系统的最高决策层，负责：
    1. 仲裁所有子智能体的分析结果
    2. 进行最终的复杂推理和决策
    3. 与子智能体群进行双向沟通
    4. 提供最权威的审核结论
    """
    
    def __init__(self, communication_hub: AgentCommunicationHub, config: Dict[str, Any]):
        self.comm_hub = communication_hub
        self.config = config
        self.agent_id = "sentox_central_arbitrator"
        
        # 仲裁配置
        self.arbitration_config = {
            "consensus_threshold": config.get("consensus_threshold", 0.75),
            "confidence_weight": 0.3,
            "expertise_weight": 0.4,
            "consistency_weight": 0.3,
            "min_agent_participation": 3
        }
        
        # 智能体专业度权重
        self.agent_expertise_weights = {
            "toxicity_detector": 1.0,      # 毒性检测专家
            "sentiment_analyzer": 0.8,     # 情感分析专家
            "semantic_analyzer": 0.9,      # 语义分析专家
            "content_analyzer": 0.7,       # 内容分析专家
            "context_analyzer": 0.8,       # 上下文专家
            "risk_assessor": 0.95          # 风险评估专家
        }
        
        # 复杂推理模板
        self.arbitration_prompt = """
你是SenTox-GLDA升级版中心化仲裁智能体，是内容审核系统的最高决策层。你需要基于多个子智能体的分析结果进行最终仲裁。

原始内容：
{content}

各子智能体分析结果：
{sub_agent_analyses}

系统检测摘要：
{detection_summary}

你的仲裁任务：
1. 深度分析各子智能体的结论和证据
2. 识别智能体间的一致性和分歧点
3. 评估每个智能体分析的可靠性
4. 综合考虑所有维度进行最终裁决
5. 提供详细的仲裁理由和建议

仲裁标准：
- 安全第一原则：在不确定时倾向于保守判断
- 证据权重：优先考虑有具体证据支持的结论
- 专业性：重视专业智能体在其领域的判断
- 一致性：多个智能体的一致结论具有更高可信度
- 上下文：充分考虑平台特性和文化背景

请按照以下JSON格式输出最终仲裁结果：
{{
    "final_decision": "approved/rejected/needs_human_review",
    "confidence_score": 0.0-1.0的置信度,
    "arbitration_reasoning": "详细的仲裁推理过程",
    "evidence_analysis": {{
        "supporting_evidence": ["支持该决策的证据"],
        "conflicting_evidence": ["存在冲突的证据"],
        "evidence_reliability": 0.0-1.0的证据可靠性评分
    }},
    "agent_consensus_analysis": {{
        "high_agreement_agents": ["高度一致的智能体"],
        "conflicting_agents": ["存在分歧的智能体"],
        "consensus_score": 0.0-1.0的共识程度评分
    }},
    "risk_assessment": {{
        "primary_risks": ["主要风险因素"],
        "risk_level": "low/medium/high/critical",
        "potential_impact": "潜在影响评估"
    }},
    "recommendations": {{
        "immediate_action": "立即应采取的行动",
        "follow_up_actions": ["后续建议行动"],
        "monitoring_suggestions": ["监控建议"]
    }},
    "arbitrator_metadata": {{
        "processing_complexity": "low/medium/high",
        "decision_certainty": 0.0-1.0的决策确定性,
        "requires_escalation": true/false,
        "escalation_reason": "如需升级的具体原因"
    }}
}}
"""
        
        # 统计信息
        self.arbitration_stats = {
            "total_arbitrations": 0,
            "approved_count": 0,
            "rejected_count": 0,
            "escalated_count": 0,
            "average_confidence": 0.0,
            "consensus_distribution": {},
            "processing_times": []
        }
    
    async def initialize(self):
        """初始化中心仲裁智能体"""
        await self.comm_hub.register_agent(self.agent_id, {
            "type": "central_arbitrator",
            "role": "supreme_decision_maker",
            "capabilities": [
                "final_arbitration", "multi_agent_coordination", "complex_reasoning",
                "evidence_synthesis", "risk_adjudication", "consensus_building"
            ],
            "authority_level": "supreme",
            "version": "SenTox-GLDA-Enhanced-2.0"
        })
        
        logger.info("SenTox中心化仲裁智能体初始化完成")
    
    async def arbitrate_content(self, content: str, sub_agent_results: Dict[str, Any], 
                               context: Dict[str, Any] = None) -> Dict[str, Any]:
        """执行内容仲裁的主要方法"""
        start_time = time.time()
        session_id = context.get("session_id", f"arbitration_{int(time.time())}")
        
        logger.info(f"开始中心仲裁 [会话ID: {session_id}]")
        
        try:
            # 1. 预处理和验证输入
            validated_results = await self._validate_sub_agent_results(sub_agent_results)
            
            # 2. 与子智能体进行双向沟通确认
            communication_results = await self._communicate_with_sub_agents(
                content, validated_results, session_id
            )
            
            # 3. 分析智能体共识和分歧
            consensus_analysis = await self._analyze_agent_consensus(validated_results)
            
            # 4. 执行复杂推理仲裁
            arbitration_result = await self._perform_complex_arbitration(
                content, validated_results, consensus_analysis, context
            )
            
            # 5. 验证和优化决策
            final_decision = await self._validate_and_optimize_decision(
                arbitration_result, validated_results, consensus_analysis
            )
            
            # 6. 生成详细报告
            comprehensive_report = await self._generate_comprehensive_report(
                content, validated_results, arbitration_result, final_decision, 
                communication_results, session_id
            )
            
            processing_time = time.time() - start_time
            
            # 更新统计信息
            self._update_arbitration_stats(final_decision, processing_time)
            
            logger.info(f"中心仲裁完成 [会话ID: {session_id}] [决策: {final_decision.get('final_decision')}] [耗时: {processing_time:.2f}s]")
            
            return {
                "session_id": session_id,
                "arbitration_result": final_decision,
                "comprehensive_report": comprehensive_report,
                "communication_log": communication_results,
                "processing_time": processing_time,
                "arbitrator_id": self.agent_id,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            error_time = time.time() - start_time
            logger.error(f"中心仲裁失败 [会话ID: {session_id}]: {str(e)}")
            
            return {
                "session_id": session_id,
                "arbitration_result": {
                    "final_decision": "needs_human_review",
                    "confidence_score": 0.2,
                    "error": str(e),
                    "arbitration_reasoning": f"仲裁过程出现异常: {str(e)}"
                },
                "processing_time": error_time,
                "error": True,
                "timestamp": datetime.utcnow().isoformat()
            }
    
    async def _validate_sub_agent_results(self, sub_agent_results: Dict[str, Any]) -> Dict[str, Any]:
        """验证和标准化子智能体结果"""
        validated_results = {}
        
        for agent_id, result in sub_agent_results.items():
            if not isinstance(result, dict):
                logger.warning(f"智能体 {agent_id} 结果格式无效")
                continue
            
            # 标准化结果格式
            standardized_result = {
                "agent_id": agent_id,
                "status": result.get("status", "unknown"),
                "confidence": self._normalize_confidence(result.get("confidence", 0.5)),
                "analysis_result": result.get("analysis_result", {}),
                "processing_time": result.get("processing_time", 0),
                "timestamp": result.get("timestamp", datetime.utcnow().isoformat())
            }
            
            # 提取关键决策信息
            standardized_result["key_findings"] = self._extract_key_findings(result)
            standardized_result["risk_indicators"] = self._extract_risk_indicators(result)
            
            validated_results[agent_id] = standardized_result
        
        logger.debug(f"验证了 {len(validated_results)} 个子智能体结果")
        return validated_results
    
    def _normalize_confidence(self, confidence: Any) -> float:
        """标准化置信度值"""
        try:
            conf = float(confidence)
            return max(0.0, min(1.0, conf))
        except (ValueError, TypeError):
            return 0.5
    
    def _extract_key_findings(self, result: Dict[str, Any]) -> List[str]:
        """提取关键发现"""
        findings = []
        
        analysis = result.get("analysis_result", {})
        
        # 从不同字段提取关键信息
        for key in ["summary", "main_finding", "conclusion", "key_points"]:
            if key in analysis and analysis[key]:
                findings.append(str(analysis[key]))
        
        # 提取风险相关信息
        for key in ["detected_issues", "risk_factors", "problems"]:
            if key in analysis and isinstance(analysis[key], list):
                findings.extend(analysis[key])
        
        return findings[:5]  # 限制数量
    
    def _extract_risk_indicators(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """提取风险指标"""
        indicators = {
            "toxicity_score": 0.0,
            "risk_level": "low",
            "severity": 1,
            "categories": []
        }
        
        analysis = result.get("analysis_result", {})
        
        # 提取各种风险评分
        for score_key in ["toxicity_score", "risk_score", "severity_score"]:
            if score_key in analysis:
                indicators["toxicity_score"] = max(indicators["toxicity_score"], 
                                                 self._normalize_confidence(analysis[score_key]))
        
        # 提取风险等级
        for level_key in ["risk_level", "severity_level", "threat_level"]:
            if level_key in analysis:
                indicators["risk_level"] = str(analysis[level_key])
        
        # 提取类别信息
        for cat_key in ["detected_types", "categories", "toxicity_categories"]:
            if cat_key in analysis and isinstance(analysis[cat_key], list):
                indicators["categories"].extend(analysis[cat_key])
        
        return indicators
    
    async def _communicate_with_sub_agents(self, content: str, validated_results: Dict[str, Any],
                                         session_id: str) -> Dict[str, Any]:
        """与子智能体进行双向沟通 (修复通信ID问题)"""
        communication_log = {}
        
        # 修复：使用实际的智能体ID而不是任务ID
        agent_id_mapping = {
            'content_analysis': 'content_analyzer',
            'semantic_analysis': 'semantic_analyzer', 
            'sentiment_analysis': 'sentiment_analyzer',
            'toxicity_detection': 'toxicity_detector',
            'context_analysis': 'context_analyzer',
            'risk_assessment': 'risk_assessor'
        }
        
        # 向需要澄清的智能体发送询问
        for task_type, result in validated_results.items():
            if result["confidence"] < 0.7 or result["status"] != "completed":
                # 获取实际的智能体ID
                actual_agent_id = agent_id_mapping.get(task_type, result.get('agent_id'))
                if not actual_agent_id:
                    logger.warning(f"无法找到任务类型 {task_type} 对应的智能体ID")
                    continue
                    
                clarification_request = {
                    "session_id": session_id,
                    "request_type": "clarification",
                    "content": content,
                    "specific_questions": self._generate_clarification_questions(result),
                    "arbitrator_concerns": self._identify_concerns(result)
                }
                
                try:
                    # 发送澄清请求到实际的智能体ID
                    await self.comm_hub.send_message(
                        self.agent_id, actual_agent_id,
                        MessageType.COLLABORATION_REQUEST,
                        clarification_request,
                        priority=2
                    )
                    
                    # 等待响应
                    response = await self._wait_for_clarification(actual_agent_id, session_id)
                    communication_log[actual_agent_id] = {
                        "request_sent": True,
                        "response_received": response is not None,
                        "clarification_data": response
                    }
                    
                except Exception as e:
                    logger.warning(f"与智能体 {actual_agent_id} 通信失败: {str(e)}")
                    communication_log[actual_agent_id] = {
                        "request_sent": False,
                        "error": str(e)
                    }
        
        return communication_log
    
    def _generate_clarification_questions(self, result: Dict[str, Any]) -> List[str]:
        """生成澄清问题"""
        questions = []
        
        if result["confidence"] < 0.5:
            questions.append("请提供更多支持你判断的具体证据")
        
        if result["status"] != "completed":
            questions.append("请说明分析未完成的具体原因")
        
        risk_indicators = result.get("risk_indicators", {})
        if risk_indicators.get("toxicity_score", 0) > 0.3:
            questions.append("请详细说明检测到的毒性内容类型和位置")
        
        return questions
    
    def _identify_concerns(self, result: Dict[str, Any]) -> List[str]:
        """识别仲裁关注点"""
        concerns = []
        
        if result["confidence"] < 0.6:
            concerns.append("置信度偏低")
        
        if not result.get("key_findings"):
            concerns.append("缺少具体发现")
        
        if result.get("risk_indicators", {}).get("toxicity_score", 0) > 0.5:
            concerns.append("检测到较高风险")
        
        return concerns
    
    async def _wait_for_clarification(self, agent_id: str, session_id: str, 
                                    timeout: float = 10.0) -> Optional[Dict[str, Any]]:
        """等待澄清响应"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            message = await self.comm_hub.receive_message(self.agent_id, timeout=1.0)
            
            if (message and 
                message.sender_id == agent_id and
                message.message_type == MessageType.COLLABORATION_RESPONSE and
                session_id in str(message.content)):
                return message.content
        
        return None
    
    async def _analyze_agent_consensus(self, validated_results: Dict[str, Any]) -> Dict[str, Any]:
        """分析智能体共识"""
        if len(validated_results) < 2:
            return {
                "consensus_score": 1.0,
                "high_agreement_agents": list(validated_results.keys()),
                "conflicting_agents": [],
                "consensus_level": "high"
            }
        
        # 计算决策一致性
        decisions = []
        confidences = []
        risk_scores = []
        
        for result in validated_results.values():
            # 提取决策倾向
            risk_indicators = result.get("risk_indicators", {})
            risk_score = risk_indicators.get("toxicity_score", 0)
            confidence = result["confidence"]
            
            risk_scores.append(risk_score)
            confidences.append(confidence)
            
            # 简化决策分类
            if risk_score > 0.6:
                decisions.append("reject")
            elif risk_score > 0.3:
                decisions.append("review")
            else:
                decisions.append("approve")
        
        # 计算共识程度
        from collections import Counter
        decision_counts = Counter(decisions)
        most_common_decision, most_common_count = decision_counts.most_common(1)[0]
        consensus_score = most_common_count / len(decisions)
        
        # 识别一致和分歧的智能体
        high_agreement = []
        conflicting = []
        
        for agent_id, result in validated_results.items():
            risk_score = result.get("risk_indicators", {}).get("toxicity_score", 0)
            agent_decision = ("reject" if risk_score > 0.6 else 
                            "review" if risk_score > 0.3 else "approve")
            
            if agent_decision == most_common_decision:
                high_agreement.append(agent_id)
            else:
                conflicting.append(agent_id)
        
        # 计算加权共识（考虑置信度和专业度）
        weighted_consensus = self._calculate_weighted_consensus(validated_results, decisions)
        
        consensus_level = ("high" if consensus_score >= 0.8 else
                          "medium" if consensus_score >= 0.6 else "low")
        
        return {
            "consensus_score": consensus_score,
            "weighted_consensus": weighted_consensus,
            "high_agreement_agents": high_agreement,
            "conflicting_agents": conflicting,
            "consensus_level": consensus_level,
            "decision_distribution": dict(decision_counts),
            "average_confidence": sum(confidences) / len(confidences),
            "average_risk_score": sum(risk_scores) / len(risk_scores)
        }
    
    def _calculate_weighted_consensus(self, validated_results: Dict[str, Any], 
                                    decisions: List[str]) -> float:
        """计算加权共识度"""
        total_weight = 0
        agreement_weight = 0
        
        decision_mode = max(set(decisions), key=decisions.count)
        
        for i, (agent_id, result) in enumerate(validated_results.items()):
            # 智能体专业度权重
            expertise_weight = self.agent_expertise_weights.get(
                agent_id.replace("_agent", ""), 0.5
            )
            
            # 置信度权重
            confidence_weight = result["confidence"]
            
            # 综合权重
            combined_weight = expertise_weight * confidence_weight
            total_weight += combined_weight
            
            # 如果该智能体与主要决策一致
            if decisions[i] == decision_mode:
                agreement_weight += combined_weight
        
        return agreement_weight / total_weight if total_weight > 0 else 0.5
    
    async def _perform_complex_arbitration(self, content: str, validated_results: Dict[str, Any],
                                         consensus_analysis: Dict[str, Any], 
                                         context: Dict[str, Any]) -> Dict[str, Any]:
        """执行复杂推理仲裁"""
        # 准备仲裁输入数据
        sub_agent_analyses = self._format_analyses_for_llm(validated_results)
        detection_summary = self._generate_detection_summary(validated_results, consensus_analysis)
        
        # 构建仲裁提示词
        arbitration_prompt = self.arbitration_prompt.format(
            content=content,
            sub_agent_analyses=sub_agent_analyses,
            detection_summary=detection_summary
        )
        
        # 调用大模型进行复杂推理
        try:
            dashscope.api_key = self.config.get('dashscope_api_key')
            
            response = Generation.call(
                model='qwen-max',  # 使用最强模型
                prompt=arbitration_prompt,
                max_tokens=2000,
                temperature=0.05  # 低温度确保稳定性
            )
            
            if response.status_code == 200:
                arbitration_result = self._parse_arbitration_result(response.output.text)
                
                # 验证结果合理性
                validated_result = self._validate_arbitration_result(
                    arbitration_result, validated_results, consensus_analysis
                )
                
                return validated_result
            else:
                logger.error(f"仲裁LLM调用失败: {response.status_code}")
                return self._fallback_arbitration(validated_results, consensus_analysis)
                
        except Exception as e:
            logger.error(f"复杂推理仲裁失败: {str(e)}")
            return self._fallback_arbitration(validated_results, consensus_analysis)
    
    def _format_analyses_for_llm(self, validated_results: Dict[str, Any]) -> str:
        """格式化分析结果供LLM使用"""
        formatted_text = ""
        
        for agent_id, result in validated_results.items():
            formatted_text += f"\n【{agent_id}】\n"
            formatted_text += f"状态: {result['status']}\n"
            formatted_text += f"置信度: {result['confidence']:.2f}\n"
            
            if result["key_findings"]:
                formatted_text += f"关键发现: {'; '.join(result['key_findings'])}\n"
            
            risk_indicators = result.get("risk_indicators", {})
            if risk_indicators.get("toxicity_score", 0) > 0:
                formatted_text += f"风险评分: {risk_indicators['toxicity_score']:.2f}\n"
                formatted_text += f"风险等级: {risk_indicators.get('risk_level', 'unknown')}\n"
            
            if risk_indicators.get("categories"):
                formatted_text += f"检测类别: {', '.join(risk_indicators['categories'])}\n"
            
            formatted_text += "\n"
        
        return formatted_text
    
    def _generate_detection_summary(self, validated_results: Dict[str, Any], 
                                   consensus_analysis: Dict[str, Any]) -> str:
        """生成检测摘要"""
        summary = f"参与分析的智能体数量: {len(validated_results)}\n"
        summary += f"共识程度: {consensus_analysis['consensus_score']:.2f} ({consensus_analysis['consensus_level']})\n"
        summary += f"平均置信度: {consensus_analysis['average_confidence']:.2f}\n"
        summary += f"平均风险评分: {consensus_analysis['average_risk_score']:.2f}\n"
        
        if consensus_analysis['conflicting_agents']:
            summary += f"存在分歧的智能体: {', '.join(consensus_analysis['conflicting_agents'])}\n"
        
        summary += f"决策分布: {consensus_analysis['decision_distribution']}\n"
        
        return summary
    
    def _parse_arbitration_result(self, llm_response: str) -> Dict[str, Any]:
        """解析LLM仲裁结果"""
        try:
            import json
            start_idx = llm_response.find('{')
            end_idx = llm_response.rfind('}') + 1
            if start_idx != -1 and end_idx > start_idx:
                json_str = llm_response[start_idx:end_idx]
                result = json.loads(json_str)
                
                # 验证必要字段
                required_fields = ["final_decision", "confidence_score", "arbitration_reasoning"]
                if all(field in result for field in required_fields):
                    return result
        except Exception as e:
            logger.warning(f"解析仲裁结果失败: {str(e)}")
        
        # 解析失败时返回默认结果
        return {
            "final_decision": "needs_human_review",
            "confidence_score": 0.3,
            "arbitration_reasoning": "LLM仲裁结果解析失败",
            "evidence_analysis": {"supporting_evidence": [], "conflicting_evidence": []},
            "agent_consensus_analysis": {"consensus_score": 0.5},
            "risk_assessment": {"risk_level": "medium"},
            "recommendations": {"immediate_action": "人工审核"},
            "arbitrator_metadata": {"requires_escalation": True}
        }
    
    def _validate_arbitration_result(self, arbitration_result: Dict[str, Any],
                                   validated_results: Dict[str, Any],
                                   consensus_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """验证仲裁结果的合理性"""
        # 检查决策是否与强共识相符
        if (consensus_analysis["consensus_score"] > 0.8 and
            consensus_analysis["weighted_consensus"] > 0.8):
            
            # 如果有强共识但仲裁结果不一致，需要调整
            majority_decision = max(consensus_analysis["decision_distribution"].items(), 
                                  key=lambda x: x[1])[0]
            
            decision_mapping = {
                "approve": "approved",
                "reject": "rejected", 
                "review": "needs_human_review"
            }
            
            expected_decision = decision_mapping.get(majority_decision, "needs_human_review")
            
            if arbitration_result["final_decision"] != expected_decision:
                logger.info(f"调整仲裁决策以符合强共识: {expected_decision}")
                arbitration_result["final_decision"] = expected_decision
                arbitration_result["confidence_score"] = min(1.0, 
                    arbitration_result["confidence_score"] + 0.2)
        
        # 确保置信度合理
        arbitration_result["confidence_score"] = max(0.0, min(1.0, 
            arbitration_result.get("confidence_score", 0.5)))
        
        return arbitration_result
    
    def _fallback_arbitration(self, validated_results: Dict[str, Any],
                            consensus_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """备用仲裁逻辑"""
        # 基于简单规则的仲裁
        avg_risk = consensus_analysis["average_risk_score"]
        consensus_score = consensus_analysis["consensus_score"]
        
        if avg_risk > 0.7 and consensus_score > 0.6:
            decision = "rejected"
            confidence = 0.8
        elif avg_risk > 0.4 or consensus_score < 0.6:
            decision = "needs_human_review"
            confidence = 0.6
        else:
            decision = "approved"
            confidence = 0.7
        
        return {
            "final_decision": decision,
            "confidence_score": confidence,
            "arbitration_reasoning": f"基于规则的备用仲裁。平均风险: {avg_risk:.2f}, 共识度: {consensus_score:.2f}",
            "evidence_analysis": {
                "supporting_evidence": [f"智能体平均风险评分: {avg_risk:.2f}"],
                "conflicting_evidence": [],
                "evidence_reliability": consensus_score
            },
            "agent_consensus_analysis": consensus_analysis,
            "risk_assessment": {
                "primary_risks": ["自动检测的风险因素"],
                "risk_level": "high" if avg_risk > 0.7 else "medium" if avg_risk > 0.4 else "low",
                "potential_impact": "基于统计分析的影响评估"
            },
            "recommendations": {
                "immediate_action": "按决策执行",
                "follow_up_actions": ["监控反馈"],
                "monitoring_suggestions": ["关注用户反应"]
            },
            "arbitrator_metadata": {
                "processing_complexity": "medium",
                "decision_certainty": confidence,
                "requires_escalation": decision == "needs_human_review",
                "escalation_reason": "风险或共识度不足" if decision == "needs_human_review" else None
            }
        }
    
    async def _validate_and_optimize_decision(self, arbitration_result: Dict[str, Any],
                                            validated_results: Dict[str, Any],
                                            consensus_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """验证和优化最终决策"""
        # 最后的合理性检查
        final_decision = arbitration_result["final_decision"]
        confidence = arbitration_result["confidence_score"]
        
        # 安全第一原则：在高风险情况下倾向保守
        if (consensus_analysis["average_risk_score"] > 0.8 and 
            final_decision == "approved"):
            logger.warning("高风险内容被批准，调整为人工审核")
            arbitration_result["final_decision"] = "needs_human_review"
            arbitration_result["arbitration_reasoning"] += " [安全原则调整]"
        
        # 低置信度自动升级
        if confidence < 0.4:
            arbitration_result["final_decision"] = "needs_human_review"
            arbitration_result["arbitrator_metadata"]["requires_escalation"] = True
            arbitration_result["arbitrator_metadata"]["escalation_reason"] = "仲裁置信度过低"
        
        return arbitration_result
    
    async def _generate_comprehensive_report(self, content: str, validated_results: Dict[str, Any],
                                           arbitration_result: Dict[str, Any], final_decision: Dict[str, Any],
                                           communication_results: Dict[str, Any], 
                                           session_id: str) -> Dict[str, Any]:
        """生成综合报告"""
        return {
            "executive_summary": {
                "session_id": session_id,
                "final_decision": final_decision["final_decision"],
                "confidence_score": final_decision["confidence_score"],
                "participating_agents": len(validated_results),
                "communication_attempts": len(communication_results),
                "processing_complexity": final_decision.get("arbitrator_metadata", {}).get("processing_complexity", "medium")
            },
            "decision_trail": {
                "sub_agent_inputs": len(validated_results),
                "consensus_achieved": final_decision.get("agent_consensus_analysis", {}).get("consensus_score", 0) > 0.7,
                "arbitration_method": "LLM-enhanced" if "arbitration_reasoning" in final_decision else "rule-based",
                "escalation_triggered": final_decision.get("arbitrator_metadata", {}).get("requires_escalation", False)
            },
            "quality_metrics": {
                "evidence_reliability": final_decision.get("evidence_analysis", {}).get("evidence_reliability", 0.5),
                "agent_participation_rate": len([r for r in validated_results.values() if r["status"] == "completed"]) / len(validated_results),
                "communication_success_rate": len([c for c in communication_results.values() if c.get("response_received")]) / max(1, len(communication_results))
            },
            "arbitrator_assessment": {
                "decision_certainty": final_decision.get("arbitrator_metadata", {}).get("decision_certainty", 0.5),
                "primary_decision_factors": final_decision.get("evidence_analysis", {}).get("supporting_evidence", []),
                "areas_of_concern": final_decision.get("evidence_analysis", {}).get("conflicting_evidence", []),
                "recommended_monitoring": final_decision.get("recommendations", {}).get("monitoring_suggestions", [])
            }
        }
    
    def _update_arbitration_stats(self, final_decision: Dict[str, Any], processing_time: float):
        """更新仲裁统计信息"""
        self.arbitration_stats["total_arbitrations"] += 1
        
        decision = final_decision["final_decision"]
        if decision == "approved":
            self.arbitration_stats["approved_count"] += 1
        elif decision == "rejected":
            self.arbitration_stats["rejected_count"] += 1
        else:
            self.arbitration_stats["escalated_count"] += 1
        
        # 更新平均置信度
        confidence = final_decision["confidence_score"]
        total = self.arbitration_stats["total_arbitrations"]
        current_avg = self.arbitration_stats["average_confidence"]
        self.arbitration_stats["average_confidence"] = (current_avg * (total - 1) + confidence) / total
        
        # 记录处理时间
        self.arbitration_stats["processing_times"].append(processing_time)
        if len(self.arbitration_stats["processing_times"]) > 100:
            self.arbitration_stats["processing_times"] = self.arbitration_stats["processing_times"][-100:]
    
    def get_arbitrator_status(self) -> Dict[str, Any]:
        """获取仲裁器状态"""
        return {
            "arbitrator_id": self.agent_id,
            "status": "active",
            "statistics": self.arbitration_stats,
            "configuration": self.arbitration_config,
            "expertise_weights": self.agent_expertise_weights,
            "average_processing_time": (sum(self.arbitration_stats["processing_times"]) / 
                                      len(self.arbitration_stats["processing_times"])
                                      if self.arbitration_stats["processing_times"] else 0)
        }
