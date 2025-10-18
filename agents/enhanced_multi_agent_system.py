import asyncio
import logging
import time
from typing import Dict, List, Any, Optional
from datetime import datetime
from .communication_hub import AgentCommunicationHub
from .central_router import CentralRouter
from .central_arbitrator import CentralArbitratorAgent
from .sub_agents import ContentAnalyzerAgent, SemanticAnalyzerAgent, SentimentAnalyzerAgent
from .toxicity_agents import ToxicityDetectorAgent, ContextAnalyzerAgent, RiskAssessorAgent

logger = logging.getLogger(__name__)

class EnhancedMultiAgentSystem:
    """增强版多智能体系统 - 总分总架构
    
    架构说明：
    1. 总（路由器）：中心路由器负责任务拆解和分配
    2. 分（子智能体群）：各专业智能体处理具体任务，可双向通信
    3. 总（仲裁器）：SenTox-GLDA升级版作为中心化仲裁智能体
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.session_counter = 0
        
        # 初始化通信中心
        self.communication_hub = AgentCommunicationHub()
        
        # 初始化中心路由器
        self.central_router = CentralRouter(self.communication_hub, config)
        
        # 初始化子智能体群
        self.sub_agents = {
            "content_analyzer": ContentAnalyzerAgent(self.communication_hub, config),
            "semantic_analyzer": SemanticAnalyzerAgent(self.communication_hub, config),
            "sentiment_analyzer": SentimentAnalyzerAgent(self.communication_hub, config),
            "toxicity_detector": ToxicityDetectorAgent(self.communication_hub, config),
            "context_analyzer": ContextAnalyzerAgent(self.communication_hub, config),
            "risk_assessor": RiskAssessorAgent(self.communication_hub, config)
        }
        
        # 初始化中心化仲裁智能体
        self.central_arbitrator = CentralArbitratorAgent(self.communication_hub, config)
        
        # 系统状态
        self.is_initialized = False
        self.is_running = False
        self.processing_history = []
        
        # 性能统计
        self.performance_stats = {
            "total_processed": 0,
            "successful_processes": 0,
            "failed_processes": 0,
            "average_processing_time": 0.0,
            "arbitration_accuracy": 0.0,
            "agent_utilization": {},
            "communication_efficiency": 0.0
        }
        
    async def initialize(self):
        """初始化整个多智能体系统"""
        if self.is_initialized:
            logger.warning("系统已经初始化")
            return
        
        logger.info("开始初始化增强版多智能体系统...")
        
        try:
            # 1. 初始化通信中心（已经初始化）
            logger.info("✓ 通信中心就绪")
            
            # 2. 初始化中心路由器
            await self.central_router.initialize()
            logger.info("✓ 中心路由器初始化完成")
            
            # 3. 初始化所有子智能体
            for agent_id, agent in self.sub_agents.items():
                await agent.initialize()
                logger.info(f"✓ 子智能体 {agent_id} 初始化完成")
            
            # 4. 初始化中心化仲裁智能体
            await self.central_arbitrator.initialize()
            logger.info("✓ 中心化仲裁智能体初始化完成")
            
            # 5. 启动子智能体处理循环
            self.agent_tasks = []
            for agent in self.sub_agents.values():
                task = asyncio.create_task(agent.start_processing())
                self.agent_tasks.append(task)
            
            self.is_initialized = True
            self.is_running = True
            
            logger.info("🎉 增强版多智能体系统初始化完成")
            logger.info(f"   - 中心路由器: 1个")
            logger.info(f"   - 子智能体: {len(self.sub_agents)}个")
            logger.info(f"   - 中心仲裁器: 1个")
            logger.info(f"   - 通信中心: 就绪")
            
        except Exception as e:
            logger.error(f"系统初始化失败: {str(e)}")
            await self._cleanup()
            raise
    
    async def process_content(self, content: str, platform: str = "unknown", 
                            context: Dict[str, Any] = None) -> Dict[str, Any]:
        """处理内容的主入口 - 完整的总分总流程"""
        if not self.is_initialized:
            raise RuntimeError("系统未初始化，请先调用 initialize()")
        
        self.session_counter += 1
        session_id = f"enhanced_session_{self.session_counter}_{int(time.time())}"
        context = context or {}
        context.update({
            "session_id": session_id,
            "platform": platform,
            "system_version": "enhanced_v2.0"
        })
        
        start_time = time.time()
        
        logger.info(f"🚀 开始增强版多智能体处理 [会话: {session_id}]")
        logger.info(f"   内容长度: {len(content)} 字符")
        logger.info(f"   来源平台: {platform}")
        
        try:
            # 阶段1: 中心路由器 - 任务拆解和分配
            logger.info(f"📋 阶段1: 中心路由器任务分解 [会话: {session_id}]")
            router_result = await self.central_router.process_content(content, platform, context)
            
            if "error" in router_result:
                raise Exception(f"路由器处理失败: {router_result['error']}")
            
            # 阶段2: 收集子智能体结果
            logger.info(f"🤖 阶段2: 收集子智能体分析结果 [会话: {session_id}]")
            sub_agent_results = router_result.get("task_results", {})
            
            # 阶段3: 子智能体间双向通信（如果需要）
            logger.info(f"🔄 阶段3: 子智能体协作通信 [会话: {session_id}]")
            collaboration_results = await self._facilitate_sub_agent_collaboration(
                content, sub_agent_results, session_id
            )
            
            # 阶段4: 中心化仲裁智能体 - 最终决策
            logger.info(f"⚖️ 阶段4: 中心化仲裁决策 [会话: {session_id}]")
            arbitration_result = await self.central_arbitrator.arbitrate_content(
                content, sub_agent_results, context
            )
            
            # 阶段5: 与仲裁器的双向沟通
            logger.info(f"💬 阶段5: 仲裁器双向沟通 [会话: {session_id}]")
            final_communication = await self._arbitrator_sub_agent_communication(
                arbitration_result, sub_agent_results, session_id
            )
            
            processing_time = time.time() - start_time
            
            # 构建最终结果
            final_result = {
                "session_id": session_id,
                "content": content,
                "platform": platform,
                "processing_time": processing_time,
                
                # 路由器结果
                "routing_phase": {
                    "task_plan": router_result.get("task_plan", {}),
                    "tasks_assigned": len(router_result.get("task_results", {})),
                    "routing_time": router_result.get("processing_time", 0)
                },
                
                # 子智能体阶段结果
                "sub_agents_phase": {
                    "agent_results": sub_agent_results,
                    "collaboration_results": collaboration_results,
                    "participating_agents": list(sub_agent_results.keys())
                },
                
                # 仲裁阶段结果
                "arbitration_phase": arbitration_result,
                
                # 最终决策
                "final_decision": arbitration_result.get("arbitration_result", {}).get("final_decision", "error"),
                "final_confidence": arbitration_result.get("arbitration_result", {}).get("confidence_score", 0.0),
                "final_reasoning": arbitration_result.get("arbitration_result", {}).get("arbitration_reasoning", ""),
                
                # 系统元数据
                "system_metadata": {
                    "total_agents_involved": 1 + len(self.sub_agents) + 1,  # 路由器 + 子智能体 + 仲裁器
                    "communication_events": len(final_communication),
                    "processing_complexity": arbitration_result.get("arbitration_result", {}).get("arbitrator_metadata", {}).get("processing_complexity", "medium"),
                    "requires_escalation": arbitration_result.get("arbitration_result", {}).get("arbitrator_metadata", {}).get("requires_escalation", False)
                },
                
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # 更新性能统计
            self._update_performance_stats(final_result, True, processing_time)
            
            # 记录处理历史
            self.processing_history.append(final_result)
            
            logger.info(f"✅ 增强版多智能体处理完成 [会话: {session_id}]")
            logger.info(f"   最终决策: {final_result['final_decision']}")
            logger.info(f"   置信度: {final_result['final_confidence']:.3f}")
            logger.info(f"   处理时间: {processing_time:.2f}秒")
            
            return final_result
            
        except Exception as e:
            error_time = time.time() - start_time
            logger.error(f"❌ 增强版多智能体处理失败 [会话: {session_id}]: {str(e)}")
            
            error_result = {
                "session_id": session_id,
                "content": content,
                "platform": platform,
                "final_decision": "error",
                "final_confidence": 0.0,
                "final_reasoning": f"系统处理异常: {str(e)}",
                "error": str(e),
                "processing_time": error_time,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            self._update_performance_stats(error_result, False, error_time)
            return error_result
    
    async def _facilitate_sub_agent_collaboration(self, content: str, sub_agent_results: Dict[str, Any],
                                                session_id: str) -> Dict[str, Any]:
        """促进子智能体间的协作"""
        collaboration_log = {}
        
        # 识别需要协作的场景
        collaboration_needs = self._identify_collaboration_needs(sub_agent_results)
        
        for collaboration in collaboration_needs:
            source_agent = collaboration["source_agent"]
            target_agent = collaboration["target_agent"]
            collaboration_type = collaboration["type"]
            
            logger.debug(f"促进协作: {source_agent} -> {target_agent} ({collaboration_type})")
            
            try:
                # 通过通信中心促进协作
                collaboration_result = await self.communication_hub.request_collaboration(
                    source_agent, target_agent,
                    f"协作请求: {collaboration_type}",
                    {
                        "session_id": session_id,
                        "content": content,
                        "collaboration_context": collaboration["context"]
                    }
                )
                
                collaboration_log[f"{source_agent}_{target_agent}"] = {
                    "type": collaboration_type,
                    "status": collaboration_result.get("status", "unknown"),
                    "result": collaboration_result
                }
                
            except Exception as e:
                logger.warning(f"协作失败 {source_agent} -> {target_agent}: {str(e)}")
                collaboration_log[f"{source_agent}_{target_agent}"] = {
                    "type": collaboration_type,
                    "status": "failed",
                    "error": str(e)
                }
        
        return collaboration_log
    
    def _identify_collaboration_needs(self, sub_agent_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """识别需要协作的场景"""
        collaborations = []
        
        # 检查结果中的低置信度或冲突
        for agent_id, result in sub_agent_results.items():
            if result.get("status") != "completed":
                continue
                
            confidence = result.get("confidence", 1.0)
            
            # 低置信度需要寻求其他智能体帮助
            if confidence < 0.6:
                if agent_id == "toxicity_detector":
                    # 毒性检测不确定时，请求语义分析帮助
                    collaborations.append({
                        "source_agent": agent_id,
                        "target_agent": "semantic_analyzer",
                        "type": "confidence_boost",
                        "context": "毒性检测置信度低，需要语义分析支持"
                    })
                elif agent_id == "sentiment_analyzer":
                    # 情感分析不确定时，请求上下文分析
                    collaborations.append({
                        "source_agent": agent_id,
                        "target_agent": "context_analyzer",
                        "type": "context_clarification",
                        "context": "情感分析需要上下文澄清"
                    })
        
        # 检查结果之间的潜在冲突
        toxicity_result = sub_agent_results.get("toxicity_detector", {})
        sentiment_result = sub_agent_results.get("sentiment_analyzer", {})
        
        if (toxicity_result.get("confidence", 0) > 0.7 and 
            sentiment_result.get("confidence", 0) > 0.7):
            
            # 如果毒性检测和情感分析结果可能冲突，促进协作
            collaborations.append({
                "source_agent": "toxicity_detector",
                "target_agent": "sentiment_analyzer",
                "type": "conflict_resolution",
                "context": "毒性检测和情感分析结果需要协调"
            })
        
        return collaborations
    
    async def _arbitrator_sub_agent_communication(self, arbitration_result: Dict[str, Any],
                                                sub_agent_results: Dict[str, Any],
                                                session_id: str) -> Dict[str, Any]:
        """仲裁器与子智能体的双向沟通"""
        communication_log = {}
        
        arbitration_data = arbitration_result.get("arbitration_result", {})
        
        # 如果仲裁器需要更多信息
        if arbitration_data.get("arbitrator_metadata", {}).get("requires_escalation"):
            # 向相关智能体请求更详细信息
            conflicting_agents = arbitration_data.get("agent_consensus_analysis", {}).get("conflicting_agents", [])
            
            for agent_id in conflicting_agents:
                if agent_id in self.sub_agents:
                    logger.debug(f"仲裁器请求 {agent_id} 提供更多信息")
                    
                    try:
                        # 发送详细信息请求
                        detailed_request = {
                            "session_id": session_id,
                            "request_type": "detailed_analysis",
                            "arbitrator_concerns": arbitration_data.get("evidence_analysis", {}).get("conflicting_evidence", []),
                            "specific_questions": [
                                "请提供更详细的分析依据",
                                "是否有其他可能的解释",
                                "置信度评估的具体原因"
                            ]
                        }
                        
                        response = await self.communication_hub.request_collaboration(
                            self.central_arbitrator.agent_id, agent_id,
                            "仲裁器请求详细信息", detailed_request
                        )
                        
                        communication_log[f"arbitrator_to_{agent_id}"] = {
                            "request_type": "detailed_analysis",
                            "response_received": response.get("status") == "completed",
                            "additional_info": response.get("result", {})
                        }
                        
                    except Exception as e:
                        logger.warning(f"仲裁器与 {agent_id} 通信失败: {str(e)}")
                        communication_log[f"arbitrator_to_{agent_id}"] = {
                            "request_type": "detailed_analysis",
                            "status": "failed",
                            "error": str(e)
                        }
        
        return communication_log
    
    async def shutdown(self):
        """关闭多智能体系统"""
        if not self.is_running:
            return
        
        logger.info("正在关闭增强版多智能体系统...")
        
        # 停止子智能体
        for agent in self.sub_agents.values():
            await agent.stop()
        
        # 取消任务
        if hasattr(self, 'agent_tasks'):
            for task in self.agent_tasks:
                task.cancel()
        
        # 注销智能体
        await self.communication_hub.unregister_agent(self.central_router.agent_id)
        await self.communication_hub.unregister_agent(self.central_arbitrator.agent_id)
        
        self.is_running = False
        logger.info("增强版多智能体系统已关闭")
    
    def _update_performance_stats(self, result: Dict[str, Any], success: bool, processing_time: float):
        """更新性能统计"""
        self.performance_stats["total_processed"] += 1
        
        if success:
            self.performance_stats["successful_processes"] += 1
        else:
            self.performance_stats["failed_processes"] += 1
        
        # 更新平均处理时间
        total = self.performance_stats["total_processed"]
        current_avg = self.performance_stats["average_processing_time"]
        self.performance_stats["average_processing_time"] = (current_avg * (total - 1) + processing_time) / total
        
        # 更新智能体利用率
        if success and "sub_agents_phase" in result:
            participating_agents = result["sub_agents_phase"].get("participating_agents", [])
            for agent_id in participating_agents:
                if agent_id not in self.performance_stats["agent_utilization"]:
                    self.performance_stats["agent_utilization"][agent_id] = 0
                self.performance_stats["agent_utilization"][agent_id] += 1
    
    def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        status = {
            "system_name": "Enhanced Multi-Agent Content Moderation System",
            "version": "2.0",
            "is_initialized": self.is_initialized,
            "is_running": self.is_running,
            "architecture": "总分总 (Router-SubAgents-Arbitrator)",
            
            "components": {
                "central_router": {
                    "status": "active" if self.is_running else "inactive",
                    "agent_id": self.central_router.agent_id if hasattr(self.central_router, 'agent_id') else "unknown"
                },
                "sub_agents": {
                    "count": len(self.sub_agents),
                    "agents": list(self.sub_agents.keys()),
                    "status": {agent_id: agent.get_status() for agent_id, agent in self.sub_agents.items()}
                },
                "central_arbitrator": {
                    "status": "active" if self.is_running else "inactive",
                    "agent_id": self.central_arbitrator.agent_id,
                    "arbitrator_stats": self.central_arbitrator.get_arbitrator_status() if self.is_running else {}
                },
                "communication_hub": self.communication_hub.get_communication_stats()
            },
            
            "performance": self.performance_stats,
            "recent_activity": {
                "total_sessions": len(self.processing_history),
                "last_session": self.processing_history[-1] if self.processing_history else None,
                "success_rate": (self.performance_stats["successful_processes"] / 
                               max(1, self.performance_stats["total_processed"]))
            }
        }
        
        return status
    
    def get_recent_processing_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """获取最近的处理历史"""
        return self.processing_history[-limit:]
    
    async def health_check(self) -> Dict[str, Any]:
        """系统健康检查"""
        if not self.is_initialized:
            return {
                "overall_status": "not_initialized",
                "message": "系统未初始化"
            }
        
        health_issues = []
        
        # 检查通信中心
        comm_health = await self.communication_hub.health_check()
        if comm_health["status"] != "healthy":
            health_issues.append(f"通信中心状态异常: {comm_health['status']}")
        
        # 检查子智能体
        for agent_id, agent in self.sub_agents.items():
            if not agent.is_running:
                health_issues.append(f"子智能体 {agent_id} 未运行")
            elif agent.error_count > agent.processed_tasks * 0.1:  # 错误率超过10%
                health_issues.append(f"子智能体 {agent_id} 错误率过高")
        
        overall_status = "healthy" if not health_issues else "degraded"
        
        return {
            "overall_status": overall_status,
            "health_issues": health_issues,
            "communication_health": comm_health,
            "agent_count": len(self.sub_agents),
            "active_agents": len([a for a in self.sub_agents.values() if a.is_running]),
            "total_processed": self.performance_stats["total_processed"],
            "success_rate": (self.performance_stats["successful_processes"] / 
                           max(1, self.performance_stats["total_processed"])),
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def _cleanup(self):
        """清理资源"""
        try:
            await self.shutdown()
        except Exception as e:
            logger.error(f"清理过程中出错: {str(e)}")
    
    def __del__(self):
        """析构函数"""
        if self.is_running:
            logger.warning("系统对象被销毁但未正确关闭，尝试清理...")
            try:
                asyncio.create_task(self._cleanup())
            except Exception:
                pass
