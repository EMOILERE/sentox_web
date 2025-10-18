import asyncio
import time
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from .base_agent import BaseAgent, AgentDecision
from .classifier_agent import ClassifierAgent
from .reasoner_agent import ReasonerAgent
from .coordinator_agent import CoordinatorAgent

logger = logging.getLogger(__name__)

class MultiAgentSystem:
    """多智能体系统管理器 - 协调整个多智能体工作流程"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.agents: Dict[str, BaseAgent] = {}
        self.processing_history: List[Dict[str, Any]] = []
        
        # 系统配置
        self.max_agents = config.get('max_agents', 5)
        self.coordination_timeout = config.get('coordination_timeout', 30)
        self.reasoning_depth = config.get('reasoning_depth', 3)
        self.consensus_threshold = config.get('consensus_threshold', 0.7)
        
        # 初始化智能体
        self._initialize_agents()
    
    def _initialize_agents(self):
        """初始化各类智能体"""
        agent_config = {
            'dashscope_api_key': self.config.get('dashscope_api_key'),
            'consensus_threshold': self.consensus_threshold
        }
        
        # 创建分类智能体
        self.agents['classifier_1'] = ClassifierAgent('classifier_1', agent_config)
        
        # 创建推理智能体
        self.agents['reasoner_1'] = ReasonerAgent('reasoner_1', agent_config)
        
        # 创建协调智能体
        self.agents['coordinator_1'] = CoordinatorAgent('coordinator_1', agent_config)
        
        logger.info(f"多智能体系统初始化完成，共{len(self.agents)}个智能体")
    
    async def process_content(self, content: str, platform: str = "unknown", 
                            user_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """处理内容的主要入口方法"""
        start_time = time.time()
        session_id = f"session_{int(time.time())}"
        
        logger.info(f"开始处理内容 [会话ID: {session_id}]")
        
        try:
            # 重置所有智能体的思维链和动作链
            for agent in self.agents.values():
                agent.reset_chains()
            
            # 第一阶段：初步分类
            classifier_result = await self._run_classification_phase(content, session_id)
            
            # 第二阶段：深度推理
            reasoner_result = await self._run_reasoning_phase(
                content, classifier_result, session_id
            )
            
            # 第三阶段：协调决策
            final_result = await self._run_coordination_phase(
                content, [classifier_result, reasoner_result], session_id
            )
            
            # 第四阶段：智能体协作
            collaboration_result = await self._run_collaboration_phase(
                content, final_result, session_id
            )
            
            processing_time = time.time() - start_time
            
            # 构建最终结果
            result = {
                "session_id": session_id,
                "content": content,
                "platform": platform,
                "processing_time": processing_time,
                "final_decision": final_result.decision,
                "confidence": final_result.confidence,
                "reasoning": final_result.reasoning,
                "supporting_evidence": final_result.supporting_evidence,
                "agent_decisions": {
                    "classifier": classifier_result.to_dict(),
                    "reasoner": reasoner_result.to_dict(),
                    "coordinator": final_result.to_dict()
                },
                "collaboration_summary": collaboration_result,
                "thought_chains": self._collect_thought_chains(),
                "action_chains": self._collect_action_chains(),
                "performance_metrics": self._collect_performance_metrics(),
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # 记录处理历史
            self.processing_history.append(result)
            
            logger.info(f"内容处理完成 [会话ID: {session_id}] [决策: {final_result.decision}] [耗时: {processing_time:.2f}s]")
            
            return result
            
        except Exception as e:
            error_time = time.time() - start_time
            logger.error(f"内容处理失败 [会话ID: {session_id}]: {str(e)}")
            
            return {
                "session_id": session_id,
                "content": content,
                "platform": platform,
                "processing_time": error_time,
                "final_decision": "error",
                "confidence": 0.0,
                "reasoning": f"处理过程中发生错误: {str(e)}",
                "supporting_evidence": ["系统错误"],
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    async def _run_classification_phase(self, content: str, session_id: str) -> AgentDecision:
        """运行分类阶段"""
        logger.debug(f"[{session_id}] 开始分类阶段")
        
        classifier = self.agents['classifier_1']
        context = {'session_id': session_id, 'phase': 'classification'}
        
        result = await classifier.process(content, context)
        logger.debug(f"[{session_id}] 分类结果: {result.decision} (置信度: {result.confidence:.2f})")
        
        return result
    
    async def _run_reasoning_phase(self, content: str, classifier_result: AgentDecision, 
                                 session_id: str) -> AgentDecision:
        """运行推理阶段"""
        logger.debug(f"[{session_id}] 开始推理阶段")
        
        reasoner = self.agents['reasoner_1']
        context = {
            'session_id': session_id,
            'phase': 'reasoning',
            'initial_classification': {
                'classification': classifier_result.decision,
                'confidence': classifier_result.confidence,
                'reasoning': classifier_result.reasoning,
                'evidence': classifier_result.supporting_evidence
            }
        }
        
        result = await reasoner.process(content, context)
        logger.debug(f"[{session_id}] 推理结果: {result.decision} (置信度: {result.confidence:.2f})")
        
        return result
    
    async def _run_coordination_phase(self, content: str, agent_decisions: List[AgentDecision], 
                                    session_id: str) -> AgentDecision:
        """运行协调阶段"""
        logger.debug(f"[{session_id}] 开始协调阶段")
        
        coordinator = self.agents['coordinator_1']
        
        # 格式化智能体决策结果
        formatted_decisions = []
        for decision in agent_decisions:
            formatted_decisions.append({
                'agent_id': decision.agent_id,
                'agent_type': self.agents[decision.agent_id].agent_type.value,
                'decision': decision.decision,
                'confidence': decision.confidence,
                'reasoning': decision.reasoning,
                'supporting_evidence': decision.supporting_evidence,
                'timestamp': decision.timestamp.isoformat()
            })
        
        context = {
            'session_id': session_id,
            'phase': 'coordination',
            'agent_decisions': formatted_decisions
        }
        
        result = await coordinator.process(content, context)
        logger.debug(f"[{session_id}] 协调结果: {result.decision} (置信度: {result.confidence:.2f})")
        
        return result
    
    async def _run_collaboration_phase(self, content: str, final_decision: AgentDecision, 
                                     session_id: str) -> Dict[str, Any]:
        """运行智能体协作阶段"""
        logger.debug(f"[{session_id}] 开始协作阶段")
        
        collaboration_results = {}
        
        # 收集所有智能体的协作数据
        for agent_id, agent in self.agents.items():
            shared_context = {
                'session_id': session_id,
                'final_decision': final_decision,
                'my_decision': final_decision if agent_id == 'coordinator_1' else None
            }
            
            if agent_id != 'coordinator_1':
                # 为非协调智能体提供它们自己的决策结果
                for decision in [d for d in self.processing_history[-1:] if 'agent_decisions' in d]:
                    agent_decisions = decision.get('agent_decisions', {})
                    if agent_id in agent_decisions:
                        shared_context['my_decision'] = agent_decisions[agent_id]
            
            try:
                collab_result = await agent.collaborate(list(self.agents.values()), shared_context)
                collaboration_results[agent_id] = collab_result
            except Exception as e:
                logger.warning(f"智能体 {agent_id} 协作失败: {str(e)}")
                collaboration_results[agent_id] = {
                    'agent_id': agent_id,
                    'error': str(e)
                }
        
        logger.debug(f"[{session_id}] 协作阶段完成")
        return collaboration_results
    
    def _collect_thought_chains(self) -> Dict[str, List[Dict[str, Any]]]:
        """收集所有智能体的思维链"""
        thought_chains = {}
        for agent_id, agent in self.agents.items():
            thought_chains[agent_id] = [thought.to_dict() for thought in agent.thought_chain]
        return thought_chains
    
    def _collect_action_chains(self) -> Dict[str, List[Dict[str, Any]]]:
        """收集所有智能体的动作链"""
        action_chains = {}
        for agent_id, agent in self.agents.items():
            action_chains[agent_id] = [action.to_dict() for action in agent.action_chain]
        return action_chains
    
    def _collect_performance_metrics(self) -> Dict[str, Dict[str, Any]]:
        """收集所有智能体的性能指标"""
        performance_metrics = {}
        for agent_id, agent in self.agents.items():
            performance_metrics[agent_id] = agent.get_performance_summary()
        return performance_metrics
    
    def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        total_processed = len(self.processing_history)
        
        # 统计决策分布
        decision_counts = {}
        total_processing_time = 0
        
        for record in self.processing_history:
            decision = record.get('final_decision', 'unknown')
            decision_counts[decision] = decision_counts.get(decision, 0) + 1
            total_processing_time += record.get('processing_time', 0)
        
        avg_processing_time = total_processing_time / total_processed if total_processed > 0 else 0
        
        return {
            "system_id": "sentox-multi-agent-system",
            "agents_count": len(self.agents),
            "total_processed": total_processed,
            "decision_distribution": decision_counts,
            "average_processing_time": avg_processing_time,
            "active_agents": list(self.agents.keys()),
            "last_activity": self.processing_history[-1]['timestamp'] if self.processing_history else None
        }
    
    def get_recent_processing_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """获取最近的处理历史"""
        return self.processing_history[-limit:]
    
    async def health_check(self) -> Dict[str, Any]:
        """系统健康检查"""
        health_status = {
            "overall_status": "healthy",
            "agents_status": {},
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # 检查各智能体状态
        for agent_id, agent in self.agents.items():
            try:
                # 简单的健康检查 - 创建一个测试决策
                test_content = "测试内容"
                test_context = {"health_check": True}
                
                start_time = time.time()
                # 这里我们不实际调用process，而是检查智能体的基本属性
                agent_health = {
                    "status": "healthy",
                    "agent_type": agent.agent_type.value,
                    "performance_metrics": agent.get_performance_summary(),
                    "last_check": datetime.utcnow().isoformat()
                }
                health_status["agents_status"][agent_id] = agent_health
                
            except Exception as e:
                health_status["agents_status"][agent_id] = {
                    "status": "unhealthy",
                    "error": str(e),
                    "last_check": datetime.utcnow().isoformat()
                }
                health_status["overall_status"] = "degraded"
        
        return health_status
