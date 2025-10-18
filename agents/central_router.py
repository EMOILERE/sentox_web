import asyncio
import logging
import time
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
import dashscope
from dashscope import Generation
from .communication_hub import AgentCommunicationHub, MessageType

logger = logging.getLogger(__name__)

class TaskType(Enum):
    """任务类型枚举"""
    CONTENT_ANALYSIS = "content_analysis"
    SEMANTIC_ANALYSIS = "semantic_analysis"
    SENTIMENT_ANALYSIS = "sentiment_analysis"
    CONTEXT_ANALYSIS = "context_analysis"
    TOXICITY_DETECTION = "toxicity_detection"
    PATTERN_RECOGNITION = "pattern_recognition"
    RISK_ASSESSMENT = "risk_assessment"

@dataclass
class TaskAssignment:
    """任务分配"""
    task_id: str
    task_type: TaskType
    assigned_agent: str
    content: str
    context: Dict[str, Any]
    priority: int
    deadline: Optional[datetime] = None
    dependencies: List[str] = None  # 依赖的其他任务ID

class CentralRouter:
    """中心路由器 - 负责任务拆解和智能体调度"""
    
    def __init__(self, communication_hub: AgentCommunicationHub, config: Dict[str, Any]):
        self.comm_hub = communication_hub
        self.config = config
        self.agent_id = "central_router"
        
        # 智能体能力映射
        self.agent_capabilities = {
            "content_analyzer": [TaskType.CONTENT_ANALYSIS, TaskType.PATTERN_RECOGNITION],
            "semantic_analyzer": [TaskType.SEMANTIC_ANALYSIS, TaskType.CONTEXT_ANALYSIS],
            "sentiment_analyzer": [TaskType.SENTIMENT_ANALYSIS],
            "toxicity_detector": [TaskType.TOXICITY_DETECTION, TaskType.RISK_ASSESSMENT],
            "context_analyzer": [TaskType.CONTEXT_ANALYSIS],
            "risk_assessor": [TaskType.RISK_ASSESSMENT]
        }
        
        # 任务队列和状态跟踪
        self.pending_tasks: Dict[str, TaskAssignment] = {}
        self.completed_tasks: Dict[str, Dict[str, Any]] = {}
        self.agent_workload: Dict[str, int] = {}
        
        # 任务拆解提示词模板
        self.task_decomposition_prompt = """
你是一个内容审核任务路由器。请分析以下内容并将其分解为具体的审核任务。

内容：{content}
平台：{platform}
上下文：{context}

请分析这个内容并确定需要哪些类型的分析任务。可选的任务类型：

1. content_analysis - 基础内容特征分析
2. semantic_analysis - 语义和深层含义分析  
3. sentiment_analysis - 情感倾向分析
4. context_analysis - 上下文和文化背景分析
5. toxicity_detection - 毒性内容检测
6. pattern_recognition - 模式识别和异常检测
7. risk_assessment - 风险评估

请按以下JSON格式返回任务分解结果：
{{
    "tasks": [
        {{
            "task_type": "任务类型",
            "priority": 1-10的优先级(1最高),
            "rationale": "选择此任务的理由",
            "specific_focus": "具体关注点",
            "dependencies": ["依赖的其他任务类型"]
        }}
    ],
    "overall_complexity": "low/medium/high",
    "estimated_processing_time": "预估处理时间(秒)",
    "special_considerations": "特殊考虑因素"
}}
"""
        
    async def initialize(self):
        """初始化路由器"""
        await self.comm_hub.register_agent(self.agent_id, {
            "type": "central_router",
            "capabilities": ["task_decomposition", "agent_coordination", "workflow_management"],
            "version": "1.0.0"
        })
        
        # 初始化智能体工作负载计数
        for agent_id in self.agent_capabilities.keys():
            self.agent_workload[agent_id] = 0
            
        logger.info("中心路由器初始化完成")
    
    async def process_content(self, content: str, platform: str = "unknown", 
                            context: Dict[str, Any] = None) -> Dict[str, Any]:
        """处理内容审核请求的主入口"""
        session_id = f"session_{int(time.time())}"
        context = context or {}
        
        logger.info(f"路由器开始处理内容 [会话ID: {session_id}]")
        
        start_time = time.time()
        
        try:
            # 1. 任务分解
            task_plan = await self._decompose_tasks(content, platform, context)
            
            # 2. 创建任务分配
            task_assignments = await self._create_task_assignments(
                session_id, content, task_plan, context
            )
            
            # 3. 分配任务给智能体
            assigned_tasks = await self._assign_tasks_to_agents(task_assignments)
            
            # 4. 监控任务执行
            task_results = await self._monitor_task_execution(assigned_tasks)
            
            # 5. 整合结果
            integrated_results = await self._integrate_results(task_results, session_id)
            
            processing_time = time.time() - start_time
            
            return {
                "session_id": session_id,
                "content": content,
                "platform": platform,
                "task_plan": task_plan,
                "task_results": task_results,
                "integrated_results": integrated_results,
                "processing_time": processing_time,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"路由器处理失败 [会话ID: {session_id}]: {str(e)}")
            return {
                "session_id": session_id,
                "error": str(e),
                "processing_time": time.time() - start_time,
                "timestamp": datetime.utcnow().isoformat()
            }
    
    async def _decompose_tasks(self, content: str, platform: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """使用大模型进行任务分解"""
        try:
            dashscope.api_key = self.config.get('dashscope_api_key')
            
            response = Generation.call(
                model='qwen-plus',
                prompt=self.task_decomposition_prompt.format(
                    content=content,
                    platform=platform,
                    context=str(context)
                ),
                max_tokens=1000,
                temperature=0.1
            )
            
            if response.status_code == 200:
                result_text = response.output.text
                
                # 尝试解析JSON
                try:
                    import json
                    start_idx = result_text.find('{')
                    end_idx = result_text.rfind('}') + 1
                    if start_idx != -1 and end_idx > start_idx:
                        json_str = result_text[start_idx:end_idx]
                        task_plan = json.loads(json_str)
                        
                        # 验证任务计划格式
                        if 'tasks' in task_plan and isinstance(task_plan['tasks'], list):
                            return task_plan
                    
                except json.JSONDecodeError as e:
                    logger.warning(f"无法解析任务分解结果: {e}")
                
                # 如果解析失败，使用备用逻辑
                return self._fallback_task_decomposition(content, platform)
            else:
                logger.error(f"任务分解API调用失败: {response.status_code}")
                return self._fallback_task_decomposition(content, platform)
                
        except Exception as e:
            logger.error(f"任务分解过程出错: {str(e)}")
            return self._fallback_task_decomposition(content, platform)
    
    def _fallback_task_decomposition(self, content: str, platform: str) -> Dict[str, Any]:
        """备用任务分解逻辑"""
        content_length = len(content)
        
        # 基于内容特征确定需要的任务
        tasks = []
        
        # 基础内容分析总是需要
        tasks.append({
            "task_type": "content_analysis",
            "priority": 8,
            "rationale": "基础内容特征分析",
            "specific_focus": "文本基础特征提取",
            "dependencies": []
        })
        
        # 语义分析
        if content_length > 20:
            tasks.append({
                "task_type": "semantic_analysis",
                "priority": 7,
                "rationale": "内容足够长，需要语义分析",
                "specific_focus": "深层语义理解",
                "dependencies": ["content_analysis"]
            })
        
        # 情感分析
        tasks.append({
            "task_type": "sentiment_analysis",
            "priority": 6,
            "rationale": "分析情感倾向",
            "specific_focus": "情感极性和强度",
            "dependencies": []
        })
        
        # 毒性检测
        tasks.append({
            "task_type": "toxicity_detection",
            "priority": 9,
            "rationale": "核心的毒性内容检测",
            "specific_focus": "有害内容识别",
            "dependencies": []
        })
        
        # 上下文分析（对于较长内容）
        if content_length > 50:
            tasks.append({
                "task_type": "context_analysis",
                "priority": 5,
                "rationale": "内容较长，需要上下文分析",
                "specific_focus": "文化背景和隐含含义",
                "dependencies": ["semantic_analysis"]
            })
        
        # 风险评估
        tasks.append({
            "task_type": "risk_assessment",
            "priority": 7,
            "rationale": "综合风险评估",
            "specific_focus": "整体风险等级",
            "dependencies": ["toxicity_detection", "sentiment_analysis"]
        })
        
        return {
            "tasks": tasks,
            "overall_complexity": "medium" if content_length > 100 else "low",
            "estimated_processing_time": min(10, max(3, content_length / 50)),
            "special_considerations": f"基于规则的任务分解 - 内容长度: {content_length}"
        }
    
    async def _create_task_assignments(self, session_id: str, content: str, 
                                     task_plan: Dict[str, Any], context: Dict[str, Any]) -> List[TaskAssignment]:
        """创建具体的任务分配"""
        assignments = []
        
        for i, task_info in enumerate(task_plan.get('tasks', [])):
            task_id = f"{session_id}_task_{i}"
            
            try:
                task_type = TaskType(task_info['task_type'])
            except ValueError:
                logger.warning(f"未知任务类型: {task_info['task_type']}")
                continue
            
            assignment = TaskAssignment(
                task_id=task_id,
                task_type=task_type,
                assigned_agent="",  # 稍后分配
                content=content,
                context={
                    **context,
                    "session_id": session_id,
                    "specific_focus": task_info.get('specific_focus', ''),
                    "rationale": task_info.get('rationale', ''),
                    "platform": context.get('platform', 'unknown')
                },
                priority=task_info.get('priority', 5),
                dependencies=task_info.get('dependencies', [])
            )
            
            assignments.append(assignment)
            self.pending_tasks[task_id] = assignment
        
        return assignments
    
    async def _assign_tasks_to_agents(self, task_assignments: List[TaskAssignment]) -> Dict[str, TaskAssignment]:
        """将任务分配给最适合的智能体"""
        assigned_tasks = {}
        
        for task in task_assignments:
            # 找到能处理此任务类型的智能体
            capable_agents = []
            for agent_id, capabilities in self.agent_capabilities.items():
                if task.task_type in capabilities:
                    capable_agents.append(agent_id)
            
            if not capable_agents:
                logger.warning(f"没有智能体能处理任务类型: {task.task_type}")
                continue
            
            # 选择工作负载最轻的智能体
            best_agent = min(capable_agents, key=lambda x: self.agent_workload.get(x, 0))
            task.assigned_agent = best_agent
            
            # 更新工作负载
            self.agent_workload[best_agent] = self.agent_workload.get(best_agent, 0) + 1
            
            assigned_tasks[task.task_id] = task
            
            logger.debug(f"任务 {task.task_id} ({task.task_type.value}) 分配给 {best_agent}")
        
        return assigned_tasks
    
    async def _monitor_task_execution(self, assigned_tasks: Dict[str, TaskAssignment]) -> Dict[str, Any]:
        """监控任务执行并收集结果"""
        task_results = {}
        
        # 发送任务给智能体
        for task_id, task in assigned_tasks.items():
            task_content = {
                "task_id": task_id,
                "task_type": task.task_type.value,
                "content": task.content,
                "context": task.context,
                "priority": task.priority,
                "dependencies": task.dependencies
            }
            
            try:
                await self.comm_hub.send_message(
                    self.agent_id,
                    task.assigned_agent,
                    MessageType.TASK_ASSIGNMENT,
                    task_content,
                    priority=task.priority
                )
            except Exception as e:
                logger.error(f"发送任务失败 {task_id} -> {task.assigned_agent}: {str(e)}")
        
        # 等待结果
        start_time = time.time()
        timeout = 30.0  # 30秒超时
        
        while len(task_results) < len(assigned_tasks) and time.time() - start_time < timeout:
            message = await self.comm_hub.receive_message(self.agent_id, timeout=1.0)
            
            if message and message.message_type == MessageType.RESULT_REPORT:
                task_id = message.content.get('task_id')
                if task_id in assigned_tasks:
                    task_results[task_id] = message.content
                    
                    # 减少智能体工作负载
                    agent_id = assigned_tasks[task_id].assigned_agent
                    if agent_id in self.agent_workload:
                        self.agent_workload[agent_id] = max(0, self.agent_workload[agent_id] - 1)
                    
                    logger.debug(f"收到任务结果: {task_id} from {message.sender_id}")
        
        # 处理超时的任务
        for task_id, task in assigned_tasks.items():
            if task_id not in task_results:
                logger.warning(f"任务超时: {task_id} (分配给 {task.assigned_agent})")
                task_results[task_id] = {
                    "task_id": task_id,
                    "status": "timeout",
                    "error": "Task execution timeout",
                    "agent_id": task.assigned_agent
                }
        
        return task_results
    
    async def _integrate_results(self, task_results: Dict[str, Any], session_id: str) -> Dict[str, Any]:
        """整合各个任务的结果"""
        successful_results = {}
        failed_tasks = []
        
        for task_id, result in task_results.items():
            if result.get('status') == 'completed':
                task_type = result.get('task_type', 'unknown')
                successful_results[task_type] = result
            else:
                failed_tasks.append({
                    "task_id": task_id,
                    "error": result.get('error', 'Unknown error'),
                    "agent_id": result.get('agent_id')
                })
        
        # 计算综合指标
        confidence_scores = []
        risk_indicators = []
        
        for result in successful_results.values():
            if 'confidence' in result:
                confidence_scores.append(result['confidence'])
            if 'risk_level' in result:
                risk_indicators.append(result['risk_level'])
        
        # 综合置信度（简单平均）
        overall_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.5
        
        # 综合风险等级（取最高）
        overall_risk = max(risk_indicators) if risk_indicators else 1
        
        integration_summary = {
            "session_id": session_id,
            "successful_tasks": len(successful_results),
            "total_tasks": len(task_results),
            "failed_tasks": len(failed_tasks),
            "overall_confidence": overall_confidence,
            "overall_risk_level": overall_risk,
            "task_results_summary": {
                task_type: {
                    "status": result.get('status'),
                    "confidence": result.get('confidence'),
                    "main_finding": result.get('analysis_result', {}).get('summary', 'N/A')
                }
                for task_type, result in successful_results.items()
            },
            "failed_tasks_summary": failed_tasks,
            "integration_timestamp": datetime.utcnow().isoformat()
        }
        
        # 更新共享上下文
        self.comm_hub.update_shared_context(
            f"integration_result_{session_id}", 
            integration_summary, 
            self.agent_id
        )
        
        return integration_summary
    
    def get_system_status(self) -> Dict[str, Any]:
        """获取路由器系统状态"""
        return {
            "router_id": self.agent_id,
            "pending_tasks": len(self.pending_tasks),
            "completed_tasks": len(self.completed_tasks),
            "agent_workload": self.agent_workload.copy(),
            "agent_capabilities": {
                agent: [t.value for t in capabilities] 
                for agent, capabilities in self.agent_capabilities.items()
            },
            "communication_stats": self.comm_hub.get_communication_stats()
        }
