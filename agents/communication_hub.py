import asyncio
import json
import logging
import time
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
import uuid

logger = logging.getLogger(__name__)

class MessageType(Enum):
    """消息类型枚举"""
    TASK_ASSIGNMENT = "task_assignment"
    RESULT_REPORT = "result_report"
    COLLABORATION_REQUEST = "collaboration_request"
    COLLABORATION_RESPONSE = "collaboration_response"
    ARBITRATION_REQUEST = "arbitration_request"
    ARBITRATION_RESULT = "arbitration_result"
    STATUS_UPDATE = "status_update"
    ERROR_REPORT = "error_report"

@dataclass
class AgentMessage:
    """智能体消息"""
    message_id: str
    sender_id: str
    receiver_id: str
    message_type: MessageType
    content: Dict[str, Any]
    timestamp: datetime
    correlation_id: Optional[str] = None  # 用于关联相关消息
    priority: int = 5  # 1-10，1最高优先级
    
    def to_dict(self):
        return {
            "message_id": self.message_id,
            "sender_id": self.sender_id,
            "receiver_id": self.receiver_id,
            "message_type": self.message_type.value,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "correlation_id": self.correlation_id,
            "priority": self.priority
        }

class AgentCommunicationHub:
    """智能体通信中心 - 实现MCP协议"""
    
    def __init__(self):
        self.message_queues: Dict[str, asyncio.Queue] = {}
        self.agent_registry: Dict[str, Dict[str, Any]] = {}
        self.message_history: List[AgentMessage] = []
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        self.communication_stats = {
            "total_messages": 0,
            "messages_by_type": {},
            "agent_activity": {},
            "errors": 0
        }
        
        # 中心存储，用于智能体间共享信息
        self.shared_context: Dict[str, Any] = {}
        
    async def register_agent(self, agent_id: str, agent_info: Dict[str, Any]):
        """注册智能体"""
        self.agent_registry[agent_id] = {
            **agent_info,
            "registered_at": datetime.utcnow(),
            "status": "active",
            "message_count": 0
        }
        
        # 为智能体创建消息队列
        if agent_id not in self.message_queues:
            self.message_queues[agent_id] = asyncio.Queue()
        
        logger.info(f"智能体 {agent_id} 已注册到通信中心")
    
    async def unregister_agent(self, agent_id: str):
        """注销智能体"""
        if agent_id in self.agent_registry:
            self.agent_registry[agent_id]["status"] = "inactive"
            self.agent_registry[agent_id]["unregistered_at"] = datetime.utcnow()
        
        logger.info(f"智能体 {agent_id} 已从通信中心注销")
    
    async def send_message(self, sender_id: str, receiver_id: str, 
                          message_type: MessageType, content: Dict[str, Any],
                          correlation_id: Optional[str] = None, priority: int = 5) -> str:
        """发送消息"""
        message_id = str(uuid.uuid4())
        
        message = AgentMessage(
            message_id=message_id,
            sender_id=sender_id,
            receiver_id=receiver_id,
            message_type=message_type,
            content=content,
            timestamp=datetime.utcnow(),
            correlation_id=correlation_id,
            priority=priority
        )
        
        # 检查接收者是否注册
        if receiver_id not in self.agent_registry:
            logger.error(f"接收者 {receiver_id} 未注册")
            raise ValueError(f"Agent {receiver_id} not registered")
        
        # 将消息放入接收者队列
        await self.message_queues[receiver_id].put(message)
        
        # 记录消息历史
        self.message_history.append(message)
        
        # 更新统计信息
        self._update_stats(message)
        
        logger.debug(f"消息已发送: {sender_id} -> {receiver_id} [{message_type.value}]")
        return message_id
    
    async def receive_message(self, agent_id: str, timeout: Optional[float] = None) -> Optional[AgentMessage]:
        """接收消息"""
        if agent_id not in self.message_queues:
            return None
        
        try:
            if timeout:
                message = await asyncio.wait_for(
                    self.message_queues[agent_id].get(), 
                    timeout=timeout
                )
            else:
                message = await self.message_queues[agent_id].get()
            
            # 更新智能体活动统计
            if agent_id in self.agent_registry:
                self.agent_registry[agent_id]["message_count"] += 1
                self.agent_registry[agent_id]["last_activity"] = datetime.utcnow()
            
            return message
            
        except asyncio.TimeoutError:
            return None
    
    async def broadcast_message(self, sender_id: str, message_type: MessageType, 
                               content: Dict[str, Any], exclude_agents: List[str] = None) -> List[str]:
        """广播消息给所有活跃智能体"""
        exclude_agents = exclude_agents or []
        message_ids = []
        
        for agent_id in self.agent_registry:
            if (agent_id != sender_id and 
                agent_id not in exclude_agents and 
                self.agent_registry[agent_id]["status"] == "active"):
                
                message_id = await self.send_message(
                    sender_id, agent_id, message_type, content
                )
                message_ids.append(message_id)
        
        return message_ids
    
    async def request_collaboration(self, requester_id: str, target_agent_id: str, 
                                   task_description: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """请求其他智能体协作"""
        correlation_id = str(uuid.uuid4())
        
        request_content = {
            "task_description": task_description,
            "data": data,
            "requester_capabilities": self.agent_registry.get(requester_id, {}).get("capabilities", [])
        }
        
        # 发送协作请求
        await self.send_message(
            requester_id, target_agent_id, 
            MessageType.COLLABORATION_REQUEST, 
            request_content,
            correlation_id=correlation_id,
            priority=3
        )
        
        # 等待响应
        start_time = time.time()
        timeout = 30.0  # 30秒超时
        
        while time.time() - start_time < timeout:
            message = await self.receive_message(requester_id, timeout=1.0)
            if (message and 
                message.message_type == MessageType.COLLABORATION_RESPONSE and
                message.correlation_id == correlation_id):
                return message.content
        
        # 超时处理
        logger.warning(f"协作请求超时: {requester_id} -> {target_agent_id}")
        return {"status": "timeout", "error": "Collaboration request timed out"}
    
    def update_shared_context(self, key: str, value: Any, agent_id: str):
        """更新共享上下文"""
        if "shared_data" not in self.shared_context:
            self.shared_context["shared_data"] = {}
        
        self.shared_context["shared_data"][key] = {
            "value": value,
            "updated_by": agent_id,
            "updated_at": datetime.utcnow().isoformat()
        }
        
        logger.debug(f"共享上下文已更新: {key} by {agent_id}")
    
    def get_shared_context(self, key: Optional[str] = None) -> Any:
        """获取共享上下文"""
        if key:
            return self.shared_context.get("shared_data", {}).get(key, {}).get("value")
        return self.shared_context.get("shared_data", {})
    
    def get_agent_status(self, agent_id: str) -> Dict[str, Any]:
        """获取智能体状态"""
        return self.agent_registry.get(agent_id, {})
    
    def get_communication_stats(self) -> Dict[str, Any]:
        """获取通信统计信息"""
        return {
            **self.communication_stats,
            "active_agents": len([a for a in self.agent_registry.values() if a["status"] == "active"]),
            "total_agents": len(self.agent_registry),
            "message_queue_sizes": {
                agent_id: queue.qsize() 
                for agent_id, queue in self.message_queues.items()
            }
        }
    
    def get_message_history(self, agent_id: Optional[str] = None, 
                           message_type: Optional[MessageType] = None,
                           limit: int = 100) -> List[Dict[str, Any]]:
        """获取消息历史"""
        filtered_messages = self.message_history
        
        if agent_id:
            filtered_messages = [
                msg for msg in filtered_messages 
                if msg.sender_id == agent_id or msg.receiver_id == agent_id
            ]
        
        if message_type:
            filtered_messages = [
                msg for msg in filtered_messages 
                if msg.message_type == message_type
            ]
        
        # 按时间倒序排列，取最新的limit条
        filtered_messages = sorted(filtered_messages, key=lambda x: x.timestamp, reverse=True)[:limit]
        
        return [msg.to_dict() for msg in filtered_messages]
    
    def _update_stats(self, message: AgentMessage):
        """更新统计信息"""
        self.communication_stats["total_messages"] += 1
        
        msg_type = message.message_type.value
        if msg_type not in self.communication_stats["messages_by_type"]:
            self.communication_stats["messages_by_type"][msg_type] = 0
        self.communication_stats["messages_by_type"][msg_type] += 1
        
        # 更新智能体活动统计
        for agent_id in [message.sender_id, message.receiver_id]:
            if agent_id not in self.communication_stats["agent_activity"]:
                self.communication_stats["agent_activity"][agent_id] = 0
            self.communication_stats["agent_activity"][agent_id] += 1
    
    async def cleanup_old_messages(self, max_age_hours: int = 24):
        """清理旧消息"""
        cutoff_time = datetime.utcnow().timestamp() - (max_age_hours * 3600)
        
        self.message_history = [
            msg for msg in self.message_history 
            if msg.timestamp.timestamp() > cutoff_time
        ]
        
        logger.info(f"已清理 {max_age_hours} 小时前的消息")
    
    async def health_check(self) -> Dict[str, Any]:
        """系统健康检查"""
        active_agents = sum(1 for a in self.agent_registry.values() if a["status"] == "active")
        total_queue_size = sum(q.qsize() for q in self.message_queues.values())
        
        # 检查是否有消息积压
        queue_health = "healthy"
        if total_queue_size > 1000:
            queue_health = "warning"
        elif total_queue_size > 5000:
            queue_health = "critical"
        
        return {
            "status": "healthy" if queue_health == "healthy" else queue_health,
            "active_agents": active_agents,
            "total_messages": self.communication_stats["total_messages"],
            "queue_health": queue_health,
            "total_queue_size": total_queue_size,
            "errors": self.communication_stats["errors"],
            "timestamp": datetime.utcnow().isoformat()
        }
