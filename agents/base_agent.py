from abc import ABC, abstractmethod
import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class AgentType(Enum):
    """Agent type enumeration"""
    CLASSIFIER = "classifier"  # Classifier agent
    REASONER = "reasoner"      # Reasoner agent
    COORDINATOR = "coordinator"  # Coordinator agent
    VALIDATOR = "validator"    # Validator agent
    ESCALATOR = "escalator"    # Escalator agent

class ActionType(Enum):
    """Action type enumeration"""
    ANALYZE = "analyze"
    CLASSIFY = "classify"
    REASON = "reason"
    VALIDATE = "validate"
    COORDINATE = "coordinate"
    ESCALATE = "escalate"
    CONSENSUS = "consensus"

@dataclass
class ThoughtStep:
    """Single thought step in the chain of thought"""
    step_id: str
    agent_id: str
    thought: str
    reasoning: str
    confidence: float
    evidence: List[str]
    timestamp: datetime
    
    def to_dict(self):
        return {
            "step_id": self.step_id,
            "agent_id": self.agent_id,
            "thought": self.thought,
            "reasoning": self.reasoning,
            "confidence": self.confidence,
            "evidence": self.evidence,
            "timestamp": self.timestamp.isoformat()
        }

@dataclass
class ActionStep:
    """Single action step in the chain of action"""
    action_id: str
    agent_id: str
    action_type: ActionType
    action_description: str
    input_data: Dict[str, Any]
    output_data: Dict[str, Any]
    success: bool
    error_message: Optional[str]
    execution_time: float
    timestamp: datetime
    
    def to_dict(self):
        return {
            "action_id": self.action_id,
            "agent_id": self.agent_id,
            "action_type": self.action_type.value,
            "action_description": self.action_description,
            "input_data": self.input_data,
            "output_data": self.output_data,
            "success": self.success,
            "error_message": self.error_message,
            "execution_time": self.execution_time,
            "timestamp": self.timestamp.isoformat()
        }

@dataclass
class AgentDecision:
    """Agent decision result"""
    agent_id: str
    decision: str
    confidence: float
    reasoning: str
    supporting_evidence: List[str]
    timestamp: datetime
    
    def to_dict(self):
        return {
            "agent_id": self.agent_id,
            "decision": self.decision,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "supporting_evidence": self.supporting_evidence,
            "timestamp": self.timestamp.isoformat()
        }

class BaseAgent(ABC):
    """Base agent class"""
    
    def __init__(self, agent_id: str, agent_type: AgentType, config: Dict[str, Any]):
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.config = config
        self.thought_chain: List[ThoughtStep] = []
        self.action_chain: List[ActionStep] = []
        self.performance_metrics = {
            "total_decisions": 0,
            "correct_decisions": 0,
            "average_confidence": 0.0,
            "average_processing_time": 0.0
        }
        
    def add_thought(self, thought: str, reasoning: str, confidence: float, evidence: List[str]) -> str:
        """Add thought step to chain of thought"""
        step_id = f"{self.agent_id}_thought_{len(self.thought_chain)}"
        thought_step = ThoughtStep(
            step_id=step_id,
            agent_id=self.agent_id,
            thought=thought,
            reasoning=reasoning,
            confidence=confidence,
            evidence=evidence,
            timestamp=datetime.utcnow()
        )
        self.thought_chain.append(thought_step)
        logger.debug(f"Agent {self.agent_id} added thought: {thought}")
        return step_id
    
    def add_action(self, action_type: ActionType, description: str, 
                   input_data: Dict[str, Any], output_data: Dict[str, Any],
                   success: bool, execution_time: float, error_message: Optional[str] = None) -> str:
        """Add action step to chain of action"""
        action_id = f"{self.agent_id}_action_{len(self.action_chain)}"
        action_step = ActionStep(
            action_id=action_id,
            agent_id=self.agent_id,
            action_type=action_type,
            action_description=description,
            input_data=input_data,
            output_data=output_data,
            success=success,
            error_message=error_message,
            execution_time=execution_time,
            timestamp=datetime.utcnow()
        )
        self.action_chain.append(action_step)
        logger.debug(f"Agent {self.agent_id} performed action: {description}")
        return action_id
    
    def get_thought_chain_summary(self) -> str:
        """Get chain of thought summary"""
        if not self.thought_chain:
            return "No thought records"
        
        summary = f"Agent {self.agent_id} thought process:\n"
        for i, thought in enumerate(self.thought_chain[-5:], 1):  # Latest 5 thought steps
            summary += f"{i}. {thought.thought}\n"
            summary += f"   Reasoning: {thought.reasoning}\n"
            summary += f"   Confidence: {thought.confidence:.2f}\n\n"
        
        return summary
    
    def get_action_chain_summary(self) -> str:
        """Get chain of action summary"""
        if not self.action_chain:
            return "No action records"
        
        summary = f"Agent {self.agent_id} action history:\n"
        for i, action in enumerate(self.action_chain[-5:], 1):  # Latest 5 actions
            status = "success" if action.success else "failed"
            summary += f"{i}. {action.action_description} [{status}]\n"
            summary += f"   Duration: {action.execution_time:.2f}s\n\n"
        
        return summary
    
    @abstractmethod
    async def process(self, content: str, context: Dict[str, Any]) -> AgentDecision:
        """Process content and return decision"""
        pass
    
    @abstractmethod
    async def collaborate(self, other_agents: List['BaseAgent'], 
                         shared_context: Dict[str, Any]) -> Dict[str, Any]:
        """Collaborate with other agents"""
        pass
    
    def update_performance_metrics(self, is_correct: bool, processing_time: float, confidence: float):
        """Update performance metrics"""
        self.performance_metrics["total_decisions"] += 1
        if is_correct:
            self.performance_metrics["correct_decisions"] += 1
        
        # Update average confidence
        current_avg = self.performance_metrics["average_confidence"]
        total = self.performance_metrics["total_decisions"]
        self.performance_metrics["average_confidence"] = (current_avg * (total - 1) + confidence) / total
        
        # Update average processing time
        current_avg_time = self.performance_metrics["average_processing_time"]
        self.performance_metrics["average_processing_time"] = (current_avg_time * (total - 1) + processing_time) / total
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        total = self.performance_metrics["total_decisions"]
        correct = self.performance_metrics["correct_decisions"]
        accuracy = (correct / total) if total > 0 else 0.0
        
        return {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type.value,
            "total_decisions": total,
            "accuracy": accuracy,
            "average_confidence": self.performance_metrics["average_confidence"],
            "average_processing_time": self.performance_metrics["average_processing_time"]
        }
    
    def reset_chains(self):
        """Reset thought chain and action chain (for new processing tasks)"""
        self.thought_chain = []
        self.action_chain = []
