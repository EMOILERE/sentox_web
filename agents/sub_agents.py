import asyncio
import logging
import time
import json
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from datetime import datetime
from enum import Enum
import dashscope
from dashscope import Generation
import jieba
import re
from .communication_hub import AgentCommunicationHub, MessageType

logger = logging.getLogger(__name__)

class SubAgentType(Enum):
    """子智能体类型"""
    CONTENT_ANALYZER = "content_analyzer"
    SEMANTIC_ANALYZER = "semantic_analyzer"
    SENTIMENT_ANALYZER = "sentiment_analyzer"
    TOXICITY_DETECTOR = "toxicity_detector"
    CONTEXT_ANALYZER = "context_analyzer"
    RISK_ASSESSOR = "risk_assessor"

class BaseSubAgent(ABC):
    """子智能体基类"""
    
    def __init__(self, agent_id: str, agent_type: SubAgentType, 
                 communication_hub: AgentCommunicationHub, config: Dict[str, Any]):
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.comm_hub = communication_hub
        self.config = config
        self.is_running = False
        self.processed_tasks = 0
        self.error_count = 0
        
        # 工具集合
        self.available_tools = self._initialize_tools()
        
    async def initialize(self):
        """初始化智能体"""
        await self.comm_hub.register_agent(self.agent_id, {
            "type": self.agent_type.value,
            "capabilities": self.get_capabilities(),
            "tools": list(self.available_tools.keys()),
            "version": "1.0.0"
        })
        
        self.is_running = True
        logger.info(f"子智能体 {self.agent_id} 初始化完成")
    
    async def start_processing(self):
        """开始处理任务循环"""
        while self.is_running:
            try:
                # 接收任务
                message = await self.comm_hub.receive_message(self.agent_id, timeout=1.0)
                
                if message and message.message_type == MessageType.TASK_ASSIGNMENT:
                    await self._handle_task(message.content)
                elif message and message.message_type == MessageType.COLLABORATION_REQUEST:
                    await self._handle_collaboration_request(message)
                    
            except Exception as e:
                logger.error(f"智能体 {self.agent_id} 处理异常: {str(e)}")
                self.error_count += 1
                await asyncio.sleep(0.1)
    
    async def stop(self):
        """停止智能体"""
        self.is_running = False
        await self.comm_hub.unregister_agent(self.agent_id)
        logger.info(f"子智能体 {self.agent_id} 已停止")
    
    async def _handle_task(self, task_content: Dict[str, Any]):
        """处理分配的任务"""
        task_id = task_content.get('task_id')
        
        try:
            logger.debug(f"智能体 {self.agent_id} 开始处理任务 {task_id}")
            
            start_time = time.time()
            
            # 初始化推导过程记录
            reasoning_chain = {
                "agent_id": self.agent_id,
                "task_id": task_id,
                "start_time": start_time,
                "reasoning_steps": [],
                "decision_points": [],
                "tool_usage": [],
                "intermediate_results": {}
            }
            
            # 执行具体分析
            analysis_result = await self.analyze_content(
                task_content.get('content', ''),
                task_content.get('context', {}),
                reasoning_chain  # 传递推导链
            )
            
            processing_time = time.time() - start_time
            reasoning_chain["end_time"] = time.time()
            reasoning_chain["total_processing_time"] = processing_time
            
            # 发送结果（包含推导过程）
            result = {
                "task_id": task_id,
                "agent_id": self.agent_id,
                "task_type": task_content.get('task_type'),
                "status": "completed",
                "analysis_result": analysis_result,
                "confidence": analysis_result.get('confidence', 0.5),
                "processing_time": processing_time,
                "reasoning_chain": reasoning_chain,  # 添加推导链
                "timestamp": datetime.utcnow().isoformat()
            }
            
            await self.comm_hub.send_message(
                self.agent_id,
                "central_router",
                MessageType.RESULT_REPORT,
                result
            )
            
            self.processed_tasks += 1
            logger.debug(f"智能体 {self.agent_id} 完成任务 {task_id}")
            
        except Exception as e:
            logger.error(f"智能体 {self.agent_id} 任务 {task_id} 执行失败: {str(e)}")
            
            # 发送错误报告
            error_result = {
                "task_id": task_id,
                "agent_id": self.agent_id,
                "status": "failed",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
            
            await self.comm_hub.send_message(
                self.agent_id,
                "central_router",
                MessageType.ERROR_REPORT,
                error_result
            )
            
            self.error_count += 1
    
    async def _handle_collaboration_request(self, message):
        """处理其他智能体的协作请求"""
        try:
            task_description = message.content.get('task_description', '')
            data = message.content.get('data', {})
            
            # 执行协作任务
            collaboration_result = await self.collaborate_with_peer(task_description, data)
            
            # 发送响应
            response = {
                "status": "completed",
                "result": collaboration_result,
                "agent_id": self.agent_id,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            await self.comm_hub.send_message(
                self.agent_id,
                message.sender_id,
                MessageType.COLLABORATION_RESPONSE,
                response,
                correlation_id=message.correlation_id
            )
            
        except Exception as e:
            logger.error(f"智能体 {self.agent_id} 协作失败: {str(e)}")
            
            error_response = {
                "status": "failed",
                "error": str(e),
                "agent_id": self.agent_id
            }
            
            await self.comm_hub.send_message(
                self.agent_id,
                message.sender_id,
                MessageType.COLLABORATION_RESPONSE,
                error_response,
                correlation_id=message.correlation_id
            )
    
    @abstractmethod
    async def analyze_content(self, content: str, context: Dict[str, Any], reasoning_chain: Dict[str, Any] = None) -> Dict[str, Any]:
        """分析内容的核心方法"""
        pass
    
    def _add_reasoning_step(self, reasoning_chain: Dict[str, Any], step_type: str, description: str, data: Any = None):
        """添加推导步骤"""
        if reasoning_chain is None:
            return
        
        step = {
            "step_type": step_type,
            "description": description,
            "timestamp": time.time(),
            "data": data
        }
        reasoning_chain["reasoning_steps"].append(step)
        logger.info(f"🧠 [{self.agent_id}] {step_type}: {description}")
    
    def _add_decision_point(self, reasoning_chain: Dict[str, Any], decision: str, rationale: str, alternatives: List[str] = None):
        """添加决策点"""
        if reasoning_chain is None:
            return
        
        decision_point = {
            "decision": decision,
            "rationale": rationale,
            "alternatives": alternatives or [],
            "timestamp": time.time()
        }
        reasoning_chain["decision_points"].append(decision_point)
        logger.info(f"🎯 [{self.agent_id}] 决策: {decision} - {rationale}")
    
    def _record_tool_usage(self, reasoning_chain: Dict[str, Any], tool_name: str, input_data: Any, output_data: Any):
        """记录工具使用"""
        if reasoning_chain is None:
            return
        
        tool_usage = {
            "tool_name": tool_name,
            "input_data": input_data,
            "output_data": output_data,
            "timestamp": time.time()
        }
        reasoning_chain["tool_usage"].append(tool_usage)
        logger.info(f"🔧 [{self.agent_id}] 使用工具: {tool_name}")
    
    def _store_intermediate_result(self, reasoning_chain: Dict[str, Any], key: str, value: Any):
        """存储中间结果"""
        if reasoning_chain is None:
            return
        
        reasoning_chain["intermediate_results"][key] = {
            "value": value,
            "timestamp": time.time()
        }
        logger.info(f"💾 [{self.agent_id}] 存储中间结果: {key}")
    
    @abstractmethod
    def get_capabilities(self) -> List[str]:
        """获取智能体能力列表"""
        pass
    
    async def collaborate_with_peer(self, task_description: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """与其他智能体协作"""
        return {
            "message": f"智能体 {self.agent_id} 收到协作请求: {task_description}",
            "data_processed": len(str(data)),
            "capabilities_offered": self.get_capabilities()
        }
    
    def _initialize_tools(self) -> Dict[str, Any]:
        """初始化可用工具"""
        return {
            "text_analyzer": self._tool_text_analyzer,
            "pattern_matcher": self._tool_pattern_matcher,
            "feature_extractor": self._tool_feature_extractor,
            "llm_caller": self._tool_llm_caller
        }
    
    async def _tool_text_analyzer(self, text: str, analysis_type: str) -> Dict[str, Any]:
        """文本分析工具"""
        words = jieba.lcut(text)
        
        return {
            "word_count": len(words),
            "char_count": len(text),
            "sentence_count": len(re.split(r'[。！？.!?]', text)),
            "avg_word_length": sum(len(w) for w in words) / len(words) if words else 0,
            "unique_words": len(set(words)),
            "analysis_type": analysis_type
        }
    
    async def _tool_pattern_matcher(self, text: str, patterns: List[str]) -> Dict[str, Any]:
        """模式匹配工具"""
        matches = {}
        
        for pattern in patterns:
            try:
                matches[pattern] = len(re.findall(pattern, text, re.IGNORECASE))
            except re.error:
                # 如果是普通字符串，直接计数
                matches[pattern] = text.lower().count(pattern.lower())
        
        return {
            "pattern_matches": matches,
            "total_matches": sum(matches.values())
        }
    
    async def _tool_feature_extractor(self, text: str) -> Dict[str, Any]:
        """特征提取工具"""
        # 标点符号统计
        punctuation_count = sum(1 for c in text if c in '！？。，；：""''（）【】')
        
        # 数字和英文统计
        digit_count = sum(1 for c in text if c.isdigit())
        alpha_count = sum(1 for c in text if c.isalpha() and ord(c) < 128)
        
        # 情感词汇
        positive_words = ['好', '棒', '赞', '喜欢', '满意', '优秀', '完美', '推荐']
        negative_words = ['差', '烂', '垃圾', '讨厌', '不满', '糟糕', '失望', '后悔']
        
        pos_count = sum(1 for word in positive_words if word in text)
        neg_count = sum(1 for word in negative_words if word in text)
        
        return {
            "punctuation_count": punctuation_count,
            "digit_count": digit_count,
            "english_char_count": alpha_count,
            "positive_word_count": pos_count,
            "negative_word_count": neg_count,
            "sentiment_polarity": (pos_count - neg_count) / max(1, pos_count + neg_count),
            "text_complexity": len(set(jieba.lcut(text))) / max(1, len(jieba.lcut(text)))
        }
    
    async def _tool_llm_caller(self, prompt: str, model: str = "qwen-plus") -> Dict[str, Any]:
        """大模型调用工具"""
        try:
            dashscope.api_key = self.config.get('dashscope_api_key')
            
            response = Generation.call(
                model=model,
                prompt=prompt,
                max_tokens=500,
                temperature=0.2
            )
            
            if response.status_code == 200:
                return {
                    "status": "success",
                    "response": response.output.text,
                    "model": model
                }
            else:
                return {
                    "status": "error",
                    "error": f"API调用失败: {response.status_code}",
                    "model": model
                }
                
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "model": model
            }
    
    def get_status(self) -> Dict[str, Any]:
        """获取智能体状态"""
        return {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type.value,
            "is_running": self.is_running,
            "processed_tasks": self.processed_tasks,
            "error_count": self.error_count,
            "error_rate": self.error_count / max(1, self.processed_tasks),
            "available_tools": list(self.available_tools.keys()),
            "capabilities": self.get_capabilities()
        }

class ContentAnalyzerAgent(BaseSubAgent):
    """内容分析智能体"""
    
    def __init__(self, communication_hub: AgentCommunicationHub, config: Dict[str, Any]):
        super().__init__("content_analyzer", SubAgentType.CONTENT_ANALYZER, communication_hub, config)
    
    async def analyze_content(self, content: str, context: Dict[str, Any], reasoning_chain: Dict[str, Any] = None) -> Dict[str, Any]:
        """分析内容基础特征"""
        self._add_reasoning_step(reasoning_chain, "开始分析", f"开始对长度为{len(content)}字符的内容进行基础特征分析")
        
        # 使用文本分析工具
        self._add_reasoning_step(reasoning_chain, "工具调用", "调用文本分析工具获取基础统计信息")
        basic_stats = await self.available_tools["text_analyzer"](content, "basic_analysis")
        self._record_tool_usage(reasoning_chain, "text_analyzer", content, basic_stats)
        self._store_intermediate_result(reasoning_chain, "basic_stats", basic_stats)
        
        # 使用特征提取工具
        self._add_reasoning_step(reasoning_chain, "特征提取", "提取文本深层特征，包括情感词汇、标点符号等")
        features = await self.available_tools["feature_extractor"](content)
        self._record_tool_usage(reasoning_chain, "feature_extractor", content, features)
        self._store_intermediate_result(reasoning_chain, "features", features)
        
        # 模式匹配检测
        suspicious_patterns = ['http://', 'https://', '@', '#', '微信', 'QQ', '电话']
        self._add_reasoning_step(reasoning_chain, "模式匹配", f"检测可疑模式：{suspicious_patterns}")
        pattern_results = await self.available_tools["pattern_matcher"](content, suspicious_patterns)
        self._record_tool_usage(reasoning_chain, "pattern_matcher", suspicious_patterns, pattern_results)
        self._store_intermediate_result(reasoning_chain, "patterns", pattern_results)
        
        # 综合分析
        self._add_reasoning_step(reasoning_chain, "风险评估", "基于提取的特征计算内容风险评分")
        risk_score = 0.0
        
        # 基于特征计算风险分数
        if features["negative_word_count"] > features["positive_word_count"]:
            risk_score += 0.3
            self._add_decision_point(reasoning_chain, "负面词汇风险+0.3", 
                                   f"检测到{features['negative_word_count']}个负面词汇，超过{features['positive_word_count']}个正面词汇")
        
        if pattern_results["total_matches"] > 2:
            risk_score += 0.2
            self._add_decision_point(reasoning_chain, "可疑模式风险+0.2", 
                                   f"检测到{pattern_results['total_matches']}个可疑模式匹配")
        
        if basic_stats["char_count"] < 10:
            risk_score += 0.1  # 过短内容可能是垃圾信息
            self._add_decision_point(reasoning_chain, "短内容风险+0.1", 
                                   f"内容过短（{basic_stats['char_count']}字符），可能是垃圾信息")
        
        confidence = 0.8 if basic_stats["word_count"] > 5 else 0.6
        self._add_decision_point(reasoning_chain, f"置信度设为{confidence}", 
                               f"基于词汇数量{basic_stats['word_count']}确定分析置信度")
        
        final_risk = min(1.0, risk_score)
        self._store_intermediate_result(reasoning_chain, "final_risk", final_risk)
        
        recommendations = self._generate_content_recommendations(risk_score, features)
        self._add_reasoning_step(reasoning_chain, "生成建议", f"基于风险评分{final_risk:.2f}生成处理建议")
        
        return {
            "summary": "内容基础特征分析完成",
            "basic_statistics": basic_stats,
            "text_features": features,
            "pattern_analysis": pattern_results,
            "risk_score": final_risk,
            "confidence": confidence,
            "recommendations": recommendations
        }
    
    def get_capabilities(self) -> List[str]:
        return ["text_statistics", "feature_extraction", "pattern_recognition", "basic_risk_assessment"]
    
    def _generate_content_recommendations(self, risk_score: float, features: Dict[str, Any]) -> List[str]:
        """生成内容建议"""
        recommendations = []
        
        if risk_score > 0.5:
            recommendations.append("建议进一步审核")
        
        if features["negative_word_count"] > 3:
            recommendations.append("检测到较多负面词汇")
        
        if features["english_char_count"] > features["digit_count"] * 2:
            recommendations.append("包含较多英文字符，注意是否为广告")
        
        return recommendations

class SemanticAnalyzerAgent(BaseSubAgent):
    """语义分析智能体"""
    
    def __init__(self, communication_hub: AgentCommunicationHub, config: Dict[str, Any]):
        super().__init__("semantic_analyzer", SubAgentType.SEMANTIC_ANALYZER, communication_hub, config)
    
    async def analyze_content(self, content: str, context: Dict[str, Any], reasoning_chain: Dict[str, Any] = None) -> Dict[str, Any]:
        """深度语义分析"""
        self._add_reasoning_step(reasoning_chain, "语义分析开始", f"开始对内容进行深度语义理解，内容长度：{len(content)}字符")
        
        # 构建语义分析提示词
        self._add_reasoning_step(reasoning_chain, "构建提示词", "为大模型构建专业的语义分析提示词")
        semantic_prompt = f"""
请对以下中文内容进行深度语义分析：

内容：{content}

请从以下角度分析：
1. 主要语义内容和话题
2. 语言风格和表达方式
3. 潜在的隐含意义
4. 是否存在讽刺、暗示等修辞手法
5. 整体语义的清晰度和逻辑性

请以JSON格式回答：
{{
    "main_topic": "主要话题",
    "semantic_clarity": 0.0-1.0的清晰度评分,
    "language_style": "语言风格描述",
    "implicit_meaning": "潜在隐含意义",
    "rhetorical_devices": ["修辞手法列表"],
    "semantic_risk_factors": ["语义风险因素"],
    "overall_assessment": "整体评估"
}}
"""
        
        # 调用大模型进行语义分析
        self._add_reasoning_step(reasoning_chain, "大模型调用", "调用大模型进行深度语义理解")
        llm_result = await self.available_tools["llm_caller"](semantic_prompt)
        self._record_tool_usage(reasoning_chain, "llm_caller", "语义分析提示词", llm_result)
        
        # 解析结果
        self._add_reasoning_step(reasoning_chain, "结果解析", "解析大模型返回的语义分析结果")
        semantic_analysis = self._parse_semantic_result(llm_result.get("response", ""))
        self._store_intermediate_result(reasoning_chain, "llm_semantic_analysis", semantic_analysis)
        
        # 补充规则基础的语义特征
        self._add_reasoning_step(reasoning_chain, "规则增强", "使用规则方法补充语义特征分析")
        rule_based_features = await self._rule_based_semantic_analysis(content)
        self._store_intermediate_result(reasoning_chain, "rule_based_features", rule_based_features)
        
        # 置信度评估
        confidence = 0.85 if llm_result.get("status") == "success" else 0.6
        self._add_decision_point(reasoning_chain, f"置信度评估为{confidence}", 
                               f"基于大模型调用状态（{llm_result.get('status')}）确定分析置信度")
        
        return {
            "summary": "深度语义分析完成",
            "llm_analysis": semantic_analysis,
            "rule_based_features": rule_based_features,
            "confidence": confidence,
            "processing_method": "LLM + Rule-based hybrid"
        }
    
    def get_capabilities(self) -> List[str]:
        return ["semantic_understanding", "implicit_meaning_detection", "rhetorical_analysis", "topic_modeling"]
    
    def _parse_semantic_result(self, llm_response: str) -> Dict[str, Any]:
        """解析大模型的语义分析结果"""
        try:
            import json
            start_idx = llm_response.find('{')
            end_idx = llm_response.rfind('}') + 1
            if start_idx != -1 and end_idx > start_idx:
                json_str = llm_response[start_idx:end_idx]
                return json.loads(json_str)
        except:
            pass
        
        # 备用解析
        return {
            "main_topic": "无法解析",
            "semantic_clarity": 0.5,
            "language_style": "分析失败",
            "implicit_meaning": "需要人工分析",
            "rhetorical_devices": [],
            "semantic_risk_factors": ["分析异常"],
            "overall_assessment": "语义分析出现异常"
        }
    
    async def _rule_based_semantic_analysis(self, content: str) -> Dict[str, Any]:
        """基于规则的语义特征分析"""
        words = jieba.lcut(content)
        
        # 情感强烈的词汇
        intense_words = ['非常', '极其', '超级', '绝对', '完全', '彻底', '严重', '巨大']
        intense_count = sum(1 for word in intense_words if word in content)
        
        # 疑问和不确定表达
        uncertainty_words = ['可能', '也许', '或许', '大概', '估计', '似乎', '好像']
        uncertainty_count = sum(1 for word in uncertainty_words if word in content)
        
        # 时间相关词汇
        time_words = ['昨天', '今天', '明天', '现在', '以前', '将来', '刚才', '马上']
        time_count = sum(1 for word in time_words if word in content)
        
        return {
            "word_count": len(words),
            "unique_word_ratio": len(set(words)) / len(words) if words else 0,
            "intensity_level": intense_count / max(1, len(words)),
            "uncertainty_level": uncertainty_count / max(1, len(words)),
            "temporal_reference": time_count > 0,
            "semantic_density": len(set(words)) / max(1, len(content))
        }

class SentimentAnalyzerAgent(BaseSubAgent):
    """情感分析智能体"""
    
    def __init__(self, communication_hub: AgentCommunicationHub, config: Dict[str, Any]):
        super().__init__("sentiment_analyzer", SubAgentType.SENTIMENT_ANALYZER, communication_hub, config)
    
    async def analyze_content(self, content: str, context: Dict[str, Any], reasoning_chain: Dict[str, Any] = None) -> Dict[str, Any]:
        """情感分析"""
        self._add_reasoning_step(reasoning_chain, "情感分析开始", f"启动多层次情感分析流程，处理{len(content)}字符的内容")
        
        # 基础情感词典分析
        self._add_reasoning_step(reasoning_chain, "词典分析", "使用情感词典进行基础情感倾向识别")
        basic_sentiment = await self._basic_sentiment_analysis(content)
        self._store_intermediate_result(reasoning_chain, "basic_sentiment", basic_sentiment)
        self._add_decision_point(reasoning_chain, f"基础情感倾向：{basic_sentiment['dominant_emotion']}", 
                               f"极性评分：{basic_sentiment['polarity_score']:.2f}")
        
        # 高级情感分析（使用大模型）
        self._add_reasoning_step(reasoning_chain, "高级分析", "调用大模型进行深度情感理解")
        advanced_sentiment = await self._advanced_sentiment_analysis(content)
        self._store_intermediate_result(reasoning_chain, "advanced_sentiment", advanced_sentiment)
        self._add_decision_point(reasoning_chain, f"深度情感分析：{advanced_sentiment['sentiment_polarity']}", 
                               f"真实性评估：{advanced_sentiment['authenticity']}")
        
        # 情感强度分析
        self._add_reasoning_step(reasoning_chain, "强度分析", "分析情感表达的强度和激烈程度")
        intensity_analysis = await self._emotion_intensity_analysis(content)
        self._store_intermediate_result(reasoning_chain, "intensity_analysis", intensity_analysis)
        self._add_decision_point(reasoning_chain, f"情感强度：{intensity_analysis['overall_intensity']:.1f}/10", 
                               f"强化词数量：{intensity_analysis['intensifier_count']}")
        
        # 综合情感评分
        self._add_reasoning_step(reasoning_chain, "综合评估", "整合多个维度的分析结果")
        overall_sentiment = self._calculate_overall_sentiment(basic_sentiment, advanced_sentiment, intensity_analysis)
        self._store_intermediate_result(reasoning_chain, "overall_sentiment", overall_sentiment)
        self._add_decision_point(reasoning_chain, f"最终情感判断：{overall_sentiment['final_sentiment']}", 
                               f"综合极性：{overall_sentiment['polarity_score']:.2f}")
        
        return {
            "summary": "情感分析完成",
            "basic_sentiment": basic_sentiment,
            "advanced_sentiment": advanced_sentiment,
            "intensity_analysis": intensity_analysis,
            "overall_sentiment": overall_sentiment,
            "confidence": overall_sentiment.get("confidence", 0.8)
        }
    
    def get_capabilities(self) -> List[str]:
        return ["sentiment_classification", "emotion_detection", "intensity_analysis", "polarity_scoring"]
    
    async def _basic_sentiment_analysis(self, content: str) -> Dict[str, Any]:
        """基础情感词典分析"""
        positive_words = [
            '好', '棒', '赞', '喜欢', '爱', '满意', '开心', '高兴', '快乐', '优秀',
            '完美', '推荐', '支持', '同意', '欣赏', '感谢', '惊喜', '兴奋', '激动'
        ]
        
        negative_words = [
            '差', '烂', '坏', '讨厌', '恨', '不满', '生气', '愤怒', '失望', '沮丧',
            '糟糕', '垃圾', '废物', '后悔', '痛苦', '悲伤', '抱怨', '批评', '反对'
        ]
        
        neutral_words = [
            '还行', '一般', '普通', '正常', '平常', '可以', '尚可', '中等', '平均'
        ]
        
        pos_count = sum(1 for word in positive_words if word in content)
        neg_count = sum(1 for word in negative_words if word in content)
        neu_count = sum(1 for word in neutral_words if word in content)
        
        total_emotion_words = pos_count + neg_count + neu_count
        
        if total_emotion_words == 0:
            polarity = 0.0
            emotion = "neutral"
        else:
            polarity = (pos_count - neg_count) / total_emotion_words
            if polarity > 0.3:
                emotion = "positive"
            elif polarity < -0.3:
                emotion = "negative"
            else:
                emotion = "neutral"
        
        return {
            "positive_word_count": pos_count,
            "negative_word_count": neg_count,
            "neutral_word_count": neu_count,
            "polarity_score": polarity,
            "dominant_emotion": emotion,
            "emotion_word_density": total_emotion_words / max(1, len(jieba.lcut(content)))
        }
    
    async def _advanced_sentiment_analysis(self, content: str) -> Dict[str, Any]:
        """高级情感分析"""
        sentiment_prompt = f"""
请对以下中文内容进行详细的情感分析：

内容：{content}

请分析：
1. 整体情感倾向（积极/消极/中性）
2. 具体情感类型（如：愤怒、喜悦、悲伤、恐惧等）
3. 情感强度（1-10级）
4. 情感的真实性（是否为讽刺、反语等）

请以JSON格式回答：
{{
    "sentiment_polarity": "positive/negative/neutral",
    "specific_emotions": ["具体情感类型"],
    "intensity_level": 1-10的强度等级,
    "authenticity": "genuine/sarcastic/ironic",
    "confidence": 0.0-1.0的置信度
}}
"""
        
        llm_result = await self.available_tools["llm_caller"](sentiment_prompt)
        
        if llm_result.get("status") == "success":
            return self._parse_sentiment_result(llm_result.get("response", ""))
        else:
            return {
                "sentiment_polarity": "neutral",
                "specific_emotions": ["unknown"],
                "intensity_level": 1,
                "authenticity": "unknown",
                "confidence": 0.3
            }
    
    def _parse_sentiment_result(self, llm_response: str) -> Dict[str, Any]:
        """解析情感分析结果"""
        try:
            import json
            start_idx = llm_response.find('{')
            end_idx = llm_response.rfind('}') + 1
            if start_idx != -1 and end_idx > start_idx:
                json_str = llm_response[start_idx:end_idx]
                return json.loads(json_str)
        except:
            pass
        
        return {
            "sentiment_polarity": "neutral",
            "specific_emotions": ["parse_error"],
            "intensity_level": 1,
            "authenticity": "unknown",
            "confidence": 0.2
        }
    
    async def _emotion_intensity_analysis(self, content: str) -> Dict[str, Any]:
        """情感强度分析"""
        # 强化词汇
        intensifiers = ['非常', '极其', '超级', '特别', '相当', '十分', '很', '太', '最']
        intensifier_count = sum(1 for word in intensifiers if word in content)
        
        # 标点符号强度
        exclamation_count = content.count('！') + content.count('!')
        question_count = content.count('？') + content.count('?')
        
        # 重复字符（表示强调）
        repeated_chars = len(re.findall(r'(.)\1{2,}', content))
        
        # 大写字母比例（如果有英文）
        if any(c.isalpha() and ord(c) < 128 for c in content):
            caps_ratio = sum(1 for c in content if c.isupper()) / len(content)
        else:
            caps_ratio = 0
        
        # 综合强度评分
        intensity_score = (
            intensifier_count * 0.3 +
            exclamation_count * 0.2 +
            question_count * 0.1 +
            repeated_chars * 0.25 +
            caps_ratio * 0.15
        )
        
        return {
            "intensifier_count": intensifier_count,
            "exclamation_marks": exclamation_count,
            "question_marks": question_count,
            "repeated_characters": repeated_chars,
            "caps_ratio": caps_ratio,
            "overall_intensity": min(10, intensity_score * 2)
        }
    
    def _calculate_overall_sentiment(self, basic: Dict, advanced: Dict, intensity: Dict) -> Dict[str, Any]:
        """计算综合情感评分"""
        # 基础极性权重70%，高级分析权重30%
        basic_polarity = basic.get("polarity_score", 0)
        
        advanced_polarity_map = {"positive": 0.7, "negative": -0.7, "neutral": 0}
        advanced_polarity = advanced_polarity_map.get(advanced.get("sentiment_polarity", "neutral"), 0)
        
        combined_polarity = basic_polarity * 0.7 + advanced_polarity * 0.3
        
        # 考虑强度调整
        intensity_factor = min(1.5, 1 + intensity.get("overall_intensity", 0) / 10)
        final_polarity = combined_polarity * intensity_factor
        
        # 最终分类
        if final_polarity > 0.4:
            final_sentiment = "positive"
        elif final_polarity < -0.4:
            final_sentiment = "negative"
        else:
            final_sentiment = "neutral"
        
        return {
            "final_sentiment": final_sentiment,
            "polarity_score": max(-1, min(1, final_polarity)),
            "confidence": (basic.get("emotion_word_density", 0) + 
                          advanced.get("confidence", 0.5)) / 2,
            "intensity_adjusted": True
        }
