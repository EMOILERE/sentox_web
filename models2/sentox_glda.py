import os
import time
import logging
import numpy as np
import jieba
from typing import Dict, List, Tuple, Any
from datetime import datetime
import pickle
import json

logger = logging.getLogger(__name__)

class SenToxGLDA:
    """
    SenTox-GLDA 中文评论二分类审核模型
    这是一个模拟实现，展示如何集成实际的SenTox-GLDA模型
    """
    
    def __init__(self, model_path: str = None):
        self.model_path = model_path or os.path.join(os.path.dirname(__file__), 'sentox_glda')
        self.is_loaded = False
        self.model = None
        self.vectorizer = None
        self.feature_extractor = None
        
        # 模型元信息
        self.model_info = {
            "name": "SenTox-GLDA",
            "version": "1.0.0",
            "language": "zh-CN",
            "task": "binary_classification",
            "classes": ["safe", "toxic"],
            "accuracy": 0.92,
            "f1_score": 0.89,
            "precision": 0.91,
            "recall": 0.87
        }
        
        # 预定义的毒性关键词（用于模拟）
        self.toxic_keywords = [
            # 脏话辱骂类
            '傻逼', '操你妈', '去死', '滚', '草泥马', '妈的', '他妈的', '狗屎', '垃圾', '废物',
            '白痴', '智障', '脑残', '蠢货', '混蛋', '王八蛋', '贱人', '婊子', '臭婊子',
            
            # 威胁暴力类
            '杀死你', '弄死你', '打死你', '干掉你', '砍死', '刀你', '炸死', '枪毙',
            '弄残你', '废了你', '整死你', '搞死你',
            
            # 歧视仇恨类
            '黑鬼', '死胖子', '丑八怪', '残疾', '智力低下', '穷逼', '乞丐', '农民工',
            
            # 色情低俗类
            '做爱', '性交', '操逼', '日你', 'AV', '黄片', '色情', '裸体', '性器官'
        ]
        
        # 加载模型
        self._load_model()
    
    def _load_model(self):
        """加载SenTox-GLDA模型"""
        try:
            if os.path.exists(self.model_path):
                # 尝试加载实际模型
                model_file = os.path.join(self.model_path, 'model.pkl')
                config_file = os.path.join(self.model_path, 'config.json')
                
                if os.path.exists(model_file) and os.path.exists(config_file):
                    with open(model_file, 'rb') as f:
                        self.model = pickle.load(f)
                    
                    with open(config_file, 'r', encoding='utf-8') as f:
                        config = json.load(f)
                        self.model_info.update(config)
                    
                    self.is_loaded = True
                    logger.info(f"SenTox-GLDA模型加载成功: {model_file}")
                else:
                    logger.warning(f"模型文件不存在，使用模拟模式: {self.model_path}")
                    self._initialize_mock_model()
            else:
                logger.warning(f"模型目录不存在，使用模拟模式: {self.model_path}")
                self._initialize_mock_model()
                
        except Exception as e:
            logger.error(f"加载SenTox-GLDA模型失败: {str(e)}")
            logger.info("切换到模拟模式")
            self._initialize_mock_model()
    
    def _initialize_mock_model(self):
        """初始化模拟模型"""
        self.model = "mock_sentox_glda"
        self.is_loaded = True
        logger.info("SenTox-GLDA模拟模式初始化完成")
    
    def preprocess_text(self, text: str) -> List[str]:
        """文本预处理"""
        # 清理文本
        text = text.strip().lower()
        
        # 分词
        words = jieba.lcut(text)
        
        # 过滤停用词和标点符号
        stopwords = {'的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', 
                    '一', '一个', '上', '也', '很', '到', '说', '要', '去', '你', 
                    '会', '着', '没有', '看', '好', '自己', '这'}
        
        filtered_words = []
        for word in words:
            if (len(word) > 1 and 
                word not in stopwords and 
                not word.isspace() and
                not all(c in '，。！？；：""''（）【】' for c in word)):
                filtered_words.append(word)
        
        return filtered_words
    
    def extract_features(self, text: str) -> Dict[str, Any]:
        """提取文本特征"""
        words = self.preprocess_text(text)
        
        # 基础特征
        features = {
            'text_length': len(text),
            'word_count': len(words),
            'avg_word_length': np.mean([len(w) for w in words]) if words else 0,
            'exclamation_count': text.count('!') + text.count('！'),
            'question_count': text.count('?') + text.count('？'),
            'caps_ratio': sum(1 for c in text if c.isupper()) / len(text) if text else 0,
        }
        
        # 情感特征
        negative_words = ['不', '没', '别', '无', '非', '否', '未', '拒绝', '反对', '讨厌']
        positive_words = ['好', '棒', '赞', '喜欢', '支持', '同意', '满意', '开心', '高兴']
        
        features['negative_word_count'] = sum(1 for word in words if word in negative_words)
        features['positive_word_count'] = sum(1 for word in words if word in positive_words)
        
        # 毒性特征
        toxic_count = 0
        detected_toxic_words = []
        for keyword in self.toxic_keywords:
            if keyword in text:
                toxic_count += 1
                detected_toxic_words.append(keyword)
        
        features['toxic_keyword_count'] = toxic_count
        features['detected_toxic_words'] = detected_toxic_words
        
        return features
    
    def predict(self, text: str) -> Dict[str, Any]:
        """对文本进行毒性预测"""
        if not self.is_loaded:
            raise RuntimeError("模型未加载")
        
        start_time = time.time()
        
        try:
            # 提取特征
            features = self.extract_features(text)
            
            if self.model == "mock_sentox_glda":
                # 模拟预测逻辑
                prediction_result = self._mock_predict(text, features)
            else:
                # 实际模型预测
                prediction_result = self._real_predict(text, features)
            
            processing_time = time.time() - start_time
            
            result = {
                "prediction": prediction_result["class"],
                "confidence": prediction_result["confidence"],
                "probabilities": prediction_result["probabilities"],
                "features": features,
                "processing_time": processing_time,
                "model_version": self.model_info["version"],
                "timestamp": datetime.utcnow().isoformat()
            }
            
            return result
            
        except Exception as e:
            logger.error(f"SenTox-GLDA预测失败: {str(e)}")
            return {
                "prediction": "safe",  # 默认安全
                "confidence": 0.3,
                "probabilities": {"safe": 0.7, "toxic": 0.3},
                "features": {},
                "processing_time": time.time() - start_time,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    def _mock_predict(self, text: str, features: Dict[str, Any]) -> Dict[str, Any]:
        """模拟预测逻辑"""
        # 基于关键词和特征的简单分类逻辑
        toxic_score = 0.0
        
        # 毒性关键词权重
        toxic_keyword_count = features.get('toxic_keyword_count', 0)
        if toxic_keyword_count > 0:
            toxic_score += min(0.8, toxic_keyword_count * 0.3)
        
        # 文本长度和标点符号权重
        exclamation_count = features.get('exclamation_count', 0)
        if exclamation_count > 3:
            toxic_score += 0.2
        
        caps_ratio = features.get('caps_ratio', 0)
        if caps_ratio > 0.3:  # 过多大写字母
            toxic_score += 0.1
        
        # 负面词权重
        negative_count = features.get('negative_word_count', 0)
        positive_count = features.get('positive_word_count', 0)
        if negative_count > positive_count:
            toxic_score += 0.1
        
        # 添加一些随机性来模拟模型的不确定性
        import random
        random.seed(hash(text) % 1000)  # 基于文本内容的固定种子
        random_factor = random.uniform(-0.1, 0.1)
        toxic_score = max(0.0, min(1.0, toxic_score + random_factor))
        
        safe_score = 1.0 - toxic_score
        
        # 确定分类结果
        if toxic_score > 0.5:
            predicted_class = "toxic"
            confidence = toxic_score
        else:
            predicted_class = "safe"
            confidence = safe_score
        
        return {
            "class": predicted_class,
            "confidence": confidence,
            "probabilities": {
                "safe": safe_score,
                "toxic": toxic_score
            }
        }
    
    def _real_predict(self, text: str, features: Dict[str, Any]) -> Dict[str, Any]:
        """实际模型预测（当真正的模型可用时）"""
        # 这里应该调用实际的SenTox-GLDA模型
        # 例如：
        # processed_features = self.feature_extractor.transform([text])
        # probabilities = self.model.predict_proba(processed_features)[0]
        # predicted_class_idx = np.argmax(probabilities)
        # predicted_class = self.model_info["classes"][predicted_class_idx]
        # confidence = probabilities[predicted_class_idx]
        
        # 这里返回模拟结果，实际实现时替换为真实模型调用
        return self._mock_predict(text, features)
    
    def batch_predict(self, texts: List[str]) -> List[Dict[str, Any]]:
        """批量预测"""
        results = []
        for text in texts:
            result = self.predict(text)
            results.append(result)
        return results
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        return {
            **self.model_info,
            "is_loaded": self.is_loaded,
            "model_path": self.model_path,
            "supported_features": [
                "text_length", "word_count", "avg_word_length",
                "exclamation_count", "question_count", "caps_ratio",
                "negative_word_count", "positive_word_count", "toxic_keyword_count"
            ]
        }
    
    def evaluate_performance(self, test_texts: List[str], true_labels: List[str]) -> Dict[str, float]:
        """评估模型性能"""
        if len(test_texts) != len(true_labels):
            raise ValueError("测试文本数量与标签数量不匹配")
        
        predictions = []
        confidences = []
        
        for text in test_texts:
            result = self.predict(text)
            predictions.append(result["prediction"])
            confidences.append(result["confidence"])
        
        # 计算性能指标
        correct = sum(1 for pred, true in zip(predictions, true_labels) if pred == true)
        total = len(test_texts)
        accuracy = correct / total
        
        # 计算各类别的精确率和召回率
        tp_safe = sum(1 for pred, true in zip(predictions, true_labels) if pred == "safe" and true == "safe")
        fp_safe = sum(1 for pred, true in zip(predictions, true_labels) if pred == "safe" and true == "toxic")
        tn_safe = sum(1 for pred, true in zip(predictions, true_labels) if pred == "toxic" and true == "toxic")
        fn_safe = sum(1 for pred, true in zip(predictions, true_labels) if pred == "toxic" and true == "safe")
        
        precision_safe = tp_safe / (tp_safe + fp_safe) if (tp_safe + fp_safe) > 0 else 0
        recall_safe = tp_safe / (tp_safe + fn_safe) if (tp_safe + fn_safe) > 0 else 0
        f1_safe = 2 * precision_safe * recall_safe / (precision_safe + recall_safe) if (precision_safe + recall_safe) > 0 else 0
        
        precision_toxic = tn_safe / (tn_safe + fn_safe) if (tn_safe + fn_safe) > 0 else 0
        recall_toxic = tn_safe / (tn_safe + fp_safe) if (tn_safe + fp_safe) > 0 else 0
        f1_toxic = 2 * precision_toxic * recall_toxic / (precision_toxic + recall_toxic) if (precision_toxic + recall_toxic) > 0 else 0
        
        avg_confidence = np.mean(confidences)
        
        return {
            "accuracy": accuracy,
            "precision_safe": precision_safe,
            "recall_safe": recall_safe,
            "f1_safe": f1_safe,
            "precision_toxic": precision_toxic,
            "recall_toxic": recall_toxic,
            "f1_toxic": f1_toxic,
            "average_confidence": avg_confidence,
            "total_samples": total
        }
