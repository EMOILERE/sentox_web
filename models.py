from datetime import datetime
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
from extensions import db
import json

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    role = db.Column(db.String(20), default='user')  # user, admin, moderator
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    is_active = db.Column(db.Boolean, default=True)
    
    def set_password(self, password):
        self.password_hash = generate_password_hash(password)
    
    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

class ContentSubmission(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    content = db.Column(db.Text, nullable=False)
    content_type = db.Column(db.String(20), default='text')  # text, image, video
    platform = db.Column(db.String(50))  # weibo, douyin, wechat等
    submitted_by = db.Column(db.Integer, db.ForeignKey('user.id'))
    submitted_at = db.Column(db.DateTime, default=datetime.utcnow)
    status = db.Column(db.String(20), default='pending')  # pending, processing, completed
    
    # 关联的审核记录
    moderation_records = db.relationship('ModerationRecord', backref='submission', lazy=True)

class ModerationRecord(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    submission_id = db.Column(db.Integer, db.ForeignKey('content_submission.id'), nullable=False)
    
    # 初步分类结果 (SenTox-GLDA)
    initial_classification = db.Column(db.String(20))  # safe, toxic
    initial_confidence = db.Column(db.Float)
    
    # 多智能体系统结果
    final_decision = db.Column(db.String(20))  # approved, rejected, needs_review
    final_confidence = db.Column(db.Float)
    reasoning_chain = db.Column(db.Text)  # JSON格式存储推理链
    
    # 审核详情
    toxicity_categories = db.Column(db.Text)  # JSON: 仇恨言论、暴力、色情等
    severity_level = db.Column(db.Integer)  # 1-5级严重程度
    
    # 智能体协作记录
    agent_decisions = db.Column(db.Text)  # JSON格式存储各智能体决策
    coordination_log = db.Column(db.Text)  # 协调过程日志
    
    # 时间戳
    started_at = db.Column(db.DateTime, default=datetime.utcnow)
    completed_at = db.Column(db.DateTime)
    processing_time = db.Column(db.Float)  # 处理时间（秒）
    
    def set_reasoning_chain(self, chain):
        self.reasoning_chain = json.dumps(chain, ensure_ascii=False)
    
    def get_reasoning_chain(self):
        return json.loads(self.reasoning_chain) if self.reasoning_chain else []
    
    def set_agent_decisions(self, decisions):
        self.agent_decisions = json.dumps(decisions, ensure_ascii=False)
    
    def get_agent_decisions(self):
        return json.loads(self.agent_decisions) if self.agent_decisions else {}
    
    def set_toxicity_categories(self, categories):
        self.toxicity_categories = json.dumps(categories, ensure_ascii=False)
    
    def get_toxicity_categories(self):
        return json.loads(self.toxicity_categories) if self.toxicity_categories else []

class AgentPerformance(db.Model):
    """记录各智能体的表现统计"""
    id = db.Column(db.Integer, primary_key=True)
    agent_type = db.Column(db.String(50), nullable=False)  # classifier, reasoner, coordinator等
    agent_name = db.Column(db.String(100), nullable=False)
    
    # 性能指标
    total_decisions = db.Column(db.Integer, default=0)
    correct_decisions = db.Column(db.Integer, default=0)
    average_confidence = db.Column(db.Float, default=0.0)
    average_processing_time = db.Column(db.Float, default=0.0)
    
    # 专项统计
    toxicity_detection_accuracy = db.Column(db.Float, default=0.0)
    false_positive_rate = db.Column(db.Float, default=0.0)
    false_negative_rate = db.Column(db.Float, default=0.0)
    
    last_updated = db.Column(db.DateTime, default=datetime.utcnow)

class SystemMetrics(db.Model):
    """系统整体性能指标"""
    id = db.Column(db.Integer, primary_key=True)
    metric_date = db.Column(db.Date, default=datetime.utcnow().date())
    
    # 处理量统计
    total_submissions = db.Column(db.Integer, default=0)
    processed_submissions = db.Column(db.Integer, default=0)
    approved_submissions = db.Column(db.Integer, default=0)
    rejected_submissions = db.Column(db.Integer, default=0)
    
    # 性能指标
    average_processing_time = db.Column(db.Float, default=0.0)
    system_accuracy = db.Column(db.Float, default=0.0)
    agent_consensus_rate = db.Column(db.Float, default=0.0)
    
    # 错误统计
    system_errors = db.Column(db.Integer, default=0)
    timeout_errors = db.Column(db.Integer, default=0)
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class Platform(db.Model):
    """接入的社交平台配置"""
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(50), unique=True, nullable=False)
    display_name = db.Column(db.String(100), nullable=False)
    api_endpoint = db.Column(db.String(255))
    api_key = db.Column(db.String(255))
    webhook_secret = db.Column(db.String(255))
    is_active = db.Column(db.Boolean, default=True)
    
    # 平台特定配置
    platform_config = db.Column(db.Text)  # JSON格式存储平台特定配置
    moderation_rules = db.Column(db.Text)  # JSON格式存储审核规则
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def set_platform_config(self, config):
        self.platform_config = json.dumps(config, ensure_ascii=False)
    
    def get_platform_config(self):
        return json.loads(self.platform_config) if self.platform_config else {}
    
    def set_moderation_rules(self, rules):
        self.moderation_rules = json.dumps(rules, ensure_ascii=False)
    
    def get_moderation_rules(self):
        return json.loads(self.moderation_rules) if self.moderation_rules else {}
