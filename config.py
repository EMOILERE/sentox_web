import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'sentox-web-secret-key-2025'
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or 'sqlite:///sentox_web.db'
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # Alibaba Cloud Bailian LLM configuration
    DASHSCOPE_API_KEY = 'sk-8b654ec58c4c49f6a30cfb3d555a95d0'
    
    # Redis configuration (for task queue)
    REDIS_URL = os.environ.get('REDIS_URL') or 'redis://localhost:6379/0'
    
    # Celery configuration
    CELERY_BROKER_URL = REDIS_URL
    CELERY_RESULT_BACKEND = REDIS_URL
    
    # File upload configuration
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB
    UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
    
    # Multi-agent system configuration
    AGENT_SYSTEM_CONFIG = {
        'max_agents': 5,
        'coordination_timeout': 30,
        'reasoning_depth': 3,
        'consensus_threshold': 0.7
    }
    
    # SenTox-GLDA model configuration
    SENTOX_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models2', 'sentox_glda')
    
    # Content moderation configuration
    MODERATION_CONFIG = {
        'enable_realtime': True,
        'batch_size': 100,
        'confidence_threshold': 0.8,
        'escalation_threshold': 0.95
    }
