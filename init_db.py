#!/usr/bin/env python3
"""
数据库初始化脚本
用于创建数据库表和初始数据
"""

import os
import sys
from datetime import datetime

# 添加项目根目录到 Python 路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app import create_app
from extensions import db
from models import User, Platform, SystemMetrics
from werkzeug.security import generate_password_hash

def init_database():
    """初始化数据库"""
    app = create_app()
    
    with app.app_context():
        print("正在创建数据库表...")
        
        # 删除所有表（谨慎使用）
        db.drop_all()
        
        # 创建所有表
        db.create_all()
        
        print("数据库表创建完成！")
        
        # 创建管理员用户
        create_admin_user()
        
        # 创建测试平台
        create_test_platforms()
        
        # 初始化系统指标
        init_system_metrics()
        
        print("数据库初始化完成！")

def create_admin_user():
    """创建管理员用户"""
    print("创建管理员用户...")
    
    # 检查是否已存在管理员
    admin = User.query.filter_by(username='admin').first()
    if admin:
        print("管理员用户已存在")
        return
    
    # 创建管理员用户
    admin_user = User(
        username='admin',
        email='admin@sentox.ai',
        role='admin'
    )
    admin_user.set_password('admin123')  # 默认密码，生产环境中应该修改
    
    db.session.add(admin_user)
    
    # 创建测试用户
    test_user = User(
        username='testuser',
        email='test@sentox.ai',
        role='user'
    )
    test_user.set_password('test123')
    
    db.session.add(test_user)
    
    # 创建审核员用户
    moderator = User(
        username='moderator',
        email='moderator@sentox.ai',
        role='moderator'
    )
    moderator.set_password('mod123')
    
    db.session.add(moderator)
    
    db.session.commit()
    print("用户创建完成：admin (密码: admin123), testuser (密码: test123), moderator (密码: mod123)")

def create_test_platforms():
    """创建测试平台"""
    print("创建测试平台配置...")
    
    platforms = [
        {
            'name': 'weibo',
            'display_name': '微博',
            'api_endpoint': 'https://api.weibo.com/2/comments/show.json',
            'platform_config': {
                'rate_limit': 1000,
                'supported_features': ['text', 'image'],
                'moderation_rules': {
                    'auto_approve_threshold': 0.9,
                    'auto_reject_threshold': 0.7,
                    'escalation_threshold': 0.5
                }
            }
        },
        {
            'name': 'douyin',
            'display_name': '抖音',
            'api_endpoint': 'https://open.douyin.com/api/comment/list/',
            'platform_config': {
                'rate_limit': 500,
                'supported_features': ['text', 'video'],
                'moderation_rules': {
                    'auto_approve_threshold': 0.85,
                    'auto_reject_threshold': 0.75,
                    'escalation_threshold': 0.6
                }
            }
        },
        {
            'name': 'wechat',
            'display_name': '微信',
            'api_endpoint': 'https://api.weixin.qq.com/cgi-bin/message/custom/send',
            'platform_config': {
                'rate_limit': 2000,
                'supported_features': ['text'],
                'moderation_rules': {
                    'auto_approve_threshold': 0.95,
                    'auto_reject_threshold': 0.8,
                    'escalation_threshold': 0.4
                }
            }
        },
        {
            'name': 'zhihu',
            'display_name': '知乎',
            'api_endpoint': 'https://www.zhihu.com/api/v4/comments',
            'platform_config': {
                'rate_limit': 800,
                'supported_features': ['text', 'image'],
                'moderation_rules': {
                    'auto_approve_threshold': 0.88,
                    'auto_reject_threshold': 0.72,
                    'escalation_threshold': 0.55
                }
            }
        },
        {
            'name': 'test_platform',
            'display_name': '测试平台',
            'api_endpoint': 'http://localhost:5000/test/webhook',
            'platform_config': {
                'rate_limit': 10000,
                'supported_features': ['text', 'image', 'video'],
                'moderation_rules': {
                    'auto_approve_threshold': 0.8,
                    'auto_reject_threshold': 0.6,
                    'escalation_threshold': 0.5
                }
            }
        }
    ]
    
    for platform_data in platforms:
        # 检查是否已存在
        existing = Platform.query.filter_by(name=platform_data['name']).first()
        if existing:
            print(f"平台 {platform_data['display_name']} 已存在")
            continue
        
        # 生成API密钥
        import secrets
        api_key = f"sk-{secrets.token_urlsafe(32)}"
        webhook_secret = secrets.token_urlsafe(16)
        
        platform = Platform(
            name=platform_data['name'],
            display_name=platform_data['display_name'],
            api_endpoint=platform_data['api_endpoint'],
            api_key=api_key,
            webhook_secret=webhook_secret,
            is_active=True
        )
        platform.set_platform_config(platform_data['platform_config'])
        
        db.session.add(platform)
        
        print(f"创建平台: {platform_data['display_name']} (API Key: {api_key})")
    
    db.session.commit()
    print("测试平台创建完成！")

def init_system_metrics():
    """初始化系统指标"""
    print("初始化系统指标...")
    
    # 创建今天的系统指标记录
    today_metrics = SystemMetrics.query.filter_by(
        metric_date=datetime.utcnow().date()
    ).first()
    
    if not today_metrics:
        metrics = SystemMetrics(
            metric_date=datetime.utcnow().date(),
            total_submissions=0,
            processed_submissions=0,
            approved_submissions=0,
            rejected_submissions=0,
            average_processing_time=0.0,
            system_accuracy=0.0,
            agent_consensus_rate=0.0
        )
        db.session.add(metrics)
        db.session.commit()
        print("系统指标初始化完成")
    else:
        print("今日系统指标已存在")

def show_api_keys():
    """显示所有平台的API密钥"""
    print("\n" + "="*60)
    print("平台API密钥信息")
    print("="*60)
    
    app = create_app()
    with app.app_context():
        platforms = Platform.query.all()
        for platform in platforms:
            print(f"平台: {platform.display_name}")
            print(f"名称: {platform.name}")
            print(f"API Key: {platform.api_key}")
            print(f"Webhook Secret: {platform.webhook_secret}")
            print("-" * 60)
    
    print("\n使用示例:")
    print("curl -X POST http://localhost:5000/api/moderate \\")
    print("     -H 'Content-Type: application/json' \\")
    print(f"     -H 'X-API-Key: {platforms[0].api_key if platforms else 'YOUR_API_KEY'}' \\")
    print("     -d '{\"content\": \"这是一段测试内容\"}'")

def create_sample_data():
    """创建示例数据"""
    print("创建示例数据...")
    
    from models import ContentSubmission, ModerationRecord
    import json
    
    sample_contents = [
        {
            'content': '这个产品质量很好，服务态度也不错，推荐购买！',
            'platform': 'test_platform',
            'decision': 'approved',
            'confidence': 0.92
        },
        {
            'content': '垃圾产品，完全是骗钱的，卖家态度恶劣！',
            'platform': 'test_platform',
            'decision': 'rejected',
            'confidence': 0.87
        },
        {
            'content': '这个设计很有意思，虽然可能不是所有人都喜欢。',
            'platform': 'test_platform',
            'decision': 'approved',
            'confidence': 0.75
        },
        {
            'content': '价格有点贵，不过质量确实不错，值得考虑。',
            'platform': 'test_platform',
            'decision': 'approved',
            'confidence': 0.68
        }
    ]
    
    for sample in sample_contents:
        # 创建内容提交
        submission = ContentSubmission(
            content=sample['content'],
            platform=sample['platform'],
            status='completed'
        )
        db.session.add(submission)
        db.session.flush()  # 获取submission.id
        
        # 创建审核记录
        record = ModerationRecord(
            submission_id=submission.id,
            final_decision=sample['decision'],
            final_confidence=sample['confidence'],
            reasoning_chain=f"基于多智能体协作分析，最终决策为: {sample['decision']}",
            processing_time=2.5,
            completed_at=datetime.utcnow()
        )
        
        # 模拟智能体决策
        agent_decisions = {
            'classifier': {
                'decision': 'safe' if sample['decision'] == 'approved' else 'toxic',
                'confidence': sample['confidence'],
                'reasoning': 'SenTox-GLDA模型分析结果'
            },
            'reasoner': {
                'decision': 'safe' if sample['decision'] == 'approved' else 'risky',
                'confidence': sample['confidence'] * 0.95,
                'reasoning': '推理智能体深度分析'
            },
            'coordinator': {
                'decision': sample['decision'],
                'confidence': sample['confidence'],
                'reasoning': '协调智能体综合决策'
            }
        }
        record.set_agent_decisions(agent_decisions)
        
        db.session.add(record)
    
    db.session.commit()
    print(f"创建了 {len(sample_contents)} 条示例数据")

if __name__ == '__main__':
    print("SenTox AI审核平台 - 数据库初始化")
    print("="*50)
    
    # 检查是否强制重新初始化
    if len(sys.argv) > 1 and sys.argv[1] == '--force':
        print("强制重新初始化模式")
        init_database()
    else:
        # 检查数据库是否已初始化
        try:
            app = create_app()
            with app.app_context():
                # 尝试查询用户表
                User.query.first()
                print("数据库已初始化")
                
                # 询问是否重新初始化
                choice = input("是否重新初始化数据库？(y/N): ").strip().lower()
                if choice in ['y', 'yes']:
                    init_database()
                else:
                    print("跳过数据库初始化")
                    
                # 显示API密钥信息
                show_choice = input("是否显示API密钥信息？(y/N): ").strip().lower()
                if show_choice in ['y', 'yes']:
                    print("API密钥信息已在初始化过程中显示，请查看上面的输出。")
                
                # 创建示例数据
                sample_choice = input("是否创建示例数据？(y/N): ").strip().lower()
                if sample_choice in ['y', 'yes']:
                    create_sample_data()
                    
        except Exception as e:
            print(f"数据库未初始化或出现错误: {e}")
            print("开始初始化数据库...")
            init_database()
            # show_api_keys() # 暂时注释掉，避免应用上下文问题
            create_sample_data()
    
    print("\n初始化完成！")
    print("启动命令: python app.py")
    print("访问地址: http://localhost:5000")
    print("管理后台: http://localhost:5000/admin (admin/admin123)")
