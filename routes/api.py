from flask import Blueprint, request, jsonify
from flask_login import login_required, current_user
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models import ContentSubmission, ModerationRecord, Platform, User
from extensions import db
from agents.multi_agent_system import MultiAgentSystem
from models2.sentox_glda import SenToxGLDA
from config import Config
import logging
import asyncio
import time
from datetime import datetime
import functools

logger = logging.getLogger(__name__)

api_bp = Blueprint('api', __name__)

# 全局变量
multi_agent_system = None
sentox_model = None

def init_systems():
    """初始化系统组件"""
    global multi_agent_system, sentox_model
    
    if multi_agent_system is None:
        config = {
            'dashscope_api_key': Config.DASHSCOPE_API_KEY,
            'max_agents': Config.AGENT_SYSTEM_CONFIG['max_agents'],
            'coordination_timeout': Config.AGENT_SYSTEM_CONFIG['coordination_timeout'],
            'reasoning_depth': Config.AGENT_SYSTEM_CONFIG['reasoning_depth'],
            'consensus_threshold': Config.AGENT_SYSTEM_CONFIG['consensus_threshold']
        }
        multi_agent_system = MultiAgentSystem(config)
    
    if sentox_model is None:
        sentox_model = SenToxGLDA(Config.SENTOX_MODEL_PATH)

def api_key_required(f):
    """API密钥验证装饰器"""
    @functools.wraps(f)
    def decorated_function(*args, **kwargs):
        api_key = request.headers.get('X-API-Key') or request.args.get('api_key')
        
        if not api_key:
            return jsonify({'error': '缺少API密钥', 'code': 'MISSING_API_KEY'}), 401
        
        # 验证API密钥（这里简化处理，实际应该从数据库验证）
        platform = Platform.query.filter_by(api_key=api_key, is_active=True).first()
        
        if not platform:
            return jsonify({'error': 'API密钥无效', 'code': 'INVALID_API_KEY'}), 401
        
        # 将平台信息添加到请求上下文
        request.platform = platform
        return f(*args, **kwargs)
    
    return decorated_function

def rate_limit_check():
    """简单的速率限制检查"""
    # 这里可以添加更复杂的速率限制逻辑
    return True

@api_bp.route('/moderate', methods=['POST'])
@api_key_required
def moderate_content():
    """内容审核API - 核心功能接口"""
    if not rate_limit_check():
        return jsonify({'error': '请求频率过高', 'code': 'RATE_LIMIT_EXCEEDED'}), 429
    
    data = request.get_json()
    if not data:
        return jsonify({'error': '请求体为空', 'code': 'EMPTY_REQUEST'}), 400
    
    content = data.get('content', '').strip()
    if not content:
        return jsonify({'error': '内容不能为空', 'code': 'EMPTY_CONTENT'}), 400
    
    if len(content) > 10000:
        return jsonify({'error': '内容长度不能超过10000字符', 'code': 'CONTENT_TOO_LONG'}), 400
    
    try:
        init_systems()
        
        platform_name = request.platform.name
        callback_url = data.get('callback_url')
        priority = data.get('priority', 'normal')  # low, normal, high
        
        # 创建内容提交记录
        submission = ContentSubmission(
            content=content,
            platform=platform_name,
            status='processing'
        )
        db.session.add(submission)
        db.session.commit()
        
        start_time = time.time()
        
        # 运行多智能体审核
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        result = loop.run_until_complete(
            multi_agent_system.process_content(content, platform_name)
        )
        
        processing_time = time.time() - start_time
        
        # 保存审核记录
        moderation_record = ModerationRecord(
            submission_id=submission.id,
            final_decision=result['final_decision'],
            final_confidence=result['confidence'],
            reasoning_chain=result['reasoning'],
            processing_time=processing_time
        )
        moderation_record.set_agent_decisions(result['agent_decisions'])
        moderation_record.completed_at = datetime.utcnow()
        
        db.session.add(moderation_record)
        submission.status = 'completed'
        db.session.commit()
        
        # 构建API响应
        api_response = {
            'request_id': submission.id,
            'status': 'completed',
            'result': {
                'decision': result['final_decision'],
                'confidence': result['confidence'],
                'reasoning': result['reasoning'],
                'categories': [],  # 可以从agent_decisions中提取
                'severity_level': _extract_severity_level(result),
            },
            'processing_time': processing_time,
            'timestamp': datetime.utcnow().isoformat(),
            'model_version': sentox_model.get_model_info()['version'],
            'agent_summary': {
                'total_agents': len(result.get('agent_decisions', {})),
                'consensus_level': _calculate_consensus_level(result),
                'thought_chain_length': len(result.get('thought_chains', {}))
            }
        }
        
        # 如果提供了回调URL，可以异步发送结果（这里简化处理）
        if callback_url:
            api_response['callback_scheduled'] = True
        
        return jsonify(api_response)
        
    except Exception as e:
        logger.error(f"API审核失败: {str(e)}")
        
        # 更新提交状态为错误
        if 'submission' in locals():
            submission.status = 'error'
            db.session.commit()
        
        return jsonify({
            'error': '内容审核失败',
            'code': 'MODERATION_FAILED',
            'details': str(e),
            'timestamp': datetime.utcnow().isoformat()
        }), 500

@api_bp.route('/batch_moderate', methods=['POST'])
@api_key_required
def batch_moderate():
    """批量内容审核API"""
    data = request.get_json()
    if not data or 'contents' not in data:
        return jsonify({'error': '请求格式错误', 'code': 'INVALID_FORMAT'}), 400
    
    contents = data['contents']
    if not isinstance(contents, list) or len(contents) == 0:
        return jsonify({'error': '内容列表不能为空', 'code': 'EMPTY_CONTENT_LIST'}), 400
    
    if len(contents) > 100:
        return jsonify({'error': '批量处理最多支持100条内容', 'code': 'BATCH_LIMIT_EXCEEDED'}), 400
    
    try:
        init_systems()
        
        results = []
        total_start_time = time.time()
        
        for i, content_item in enumerate(contents):
            if isinstance(content_item, str):
                content = content_item
                content_id = f"batch_{int(time.time())}_{i}"
            elif isinstance(content_item, dict):
                content = content_item.get('content', '')
                content_id = content_item.get('id', f"batch_{int(time.time())}_{i}")
            else:
                results.append({
                    'id': f"batch_{int(time.time())}_{i}",
                    'status': 'error',
                    'error': '内容格式错误'
                })
                continue
            
            if not content or len(content.strip()) == 0:
                results.append({
                    'id': content_id,
                    'status': 'error',
                    'error': '内容不能为空'
                })
                continue
            
            try:
                # 处理单个内容
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                result = loop.run_until_complete(
                    multi_agent_system.process_content(content, request.platform.name)
                )
                
                results.append({
                    'id': content_id,
                    'status': 'completed',
                    'decision': result['final_decision'],
                    'confidence': result['confidence'],
                    'reasoning': result['reasoning'][:200] + '...' if len(result['reasoning']) > 200 else result['reasoning'],
                    'processing_time': result['processing_time']
                })
                
            except Exception as e:
                logger.error(f"批量处理第{i}项失败: {str(e)}")
                results.append({
                    'id': content_id,
                    'status': 'error',
                    'error': str(e)
                })
        
        total_processing_time = time.time() - total_start_time
        
        # 统计结果
        completed_count = sum(1 for r in results if r['status'] == 'completed')
        error_count = sum(1 for r in results if r['status'] == 'error')
        
        response = {
            'batch_id': f"batch_{int(time.time())}",
            'total_items': len(contents),
            'completed': completed_count,
            'errors': error_count,
            'results': results,
            'total_processing_time': total_processing_time,
            'average_processing_time': total_processing_time / len(contents),
            'timestamp': datetime.utcnow().isoformat()
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"批量审核API失败: {str(e)}")
        return jsonify({
            'error': '批量审核失败',
            'code': 'BATCH_MODERATION_FAILED',
            'details': str(e)
        }), 500

@api_bp.route('/status/<int:request_id>')
@api_key_required
def get_moderation_status(request_id):
    """获取审核状态"""
    submission = ContentSubmission.query.get(request_id)
    if not submission:
        return jsonify({'error': '请求ID不存在', 'code': 'REQUEST_NOT_FOUND'}), 404
    
    moderation_record = ModerationRecord.query.filter_by(submission_id=request_id).first()
    
    response = {
        'request_id': request_id,
        'status': submission.status,
        'submitted_at': submission.submitted_at.isoformat(),
        'platform': submission.platform
    }
    
    if moderation_record:
        response.update({
            'result': {
                'decision': moderation_record.final_decision,
                'confidence': moderation_record.final_confidence,
                'reasoning': moderation_record.reasoning_chain,
            },
            'processing_time': moderation_record.processing_time,
            'completed_at': moderation_record.completed_at.isoformat() if moderation_record.completed_at else None
        })
    
    return jsonify(response)

@api_bp.route('/models/info')
@api_key_required
def get_models_info():
    """获取模型信息"""
    init_systems()
    
    sentox_info = sentox_model.get_model_info()
    system_status = multi_agent_system.get_system_status()
    
    return jsonify({
        'sentox_glda': sentox_info,
        'multi_agent_system': {
            'agents_count': system_status['agents_count'],
            'active_agents': system_status['active_agents'],
            'total_processed': system_status['total_processed'],
            'average_processing_time': system_status['average_processing_time']
        },
        'api_version': '1.0.0',
        'timestamp': datetime.utcnow().isoformat()
    })

@api_bp.route('/statistics')
@api_key_required
def get_statistics():
    """获取平台统计信息"""
    platform_name = request.platform.name
    
    # 获取该平台的统计数据
    total_submissions = ContentSubmission.query.filter_by(platform=platform_name).count()
    
    total_approved = db.session.query(ModerationRecord).join(ContentSubmission).filter(
        ContentSubmission.platform == platform_name,
        ModerationRecord.final_decision == 'approved'
    ).count()
    
    total_rejected = db.session.query(ModerationRecord).join(ContentSubmission).filter(
        ContentSubmission.platform == platform_name,
        ModerationRecord.final_decision == 'rejected'
    ).count()
    
    total_escalated = db.session.query(ModerationRecord).join(ContentSubmission).filter(
        ContentSubmission.platform == platform_name,
        ModerationRecord.final_decision == 'escalated'
    ).count()
    
    # 计算平均处理时间
    avg_processing_time = db.session.query(db.func.avg(ModerationRecord.processing_time)).join(ContentSubmission).filter(
        ContentSubmission.platform == platform_name
    ).scalar() or 0.0
    
    return jsonify({
        'platform': platform_name,
        'statistics': {
            'total_submissions': total_submissions,
            'total_approved': total_approved,
            'total_rejected': total_rejected,
            'total_escalated': total_escalated,
            'approval_rate': total_approved / total_submissions if total_submissions > 0 else 0,
            'average_processing_time': avg_processing_time
        },
        'timestamp': datetime.utcnow().isoformat()
    })

@api_bp.route('/webhooks/register', methods=['POST'])
@api_key_required
def register_webhook():
    """注册Webhook回调"""
    data = request.get_json()
    if not data or 'webhook_url' not in data:
        return jsonify({'error': '缺少webhook_url', 'code': 'MISSING_WEBHOOK_URL'}), 400
    
    webhook_url = data['webhook_url']
    events = data.get('events', ['moderation_completed'])
    
    # 更新平台的webhook配置
    platform = request.platform
    config = platform.get_platform_config()
    config.update({
        'webhook_url': webhook_url,
        'webhook_events': events,
        'webhook_registered_at': datetime.utcnow().isoformat()
    })
    platform.set_platform_config(config)
    db.session.commit()
    
    return jsonify({
        'message': 'Webhook注册成功',
        'webhook_url': webhook_url,
        'events': events,
        'timestamp': datetime.utcnow().isoformat()
    })

@api_bp.route('/agents/performance')
@api_key_required
def get_agents_performance():
    """获取智能体性能数据"""
    init_systems()
    
    performance_metrics = multi_agent_system._collect_performance_metrics()
    
    return jsonify({
        'agents_performance': performance_metrics,
        'system_summary': multi_agent_system.get_system_status(),
        'timestamp': datetime.utcnow().isoformat()
    })

@api_bp.errorhandler(404)
def api_not_found(error):
    """API 404处理"""
    return jsonify({
        'error': 'API端点不存在',
        'code': 'ENDPOINT_NOT_FOUND',
        'timestamp': datetime.utcnow().isoformat()
    }), 404

@api_bp.errorhandler(405)
def method_not_allowed(error):
    """API方法不允许处理"""
    return jsonify({
        'error': '请求方法不允许',
        'code': 'METHOD_NOT_ALLOWED',
        'allowed_methods': error.description.get('valid_methods', []),
        'timestamp': datetime.utcnow().isoformat()
    }), 405

def _extract_severity_level(result):
    """从结果中提取严重程度等级"""
    # 根据置信度和决策类型推断严重程度
    confidence = result['confidence']
    decision = result['final_decision']
    
    if decision == 'approved':
        return 1
    elif decision == 'rejected':
        if confidence > 0.9:
            return 5
        elif confidence > 0.7:
            return 4
        else:
            return 3
    else:  # escalated
        return 3

def _calculate_consensus_level(result):
    """计算智能体共识程度"""
    agent_decisions = result.get('agent_decisions', {})
    if len(agent_decisions) < 2:
        return 1.0
    
    # 简化的共识计算
    decisions = []
    for agent_data in agent_decisions.values():
        if isinstance(agent_data, dict) and 'decision' in agent_data:
            decisions.append(agent_data['decision'])
    
    if not decisions:
        return 0.5
    
    # 计算最多决策的占比
    from collections import Counter
    decision_counts = Counter(decisions)
    most_common_count = decision_counts.most_common(1)[0][1]
    
    return most_common_count / len(decisions)
