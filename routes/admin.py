from flask import Blueprint, render_template, request, jsonify, redirect, url_for, flash
from flask_login import login_required, current_user
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models import User, ContentSubmission, ModerationRecord, Platform, AgentPerformance, SystemMetrics
from extensions import db
from agents.multi_agent_system import MultiAgentSystem
from models2.sentox_glda import SenToxGLDA
from config import Config
import logging
from datetime import datetime, timedelta
import json
from functools import wraps

logger = logging.getLogger(__name__)

admin_bp = Blueprint('admin', __name__)

def admin_required(f):
    """管理员权限验证装饰器"""
    @wraps(f)
    @login_required
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated or current_user.role != 'admin':
            flash('需要管理员权限', 'error')
            return redirect(url_for('main.index'))
        return f(*args, **kwargs)
    return decorated_function

@admin_bp.route('/')
@admin_required
def dashboard():
    """管理后台首页"""
    # 获取系统概览统计
    stats = {
        'total_users': User.query.count(),
        'total_submissions': ContentSubmission.query.count(),
        'total_moderated': ModerationRecord.query.count(),
        'total_platforms': Platform.query.filter_by(is_active=True).count(),
        'active_users_today': _get_active_users_today(),
        'submissions_today': _get_submissions_today(),
        'avg_processing_time': _get_avg_processing_time(),
        'system_accuracy': _get_system_accuracy()
    }
    
    # 获取最近7天的趋势数据
    trend_data = _get_trend_data(7)
    
    # 获取决策分布
    decision_distribution = _get_decision_distribution()
    
    # 获取平台活跃度
    platform_activity = _get_platform_activity()
    
    # 获取最近的系统事件
    recent_events = _get_recent_system_events()
    
    return render_template('admin/dashboard.html',
                         stats=stats,
                         trend_data=trend_data,
                         decision_distribution=decision_distribution,
                         platform_activity=platform_activity,
                         recent_events=recent_events)

@admin_bp.route('/users')
@admin_required
def manage_users():
    """用户管理"""
    page = request.args.get('page', 1, type=int)
    per_page = 20
    
    users = User.query.paginate(
        page=page, per_page=per_page, error_out=False
    )
    
    return render_template('admin/users.html', users=users)

@admin_bp.route('/users/<int:user_id>/toggle_status', methods=['POST'])
@admin_required
def toggle_user_status(user_id):
    """切换用户状态"""
    user = User.query.get_or_404(user_id)
    user.is_active = not user.is_active
    db.session.commit()
    
    status = "激活" if user.is_active else "禁用"
    flash(f'用户 {user.username} 已{status}', 'success')
    return redirect(url_for('admin.manage_users'))

@admin_bp.route('/users/<int:user_id>/change_role', methods=['POST'])
@admin_required
def change_user_role(user_id):
    """修改用户角色"""
    user = User.query.get_or_404(user_id)
    new_role = request.form.get('role')
    
    if new_role in ['user', 'admin', 'moderator']:
        user.role = new_role
        db.session.commit()
        flash(f'用户 {user.username} 的角色已更改为 {new_role}', 'success')
    else:
        flash('无效的角色类型', 'error')
    
    return redirect(url_for('admin.manage_users'))

@admin_bp.route('/platforms')
@admin_required
def manage_platforms():
    """平台管理"""
    platforms = Platform.query.all()
    return render_template('admin/platforms.html', platforms=platforms)

@admin_bp.route('/platforms/add', methods=['GET', 'POST'])
@admin_required
def add_platform():
    """添加新平台"""
    if request.method == 'POST':
        name = request.form.get('name')
        display_name = request.form.get('display_name')
        api_endpoint = request.form.get('api_endpoint')
        
        if not name or not display_name:
            flash('平台名称和显示名称不能为空', 'error')
            return render_template('admin/add_platform.html')
        
        # 检查平台名称是否已存在
        if Platform.query.filter_by(name=name).first():
            flash('平台名称已存在', 'error')
            return render_template('admin/add_platform.html')
        
        # 生成API密钥
        import secrets
        api_key = f"sk-{secrets.token_urlsafe(32)}"
        webhook_secret = secrets.token_urlsafe(16)
        
        platform = Platform(
            name=name,
            display_name=display_name,
            api_endpoint=api_endpoint,
            api_key=api_key,
            webhook_secret=webhook_secret
        )
        
        db.session.add(platform)
        db.session.commit()
        
        flash(f'平台 {display_name} 添加成功', 'success')
        return redirect(url_for('admin.manage_platforms'))
    
    return render_template('admin/add_platform.html')

@admin_bp.route('/platforms/<int:platform_id>/toggle_status', methods=['POST'])
@admin_required
def toggle_platform_status(platform_id):
    """切换平台状态"""
    platform = Platform.query.get_or_404(platform_id)
    platform.is_active = not platform.is_active
    db.session.commit()
    
    status = "激活" if platform.is_active else "禁用"
    flash(f'平台 {platform.display_name} 已{status}', 'success')
    return redirect(url_for('admin.manage_platforms'))

@admin_bp.route('/submissions')
@admin_required
def manage_submissions():
    """内容提交管理"""
    page = request.args.get('page', 1, type=int)
    status_filter = request.args.get('status')
    platform_filter = request.args.get('platform')
    per_page = 50
    
    query = ContentSubmission.query
    
    if status_filter:
        query = query.filter_by(status=status_filter)
    if platform_filter:
        query = query.filter_by(platform=platform_filter)
    
    submissions = query.order_by(ContentSubmission.submitted_at.desc()).paginate(
        page=page, per_page=per_page, error_out=False
    )
    
    # 获取过滤选项
    platforms = db.session.query(ContentSubmission.platform).distinct().all()
    platforms = [p[0] for p in platforms if p[0]]
    
    return render_template('admin/submissions.html', 
                         submissions=submissions,
                         platforms=platforms,
                         current_status=status_filter,
                         current_platform=platform_filter)

@admin_bp.route('/submissions/<int:submission_id>')
@admin_required
def view_submission_detail(submission_id):
    """查看提交详情"""
    submission = ContentSubmission.query.get_or_404(submission_id)
    moderation_record = ModerationRecord.query.filter_by(submission_id=submission_id).first()
    
    # 解析智能体决策数据
    agent_decisions = {}
    thought_chains = {}
    
    if moderation_record and moderation_record.agent_decisions:
        try:
            agent_decisions = json.loads(moderation_record.agent_decisions)
        except:
            pass
    
    return render_template('admin/submission_detail.html',
                         submission=submission,
                         record=moderation_record,
                         agent_decisions=agent_decisions,
                         thought_chains=thought_chains)

@admin_bp.route('/analytics')
@admin_required
def analytics():
    """数据分析页面"""
    # 获取时间范围参数
    days = request.args.get('days', 30, type=int)
    end_date = datetime.utcnow().date()
    start_date = end_date - timedelta(days=days-1)
    
    # 获取趋势数据
    daily_stats = _get_daily_stats(start_date, end_date)
    
    # 获取平台分析
    platform_analysis = _get_platform_analysis(start_date, end_date)
    
    # 获取决策准确性分析
    accuracy_analysis = _get_accuracy_analysis(start_date, end_date)
    
    # 获取性能分析
    performance_analysis = _get_performance_analysis(start_date, end_date)
    
    return render_template('admin/analytics.html',
                         daily_stats=daily_stats,
                         platform_analysis=platform_analysis,
                         accuracy_analysis=accuracy_analysis,
                         performance_analysis=performance_analysis,
                         days=days)

@admin_bp.route('/system')
@admin_required
def system_management():
    """系统管理"""
    # 获取系统状态
    try:
        # 初始化系统组件
        config = {
            'dashscope_api_key': Config.DASHSCOPE_API_KEY,
            'max_agents': Config.AGENT_SYSTEM_CONFIG['max_agents'],
            'coordination_timeout': Config.AGENT_SYSTEM_CONFIG['coordination_timeout'],
            'reasoning_depth': Config.AGENT_SYSTEM_CONFIG['reasoning_depth'],
            'consensus_threshold': Config.AGENT_SYSTEM_CONFIG['consensus_threshold']
        }
        multi_agent_system = MultiAgentSystem(config)
        sentox_model = SenToxGLDA(Config.SENTOX_MODEL_PATH)
        
        system_status = multi_agent_system.get_system_status()
        model_info = sentox_model.get_model_info()
        
        # 获取系统配置
        system_config = {
            'agent_system': Config.AGENT_SYSTEM_CONFIG,
            'moderation': Config.MODERATION_CONFIG,
            'database_url': Config.SQLALCHEMY_DATABASE_URI,
            'redis_url': Config.REDIS_URL
        }
        
    except Exception as e:
        logger.error(f"获取系统状态失败: {str(e)}")
        system_status = {'error': str(e)}
        model_info = {'error': str(e)}
        system_config = {}
    
    return render_template('admin/system.html',
                         system_status=system_status,
                         model_info=model_info,
                         system_config=system_config)

@admin_bp.route('/logs')
@admin_required
def view_logs():
    """查看系统日志"""
    page = request.args.get('page', 1, type=int)
    level = request.args.get('level', 'all')
    per_page = 100
    
    # 这里简化处理，实际应该从日志文件或日志数据库读取
    # 模拟日志数据
    logs = _get_system_logs(page, per_page, level)
    
    return render_template('admin/logs.html', logs=logs, current_level=level)

@admin_bp.route('/api/system_health')
@admin_required
def api_system_health():
    """系统健康状态API"""
    try:
        # 检查数据库连接
        db.session.execute('SELECT 1')
        db_status = 'healthy'
        
        # 检查模型状态
        sentox_model = SenToxGLDA(Config.SENTOX_MODEL_PATH)
        model_status = 'healthy' if sentox_model.is_loaded else 'error'
        
        # 检查多智能体系统
        config = {'dashscope_api_key': Config.DASHSCOPE_API_KEY}
        multi_agent_system = MultiAgentSystem(config)
        agent_status = 'healthy'
        
        overall_status = 'healthy'
        if any(status == 'error' for status in [db_status, model_status, agent_status]):
            overall_status = 'error'
        
        return jsonify({
            'overall_status': overall_status,
            'components': {
                'database': db_status,
                'sentox_model': model_status,
                'multi_agent_system': agent_status
            },
            'timestamp': datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            'overall_status': 'error',
            'error': str(e),
            'timestamp': datetime.utcnow().isoformat()
        }), 500

# 辅助函数
def _get_active_users_today():
    """获取今日活跃用户数"""
    today = datetime.utcnow().date()
    return User.query.filter(
        db.func.date(User.created_at) == today
    ).count()

def _get_submissions_today():
    """获取今日提交数"""
    today = datetime.utcnow().date()
    return ContentSubmission.query.filter(
        db.func.date(ContentSubmission.submitted_at) == today
    ).count()

def _get_avg_processing_time():
    """获取平均处理时间"""
    return db.session.query(
        db.func.avg(ModerationRecord.processing_time)
    ).scalar() or 0.0

def _get_system_accuracy():
    """获取系统准确率（模拟）"""
    return 0.892  # 模拟数据

def _get_trend_data(days):
    """获取趋势数据"""
    end_date = datetime.utcnow().date()
    start_date = end_date - timedelta(days=days-1)
    
    # 获取每日统计
    daily_data = []
    current_date = start_date
    
    while current_date <= end_date:
        submissions_count = ContentSubmission.query.filter(
            db.func.date(ContentSubmission.submitted_at) == current_date
        ).count()
        
        moderated_count = db.session.query(ModerationRecord).join(ContentSubmission).filter(
            db.func.date(ContentSubmission.submitted_at) == current_date
        ).count()
        
        daily_data.append({
            'date': current_date.isoformat(),
            'submissions': submissions_count,
            'moderated': moderated_count
        })
        
        current_date += timedelta(days=1)
    
    return daily_data

def _get_decision_distribution():
    """获取决策分布"""
    distribution = db.session.query(
        ModerationRecord.final_decision,
        db.func.count(ModerationRecord.id)
    ).group_by(ModerationRecord.final_decision).all()
    
    return [{'decision': d[0], 'count': d[1]} for d in distribution]

def _get_platform_activity():
    """获取平台活跃度"""
    activity = db.session.query(
        ContentSubmission.platform,
        db.func.count(ContentSubmission.id)
    ).group_by(ContentSubmission.platform).all()
    
    return [{'platform': p[0], 'count': p[1]} for p in activity]

def _get_recent_system_events():
    """获取最近系统事件"""
    # 这里返回模拟数据，实际应该从系统日志获取
    return [
        {
            'timestamp': '2024-01-01T12:00:00Z',
            'level': 'info',
            'message': '多智能体系统启动完成',
            'component': 'multi_agent_system'
        },
        {
            'timestamp': '2024-01-01T11:55:00Z',
            'level': 'info',
            'message': 'SenTox-GLDA模型加载成功',
            'component': 'sentox_model'
        }
    ]

def _get_daily_stats(start_date, end_date):
    """获取每日统计数据"""
    return _get_trend_data((end_date - start_date).days + 1)

def _get_platform_analysis(start_date, end_date):
    """获取平台分析数据"""
    return _get_platform_activity()  # 简化处理

def _get_accuracy_analysis(start_date, end_date):
    """获取准确性分析"""
    return {'overall_accuracy': 0.892, 'by_category': {}}  # 模拟数据

def _get_performance_analysis(start_date, end_date):
    """获取性能分析"""
    return {'avg_processing_time': 2.5, 'throughput': 1000}  # 模拟数据

def _get_system_logs(page, per_page, level):
    """获取系统日志（模拟）"""
    # 实际实现中应该从日志文件或日志系统读取
    return {
        'logs': [
            {
                'timestamp': '2024-01-01T12:00:00Z',
                'level': 'INFO',
                'message': '用户提交内容审核请求',
                'component': 'api'
            }
        ],
        'total': 1,
        'page': page,
        'per_page': per_page
    }
