from flask import Blueprint, render_template, request, jsonify, redirect, url_for, flash, session
from flask_login import login_user, logout_user, login_required, current_user
from werkzeug.security import check_password_hash, generate_password_hash
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models import User, ContentSubmission, ModerationRecord
from extensions import db
from agents.multi_agent_system import MultiAgentSystem
from models2.sentox_glda import SenToxGLDA
from config import Config
import logging
import asyncio

logger = logging.getLogger(__name__)

main_bp = Blueprint('main', __name__)

# Global variables: Multi-agent system and SenTox model
multi_agent_system = None
sentox_model = None

def init_systems():
    """Initialize multi-agent system and SenTox model"""
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
        logger.info("Multi-agent system initialization completed")
    
    if sentox_model is None:
        sentox_model = SenToxGLDA(Config.SENTOX_MODEL_PATH)
        logger.info("SenTox-GLDA model initialization completed")

@main_bp.route('/')
def index():
    """Homepage - Display platform overview and feature introduction"""
    init_systems()
    
    # Get system statistics
    total_submissions = ContentSubmission.query.count()
    total_moderated = ModerationRecord.query.count()
    
    # Get recent moderation activities
    recent_records = ModerationRecord.query.order_by(ModerationRecord.started_at.desc()).limit(5).all()
    
    return render_template('index.html', 
                         total_submissions=total_submissions,
                         total_moderated=total_moderated,
                         recent_records=recent_records)

@main_bp.route('/submit', methods=['GET', 'POST'])
def submit_content():
    """Content submission page"""
    if request.method == 'POST':
        content = request.form.get('content', '').strip()
        platform = request.form.get('platform', 'web')
        
        if not content:
            flash('Please enter content to be moderated', 'error')
            return redirect(url_for('main.submit_content'))
        
        if len(content) > 5000:
            flash('Content length cannot exceed 5000 characters', 'error')
            return redirect(url_for('main.submit_content'))
        
        try:
            # Create content submission record
            submission = ContentSubmission(
                content=content,
                platform=platform,
                submitted_by=current_user.id if current_user.is_authenticated else None,
                status='processing'
            )
            db.session.add(submission)
            db.session.commit()
            
            # Process content moderation asynchronously
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            init_systems()
            result = loop.run_until_complete(
                multi_agent_system.process_content(content, platform)
            )
            
            # Save moderation results
            moderation_record = ModerationRecord(
                submission_id=submission.id,
                final_decision=result['final_decision'],
                final_confidence=result['confidence'],
                reasoning_chain=result['reasoning'],
                agent_decisions=str(result['agent_decisions']),
                processing_time=result['processing_time']
            )
            moderation_record.completed_at = moderation_record.started_at
            
            db.session.add(moderation_record)
            submission.status = 'completed'
            db.session.commit()
            
            flash('Content moderation completed!', 'success')
            return redirect(url_for('main.view_result', submission_id=submission.id))
            
        except Exception as e:
            logger.error(f"Content processing failed: {str(e)}")
            flash(f'Processing failed: {str(e)}', 'error')
            return redirect(url_for('main.submit_content'))
    
    return render_template('submit.html')

@main_bp.route('/result/<int:submission_id>')
def view_result(submission_id):
    """View moderation results"""
    submission = ContentSubmission.query.get_or_404(submission_id)
    moderation_record = ModerationRecord.query.filter_by(submission_id=submission_id).first()
    
    if not moderation_record:
        flash('Moderation result does not exist', 'error')
        return redirect(url_for('main.index'))
    
    # Parse agent decision data
    agent_decisions = {}
    reasoning_chain = []
    
    try:
        import json
        if moderation_record.agent_decisions:
            agent_decisions = eval(moderation_record.agent_decisions)  # Note: should use json.loads in production
        
        if moderation_record.reasoning_chain:
            reasoning_chain = json.loads(moderation_record.reasoning_chain)
    except:
        pass
    
    return render_template('result.html', 
                         submission=submission,
                         record=moderation_record,
                         agent_decisions=agent_decisions,
                         reasoning_chain=reasoning_chain)

@main_bp.route('/batch')
@login_required
def batch_moderation():
    """Batch moderation page"""
    return render_template('batch.html')

@main_bp.route('/realtime')
@login_required
def realtime_monitor():
    """Real-time monitoring page"""
    init_systems()
    
    # Get system status
    system_status = multi_agent_system.get_system_status()
    
    # Get recent processing history
    recent_history = multi_agent_system.get_recent_processing_history(20)
    
    return render_template('realtime.html', 
                         system_status=system_status,
                         recent_history=recent_history)

@main_bp.route('/analytics')
@login_required
def analytics():
    """Data analytics page"""
    # Get statistical data
    stats = {
        'total_submissions': ContentSubmission.query.count(),
        'total_approved': ModerationRecord.query.filter_by(final_decision='approved').count(),
        'total_rejected': ModerationRecord.query.filter_by(final_decision='rejected').count(),
        'total_escalated': ModerationRecord.query.filter_by(final_decision='escalated').count(),
    }
    
    # Get platform distribution
    platform_stats = db.session.query(
        ContentSubmission.platform,
        db.func.count(ContentSubmission.id)
    ).group_by(ContentSubmission.platform).all()
    
    # Get moderation trends for the last 7 days
    from datetime import datetime, timedelta
    seven_days_ago = datetime.utcnow() - timedelta(days=7)
    
    daily_stats = db.session.query(
        db.func.date(ModerationRecord.started_at),
        db.func.count(ModerationRecord.id)
    ).filter(ModerationRecord.started_at >= seven_days_ago).group_by(
        db.func.date(ModerationRecord.started_at)
    ).all()
    
    return render_template('analytics.html', 
                         stats=stats,
                         platform_stats=platform_stats,
                         daily_stats=daily_stats)

@main_bp.route('/login', methods=['GET', 'POST'])
def login():
    """Login page"""
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        if not username or not password:
            flash('Please enter username and password', 'error')
            return render_template('login.html')
        
        user = User.query.filter_by(username=username).first()
        
        if user and user.check_password(password):
            login_user(user)
            flash('Login successful', 'success')
            next_page = request.args.get('next')
            return redirect(next_page or url_for('main.index'))
        else:
            flash('Incorrect username or password', 'error')
    
    return render_template('login.html')

@main_bp.route('/register', methods=['GET', 'POST'])
def register():
    """Registration page"""
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        
        # Validate input
        if not all([username, email, password, confirm_password]):
            flash('All fields are required', 'error')
            return render_template('register.html')
        
        if password != confirm_password:
            flash('Passwords do not match', 'error')
            return render_template('register.html')
        
        if len(password) < 6:
            flash('Password must be at least 6 characters long', 'error')
            return render_template('register.html')
        
        # Check if username and email already exist
        if User.query.filter_by(username=username).first():
            flash('Username already exists', 'error')
            return render_template('register.html')
        
        if User.query.filter_by(email=email).first():
            flash('Email already registered', 'error')
            return render_template('register.html')
        
        # Create new user
        try:
            user = User(username=username, email=email)
            user.set_password(password)
            db.session.add(user)
            db.session.commit()
            
            flash('Registration successful, please log in', 'success')
            return redirect(url_for('main.login'))
        except Exception as e:
            logger.error(f"User registration failed: {str(e)}")
            flash('Registration failed, please try again later', 'error')
    
    return render_template('register.html')

@main_bp.route('/logout')
@login_required
def logout():
    """Logout"""
    logout_user()
    flash('Logged out successfully', 'info')
    return redirect(url_for('main.index'))

@main_bp.route('/profile')
@login_required
def profile():
    """User profile page"""
    user_submissions = ContentSubmission.query.filter_by(
        submitted_by=current_user.id
    ).order_by(ContentSubmission.submitted_at.desc()).limit(10).all()
    
    return render_template('profile.html', submissions=user_submissions)


@main_bp.route('/client-hub')
def client_hub():
    """Client integration hub for content creators"""
    return render_template('client_hub.html')

@main_bp.route('/professional-dashboard')
@login_required
def professional_dashboard():
    """Professional analytics dashboard"""
    # Generate mock analytics data
    analytics_data = {
        'total_processed': 247891,
        'accuracy': 96.7,
        'response_time': 1.8,
        'active_creators': 15247,
        'platforms': [
            {'name': 'Twitter', 'volume': 89247, 'accuracy': 97.2},
            {'name': 'YouTube', 'volume': 156891, 'accuracy': 95.8},
            {'name': 'Instagram', 'volume': 67523, 'accuracy': 96.4},
            {'name': 'TikTok', 'volume': 42156, 'accuracy': 94.1}
        ]
    }
    
    return render_template('professional_dashboard.html', analytics=analytics_data)

@main_bp.route('/comment-environment-analysis')
@login_required  
def comment_environment_analysis():
    """Comment environment analysis for social media creators"""
    # Mock environment analysis data
    environment_data = {
        'overall_health_score': 87,
        'toxicity_level': 'Low',
        'engagement_quality': 'High',
        'sentiment_distribution': {
            'positive': 68.5,
            'neutral': 24.2,
            'negative': 7.3
        },
        'recommendations': [
            'Continue fostering positive community engagement',
            'Monitor for potential brigading patterns',
            'Consider implementing comment rewards for constructive feedback'
        ]
    }
    
    return render_template('comment_environment_analysis.html', data=environment_data)

@main_bp.route('/smart-comment-filter')
@login_required
def smart_comment_filter():
    """Smart comment filtering interface for creators"""
    # Mock filtered comments data
    filtered_data = {
        'total_comments': 1247,
        'filtered_out': 89,
        'approved': 1158,
        'pending_review': 23,
        'filter_effectiveness': 94.2
    }
    
    return render_template('smart_comment_filter.html', data=filtered_data)

@main_bp.route('/user-behavior-profiling')
@login_required
def user_behavior_profiling():
    """User behavior profiling and risk assessment system"""
    return render_template('user_behavior_profiling.html')

@main_bp.route('/cyberbullying-prevention')
@login_required
def cyberbullying_prevention():
    """Anti-cyberbullying protection center"""
    return render_template('cyberbullying_prevention.html')

@main_bp.route('/creator-protection-center')
@login_required
def creator_protection_center():
    """Content creator protection hub"""
    return render_template('creator_protection_center.html')

@main_bp.route('/harassment-detection')
@login_required
def harassment_detection():
    """Coordinated harassment and brigading detection system"""
    return render_template('harassment_detection.html')

@main_bp.route('/mental-health-support')
def mental_health_support():
    """Mental health resources and crisis support center"""
    return render_template('mental_health_support.html')

@main_bp.route('/health')
def health_check():
    """System health check"""
    try:
        init_systems()
        
        # Check database connection
        db.session.execute('SELECT 1')
        
        # Check model status
        sentox_status = sentox_model.get_model_info()
        
        # Asynchronously check multi-agent system
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        agent_health = loop.run_until_complete(multi_agent_system.health_check())
        
        return jsonify({
            'status': 'healthy',
            'timestamp': '2024-01-01T00:00:00Z',
            'database': 'connected',
            'sentox_model': sentox_status,
            'multi_agent_system': agent_health
        })
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return jsonify({
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': '2024-01-01T00:00:00Z'
        }), 500
