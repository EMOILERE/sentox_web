from flask import Flask
from config import Config
from extensions import db, migrate, login_manager
from routes import main_bp, api_bp, admin_bp

def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)
    
    # Initialize extensions
    db.init_app(app)
    migrate.init_app(app, db)
    login_manager.init_app(app)
    login_manager.login_view = 'main.login'
    login_manager.login_message = 'Please log in to access this page'
    
    # Configure user loader
    @login_manager.user_loader
    def load_user(user_id):
        # Import User model (import inside function to avoid circular imports)
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from models import User
        return User.query.get(int(user_id))
    
    # Register blueprints
    app.register_blueprint(main_bp)
    app.register_blueprint(api_bp, url_prefix='/api')
    app.register_blueprint(admin_bp, url_prefix='/admin')
    
    return app

if __name__ == '__main__':
    app = create_app()
    app.run(debug=True, host='0.0.0.0', port=5000)
