#!/usr/bin/env python3
"""
SenTox AI审核平台启动脚本
"""

import os
import sys
from pathlib import Path

def check_dependencies():
    """检查依赖是否已安装"""
    required_packages = [
        'flask',
        'flask_sqlalchemy', 
        'flask_migrate',
        'flask_login',
        'dashscope',
        'jieba',
        'requests'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ 缺少以下依赖包:")
        for package in missing_packages:
            print(f"   - {package}")
        
        print("\n请运行以下命令安装依赖:")
        print("pip install -r requirements.txt")
        print("\n或者逐个安装:")
        print("pip install " + " ".join(missing_packages))
        return False
    
    return True

def setup_environment():
    """设置环境变量"""
    if not os.path.exists('.env'):
        print("⚠️ 未找到 .env 文件")
        print("请创建 .env 文件并配置必要的环境变量，参考 .env.example")
        
        # 创建基本的环境配置
        env_content = """# SenTox AI审核平台基本配置
SECRET_KEY=sentox-web-secret-key-2025
DASHSCOPE_API_KEY=sk-8b654ec58c4c49f6a30cfb3d555a95d0
FLASK_ENV=development
FLASK_DEBUG=True
"""
        with open('.env', 'w', encoding='utf-8') as f:
            f.write(env_content)
        print("✅ 已创建基本的 .env 配置文件")

def initialize_database():
    """初始化数据库"""
    if not os.path.exists('sentox_web.db'):
        print("🔧 正在初始化数据库...")
        try:
            from init_db import init_database, create_sample_data, show_api_keys
            init_database()
            create_sample_data()
            show_api_keys()
            print("✅ 数据库初始化完成")
        except Exception as e:
            print(f"❌ 数据库初始化失败: {e}")
            return False
    else:
        print("ℹ️ 数据库已存在，跳过初始化")
    
    return True

def start_application():
    """启动应用"""
    try:
        print("🚀 正在启动 SenTox AI审核平台...")
        print("访问地址: http://localhost:5000")
        print("管理后台: http://localhost:5000/admin")
        print("默认管理员账户: admin / admin123")
        print("-" * 50)
        
        from app import create_app
        app = create_app()
        app.run(host='0.0.0.0', port=5000, debug=True)
        
    except KeyboardInterrupt:
        print("\n👋 应用已停止")
    except Exception as e:
        print(f"❌ 应用启动失败: {e}")

def main():
    """主函数"""
    print("=" * 50)
    print("🛡️  SenTox AI 多智能体内容审核平台")
    print("=" * 50)
    
    # 检查依赖
    if not check_dependencies():
        sys.exit(1)
    
    print("✅ 依赖检查完成")
    
    # 设置环境
    setup_environment()
    
    # 初始化数据库
    if not initialize_database():
        sys.exit(1)
    
    # 启动应用
    start_application()

if __name__ == '__main__':
    main()
