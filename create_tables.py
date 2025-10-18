#!/usr/bin/env python3
"""
简单的数据库表创建脚本
不依赖Flask扩展，直接使用SQLite
"""

import sqlite3
import os
from datetime import datetime

def create_database_tables():
    """创建所有必要的数据库表"""
    
    # 确保instance目录存在
    if not os.path.exists('instance'):
        os.makedirs('instance')
    
    # 连接到数据库
    db_path = 'instance/sentox_web.db'
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    print(f"正在创建数据库表...")
    
    try:
        # 创建用户表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username VARCHAR(80) UNIQUE NOT NULL,
                email VARCHAR(120) UNIQUE NOT NULL,
                password_hash VARCHAR(255) NOT NULL,
                role VARCHAR(20) DEFAULT 'user',
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                is_active BOOLEAN DEFAULT 1
            )
        ''')
        print("✓ 用户表创建完成")
        
        # 创建内容提交表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS content_submission (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content TEXT NOT NULL,
                content_type VARCHAR(20) DEFAULT 'text',
                platform VARCHAR(50),
                submitted_by INTEGER,
                submitted_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                status VARCHAR(20) DEFAULT 'pending',
                FOREIGN KEY (submitted_by) REFERENCES user (id)
            )
        ''')
        print("✓ 内容提交表创建完成")
        
        # 创建审核记录表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS moderation_record (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                submission_id INTEGER NOT NULL,
                initial_classification VARCHAR(20),
                initial_confidence REAL,
                final_decision VARCHAR(20),
                final_confidence REAL,
                reasoning_chain TEXT,
                toxicity_categories TEXT,
                severity_level INTEGER,
                agent_decisions TEXT,
                coordination_log TEXT,
                started_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                completed_at DATETIME,
                processing_time REAL,
                FOREIGN KEY (submission_id) REFERENCES content_submission (id)
            )
        ''')
        print("✓ 审核记录表创建完成")
        
        # 创建智能体性能表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS agent_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                agent_type VARCHAR(50) NOT NULL,
                agent_name VARCHAR(100) NOT NULL,
                total_decisions INTEGER DEFAULT 0,
                correct_decisions INTEGER DEFAULT 0,
                average_confidence REAL DEFAULT 0.0,
                average_processing_time REAL DEFAULT 0.0,
                toxicity_detection_accuracy REAL DEFAULT 0.0,
                false_positive_rate REAL DEFAULT 0.0,
                false_negative_rate REAL DEFAULT 0.0,
                last_updated DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        print("✓ 智能体性能表创建完成")
        
        # 创建系统指标表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS system_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                metric_date DATE DEFAULT CURRENT_DATE,
                total_submissions INTEGER DEFAULT 0,
                processed_submissions INTEGER DEFAULT 0,
                approved_submissions INTEGER DEFAULT 0,
                rejected_submissions INTEGER DEFAULT 0,
                average_processing_time REAL DEFAULT 0.0,
                system_accuracy REAL DEFAULT 0.0,
                agent_consensus_rate REAL DEFAULT 0.0,
                system_errors INTEGER DEFAULT 0,
                timeout_errors INTEGER DEFAULT 0,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        print("✓ 系统指标表创建完成")
        
        # 创建平台表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS platform (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name VARCHAR(50) UNIQUE NOT NULL,
                display_name VARCHAR(100) NOT NULL,
                api_endpoint VARCHAR(255),
                api_key VARCHAR(255),
                webhook_secret VARCHAR(255),
                is_active BOOLEAN DEFAULT 1,
                platform_config TEXT,
                moderation_rules TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        print("✓ 平台表创建完成")
        
        # 提交事务
        conn.commit()
        print("\n🎉 所有数据库表创建成功！")
        
        # 插入一些初始数据
        create_initial_data(cursor, conn)
        
    except Exception as e:
        print(f"❌ 创建表时发生错误: {e}")
        conn.rollback()
    finally:
        conn.close()

def create_initial_data(cursor, conn):
    """创建初始数据"""
    print("\n正在创建初始数据...")
    
    try:
        # 检查是否已有管理员用户
        cursor.execute("SELECT COUNT(*) FROM user WHERE role = 'admin'")
        admin_count = cursor.fetchone()[0]
        
        if admin_count == 0:
            # 创建默认管理员用户（密码：admin123）
            from werkzeug.security import generate_password_hash
            password_hash = generate_password_hash('admin123')
            
            cursor.execute('''
                INSERT INTO user (username, email, password_hash, role, is_active)
                VALUES (?, ?, ?, ?, ?)
            ''', ('admin', 'admin@sentox.com', password_hash, 'admin', 1))
            print("✓ 默认管理员用户创建完成 (用户名: admin, 密码: admin123)")
        
        # 检查是否已有平台配置
        cursor.execute("SELECT COUNT(*) FROM platform")
        platform_count = cursor.fetchone()[0]
        
        if platform_count == 0:
            # 添加默认平台
            platforms = [
                ('weibo', '微博', 1),
                ('douyin', '抖音', 1),
                ('wechat', '微信', 1),
                ('zhihu', '知乎', 1),
                ('bilibili', 'B站', 1)
            ]
            
            for name, display_name, is_active in platforms:
                cursor.execute('''
                    INSERT INTO platform (name, display_name, is_active)
                    VALUES (?, ?, ?)
                ''', (name, display_name, is_active))
            
            print("✓ 默认平台配置创建完成")
        
        # 提交初始数据
        conn.commit()
        print("✓ 初始数据创建完成")
        
    except ImportError:
        print("⚠️ 无法导入werkzeug，跳过管理员用户创建")
        print("   请手动安装：pip install werkzeug")
    except Exception as e:
        print(f"⚠️ 创建初始数据时发生错误: {e}")

if __name__ == '__main__':
    create_database_tables()
    print("\n数据库初始化完成！现在可以启动应用了。")
