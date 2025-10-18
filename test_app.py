#!/usr/bin/env python3
"""
SenTox AI审核平台基础功能测试
"""

import unittest
import json
import asyncio
from app import create_app
from extensions import db
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from models import User, ContentSubmission, Platform
from agents.multi_agent_system import MultiAgentSystem
from models2.sentox_glda import SenToxGLDA


class SenToxTestCase(unittest.TestCase):
    """SenTox平台基础测试类"""
    
    def setUp(self):
        """测试前设置"""
        self.app = create_app()
        self.app.config['TESTING'] = True
        self.app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///:memory:'
        
        self.client = self.app.test_client()
        self.app_context = self.app.app_context()
        self.app_context.push()
        
        # 创建测试数据库
        db.create_all()
        self._create_test_data()
    
    def tearDown(self):
        """测试后清理"""
        db.session.remove()
        db.drop_all()
        self.app_context.pop()
    
    def _create_test_data(self):
        """创建测试数据"""
        # 创建测试用户
        test_user = User(username='testuser', email='test@example.com')
        test_user.set_password('testpass')
        db.session.add(test_user)
        
        # 创建管理员用户
        admin_user = User(username='admin', email='admin@example.com', role='admin')
        admin_user.set_password('adminpass')
        db.session.add(admin_user)
        
        # 创建测试平台
        test_platform = Platform(
            name='test_platform',
            display_name='测试平台',
            api_key='test-api-key-123',
            webhook_secret='test-webhook-secret',
            is_active=True
        )
        db.session.add(test_platform)
        
        db.session.commit()

class WebInterfaceTest(SenToxTestCase):
    """Web界面测试"""
    
    def test_home_page(self):
        """测试首页"""
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn('SenTox', response.get_data(as_text=True))
    
    def test_submit_page(self):
        """测试提交页面"""
        response = self.client.get('/submit')
        self.assertEqual(response.status_code, 200)
        self.assertIn('智能内容审核', response.get_data(as_text=True))
    
    def test_content_submission(self):
        """测试内容提交"""
        response = self.client.post('/submit', data={
            'content': '这是一个测试内容',
            'platform': 'test_platform'
        }, follow_redirects=True)
        
        # 应该重定向到结果页面
        self.assertEqual(response.status_code, 200)
        
        # 检查是否创建了提交记录
        submission = ContentSubmission.query.first()
        self.assertIsNotNone(submission)
        self.assertEqual(submission.content, '这是一个测试内容')
    
    def test_login(self):
        """测试登录功能"""
        response = self.client.post('/login', data={
            'username': 'testuser',
            'password': 'testpass'
        }, follow_redirects=True)
        
        self.assertEqual(response.status_code, 200)
        
        # 测试错误登录
        response = self.client.post('/login', data={
            'username': 'testuser',
            'password': 'wrongpass'
        })
        
        self.assertEqual(response.status_code, 200)
        self.assertIn('用户名或密码错误', response.get_data(as_text=True))

class APITest(SenToxTestCase):
    """API接口测试"""
    
    def test_moderate_api_without_key(self):
        """测试无API密钥的审核请求"""
        response = self.client.post('/api/moderate', 
                                   data=json.dumps({'content': '测试内容'}),
                                   content_type='application/json')
        
        self.assertEqual(response.status_code, 401)
        data = json.loads(response.data)
        self.assertEqual(data['code'], 'MISSING_API_KEY')
    
    def test_moderate_api_with_key(self):
        """测试带API密钥的审核请求"""
        headers = {
            'X-API-Key': 'test-api-key-123',
            'Content-Type': 'application/json'
        }
        
        response = self.client.post('/api/moderate', 
                                   data=json.dumps({'content': '这是正常的测试内容'}),
                                   headers=headers)
        
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('result', data)
        self.assertIn('decision', data['result'])
    
    def test_batch_moderate_api(self):
        """测试批量审核API"""
        headers = {
            'X-API-Key': 'test-api-key-123',
            'Content-Type': 'application/json'
        }
        
        response = self.client.post('/api/batch_moderate', 
                                   data=json.dumps({
                                       'contents': ['内容1', '内容2', '内容3']
                                   }),
                                   headers=headers)
        
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('results', data)
        self.assertEqual(len(data['results']), 3)
    
    def test_health_api(self):
        """测试系统健康检查API"""
        response = self.client.get('/health')
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.data)
        self.assertIn('status', data)

class AgentSystemTest(SenToxTestCase):
    """智能体系统测试"""
    
    def test_multi_agent_system_init(self):
        """测试多智能体系统初始化"""
        config = {
            'dashscope_api_key': 'test-key',
            'max_agents': 3,
            'coordination_timeout': 10,
            'reasoning_depth': 2,
            'consensus_threshold': 0.7
        }
        
        mas = MultiAgentSystem(config)
        self.assertIsNotNone(mas)
        self.assertEqual(len(mas.agents), 3)
        
    def test_system_status(self):
        """测试系统状态获取"""
        config = {
            'dashscope_api_key': 'test-key',
            'max_agents': 3
        }
        
        mas = MultiAgentSystem(config)
        status = mas.get_system_status()
        
        self.assertIn('agents_count', status)
        self.assertIn('total_processed', status)
        self.assertEqual(status['agents_count'], 3)

class SenToxModelTest(SenToxTestCase):
    """SenTox模型测试"""
    
    def test_model_init(self):
        """测试模型初始化"""
        model = SenToxGLDA()
        self.assertIsNotNone(model)
        self.assertTrue(model.is_loaded)
    
    def test_model_prediction(self):
        """测试模型预测"""
        model = SenToxGLDA()
        
        # 测试正常内容
        result = model.predict("这是一个很好的产品，推荐购买")
        self.assertIn('prediction', result)
        self.assertIn('confidence', result)
        self.assertIn(result['prediction'], ['safe', 'toxic'])
        
        # 测试有问题的内容
        result = model.predict("垃圾产品，完全是骗钱的")
        self.assertIn('prediction', result)
        self.assertIn('confidence', result)
    
    def test_batch_prediction(self):
        """测试批量预测"""
        model = SenToxGLDA()
        
        texts = [
            "这个产品很好",
            "服务态度不错",
            "垃圾东西"
        ]
        
        results = model.batch_predict(texts)
        self.assertEqual(len(results), 3)
        
        for result in results:
            self.assertIn('prediction', result)
            self.assertIn('confidence', result)

class PerformanceTest(SenToxTestCase):
    """性能测试"""
    
    def test_single_request_performance(self):
        """测试单个请求性能"""
        import time
        
        headers = {
            'X-API-Key': 'test-api-key-123',
            'Content-Type': 'application/json'
        }
        
        start_time = time.time()
        
        response = self.client.post('/api/moderate', 
                                   data=json.dumps({'content': '性能测试内容'}),
                                   headers=headers)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        self.assertEqual(response.status_code, 200)
        self.assertLess(processing_time, 10.0)  # 应该在10秒内完成
        
        data = json.loads(response.data)
        self.assertIn('processing_time', data)
    
    def test_concurrent_requests(self):
        """测试并发请求（简化版）"""
        import threading
        import time
        
        results = []
        
        def make_request():
            headers = {
                'X-API-Key': 'test-api-key-123',
                'Content-Type': 'application/json'
            }
            
            response = self.client.post('/api/moderate', 
                                       data=json.dumps({'content': f'并发测试内容{threading.current_thread().ident}'}),
                                       headers=headers)
            results.append(response.status_code)
        
        # 创建多个线程
        threads = []
        for i in range(5):
            thread = threading.Thread(target=make_request)
            threads.append(thread)
        
        # 启动所有线程
        start_time = time.time()
        for thread in threads:
            thread.start()
        
        # 等待所有线程完成
        for thread in threads:
            thread.join()
        
        end_time = time.time()
        
        # 检查结果
        self.assertEqual(len(results), 5)
        self.assertTrue(all(status == 200 for status in results))
        self.assertLess(end_time - start_time, 30.0)  # 总时间应在30秒内

def run_tests():
    """运行所有测试"""
    print("开始运行SenTox AI审核平台测试...")
    print("=" * 50)
    
    # 创建测试套件
    test_suite = unittest.TestSuite()
    
    # 添加测试类
    test_classes = [
        WebInterfaceTest,
        APITest,
        AgentSystemTest,
        SenToxModelTest,
        PerformanceTest
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # 输出结果
    print("\n" + "=" * 50)
    print("测试完成！")
    print(f"运行测试: {result.testsRun}")
    print(f"成功: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"失败: {len(result.failures)}")
    print(f"错误: {len(result.errors)}")
    
    if result.failures:
        print("\n失败的测试:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")
    
    if result.errors:
        print("\n错误的测试:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")
    
    return result.wasSuccessful()

if __name__ == '__main__':
    success = run_tests()
    exit(0 if success else 1)
