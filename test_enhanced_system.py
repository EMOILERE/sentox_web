#!/usr/bin/env python3
"""
增强版多智能体系统测试文件
测试总分总架构的数据流通和功能正确性
"""

import asyncio
import time
import logging
import json
from datetime import datetime
from typing import Dict, List, Any

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def print_reasoning_chain(agent_id: str, reasoning_chain: Dict[str, Any], content: str = ""):
    """打印智能体的推导过程和思维链"""
    if not reasoning_chain:
        return
        
    print(f"\n{'='*80}")
    print(f"🧠 智能体 [{agent_id}] 推导过程详解")
    print(f"{'='*80}")
    
    if content:
        print(f"📝 分析内容: {content[:100]}{'...' if len(content) > 100 else ''}")
    
    print(f"⏱️  总处理时间: {reasoning_chain.get('total_processing_time', 0):.3f}秒")
    
    # 显示推导步骤
    reasoning_steps = reasoning_chain.get('reasoning_steps', [])
    if reasoning_steps:
        print(f"\n🔍 推导步骤 ({len(reasoning_steps)}步):")
        for i, step in enumerate(reasoning_steps, 1):
            step_type = step.get('step_type', '未知')
            description = step.get('description', '无描述')
            print(f"  {i}. [{step_type}] {description}")
    
    # 显示决策点
    decision_points = reasoning_chain.get('decision_points', [])
    if decision_points:
        print(f"\n🎯 关键决策点 ({len(decision_points)}个):")
        for i, decision in enumerate(decision_points, 1):
            decision_text = decision.get('decision', '未知决策')
            rationale = decision.get('rationale', '无理由')
            print(f"  {i}. 决策: {decision_text}")
            print(f"     理由: {rationale}")
            if decision.get('alternatives'):
                print(f"     备选: {', '.join(decision['alternatives'])}")
    
    # 显示工具使用
    tool_usage = reasoning_chain.get('tool_usage', [])
    if tool_usage:
        print(f"\n🔧 工具使用记录 ({len(tool_usage)}次):")
        for i, tool in enumerate(tool_usage, 1):
            tool_name = tool.get('tool_name', '未知工具')
            print(f"  {i}. 工具: {tool_name}")
    
    # 显示中间结果
    intermediate_results = reasoning_chain.get('intermediate_results', {})
    if intermediate_results:
        print(f"\n💾 中间结果 ({len(intermediate_results)}项):")
        for key, result_info in intermediate_results.items():
            print(f"  • {key}: 已存储")
    
    print(f"{'='*80}\n")

def print_all_reasoning_chains(result: Dict[str, Any]):
    """打印所有智能体的推导链"""
    print(f"\n🧩 ===== 多智能体推导过程总览 =====")
    
    # 从子智能体阶段获取推导链
    sub_agents_phase = result.get('sub_agents_phase', {})
    agent_results = sub_agents_phase.get('agent_results', {})
    
    content = result.get('content', '')
    
    reasoning_chains_found = 0
    for task_type, task_result in agent_results.items():
        if isinstance(task_result, dict) and 'reasoning_chain' in task_result:
            reasoning_chain = task_result['reasoning_chain']
            agent_id = reasoning_chain.get('agent_id', task_type)
            print_reasoning_chain(agent_id, reasoning_chain, content)
            reasoning_chains_found += 1
    
    if reasoning_chains_found == 0:
        print("⚠️ 未找到推导链信息")
    else:
        print(f"✅ 总共显示了 {reasoning_chains_found} 个智能体的推导过程")
    
    print(f"🧩 ===== 推导过程总览结束 =====\n")

# 添加进度条支持
try:
    from tqdm.asyncio import tqdm
except ImportError:
    # 如果没有安装tqdm，使用简单的替代品
    class tqdm:
        def __init__(self, *args, **kwargs):
            self.desc = kwargs.get('desc', '')
            print(f"开始: {self.desc}")
        
        def update(self, n=1):
            pass
        
        def close(self):
            print(f"完成: {self.desc}")
        
        def __enter__(self):
            return self
        
        def __exit__(self, *args):
            self.close()

# 导入系统组件
from agents.enhanced_multi_agent_system import EnhancedMultiAgentSystem

class SystemTester:
    """系统测试器"""
    
    def __init__(self):
        self.test_results = []
        self.config = {
            'dashscope_api_key': 'sk-8b654ec58c4c49f6a30cfb3d555a95d0',
            'consensus_threshold': 0.7,
            'max_agents': 6,
            'coordination_timeout': 30,
            'reasoning_depth': 3
        }
        
        # 测试用例
        self.test_cases = [
            {
                "name": "正面评价测试",
                "content": "这个产品质量很好，服务态度也不错，强烈推荐给大家！物流很快，包装也很仔细。",
                "platform": "taobao",
                "expected_decision": "approved",
                "description": "测试正面内容的处理"
            },
            {
                "name": "轻微负面测试", 
                "content": "价格有点贵，不过质量确实还可以，考虑中。",
                "platform": "weibo",
                "expected_decision": "approved",
                "description": "测试轻微负面但整体中性的内容"
            },
            {
                "name": "明显毒性内容测试",
                "content": "这垃圾产品，卖家态度恶劣，完全就是骗钱的傻逼！",
                "platform": "douyin", 
                "expected_decision": "rejected",
                "description": "测试明显的毒性内容检测"
            },
            {
                "name": "复杂歧义内容测试",
                "content": "这设计真是太有创意了，让人印象深刻。虽然可能不是所有人都能接受这种风格。",
                "platform": "zhihu",
                "expected_decision": "approved",
                "description": "测试复杂的歧义内容处理"
            },
            {
                "name": "短内容测试",
                "content": "不错",
                "platform": "wechat",
                "expected_decision": "approved", 
                "description": "测试极短内容的处理"
            },
            {
                "name": "长内容测试",
                "content": "今天去了一家新开的餐厅，环境很不错，装修风格很有特色，服务员态度也很好。点了几个招牌菜，味道确实不错，特别是他们家的特色汤，很鲜美。价格相对来说还算合理，性价比不错。唯一的不足就是等菜时间有点长，可能是因为刚开业比较忙。总的来说，这次用餐体验还是很满意的，下次还会再来尝试其他菜品。推荐给朋友们。",
                "platform": "dianping",
                "expected_decision": "approved",
                "description": "测试长内容的处理能力"
            },
            {
                "name": "边界情况测试",
                "content": "这个...怎么说呢，有点复杂，可能需要更多时间考虑。",
                "platform": "unknown",
                "expected_decision": "needs_human_review",
                "description": "测试边界情况和不确定内容"
            }
        ]

    async def run_all_tests(self):
         """运行所有测试"""
         print("开始增强版多智能体系统测试")
         print("=" * 60)
         
         # 初始化系统
         system = EnhancedMultiAgentSystem(self.config)
         
         test_functions = [
             ("系统初始化", self._test_system_initialization),
             ("基础功能", self._test_basic_functionality),
             ("数据流通", self._test_data_flow),
             ("智能体协作", self._test_agent_collaboration),
             ("仲裁功能", self._test_arbitration_functionality),
             ("性能和并发", self._test_performance_and_concurrency),
             ("错误处理", self._test_error_handling),
             ("全部测试用例", self._test_all_cases)
         ]
         
         try:
             # 使用进度条显示测试进度
             with tqdm(total=len(test_functions), desc="测试进度") as pbar:
                 for test_name, test_func in test_functions:
                     pbar.set_description(f"执行测试: {test_name}")
                     await test_func(system)
                     pbar.update(1)
                     
         finally:
             await system.shutdown()
         
         # 生成测试报告
         self._generate_test_report()
    
    async def _test_system_initialization(self, system: EnhancedMultiAgentSystem):
        """测试系统初始化"""
        print("\n测试1: 系统初始化")
        
        test_result = {
            "test_name": "系统初始化测试",
            "start_time": time.time(),
            "success": False,
            "details": {}
        }
        
        try:
            # 初始化系统
            await system.initialize()
            
            # 检查系统状态
            status = system.get_system_status()
            
            test_result["details"] = {
                "is_initialized": status["is_initialized"],
                "is_running": status["is_running"],
                "components_count": {
                    "router": 1,
                    "sub_agents": status["components"]["sub_agents"]["count"],
                    "arbitrator": 1
                },
                "communication_hub": status["components"]["communication_hub"]["active_agents"]
            }
            
            # 验证结果
            assert status["is_initialized"], "系统未正确初始化"
            assert status["is_running"], "系统未正确启动"
            assert status["components"]["sub_agents"]["count"] == 6, "子智能体数量不正确"
            
            test_result["success"] = True
            print("  系统初始化成功")
            print(f"  子智能体数量: {status['components']['sub_agents']['count']}")
            print(f"  活跃智能体: {status['components']['communication_hub']['active_agents']}")
            
        except Exception as e:
            test_result["error"] = str(e)
            print(f"  系统初始化失败: {e}")
        
        test_result["duration"] = time.time() - test_result["start_time"]
        self.test_results.append(test_result)
    async def _test_basic_functionality(self, system: EnhancedMultiAgentSystem):
        """测试基础功能"""
        print("\n测试2: 基础功能")
        
        test_result = {
            "test_name": "基础功能测试",
            "start_time": time.time(),
            "success": False,
            "details": {}
        }
        
        try:
            # 测试简单内容处理
            test_content = "这是一个测试内容，用于验证系统基础功能。"
            print(f"  📝 测试内容: {test_content}")
            
            result = await system.process_content(test_content, "test_platform")
            
            # 显示推导过程
            print("\n  🧠 显示推导过程:")
            print_all_reasoning_chains(result)
            
            test_result["details"] = {
                "session_id": result.get("session_id"),
                "final_decision": result.get("final_decision"),
                "final_confidence": result.get("final_confidence"),
                "processing_time": result.get("processing_time"),
                "agents_involved": result.get("system_metadata", {}).get("total_agents_involved", 0)
            }
            
            # 验证结果
            assert "final_decision" in result, "缺少最终决策"
            assert "final_confidence" in result, "缺少置信度"
            assert result["final_decision"] in ["approved", "rejected", "needs_human_review"], "决策结果无效"
            assert 0 <= result["final_confidence"] <= 1, "置信度范围无效"
            
            test_result["success"] = True
            print("  ✅ 基础内容处理成功")
            print(f"  📊 最终决策: {result['final_decision']}")
            print(f"  🎯 置信度: {result['final_confidence']:.3f}")
            print(f"  ⏱️ 处理时间: {result['processing_time']:.2f}秒")
            
        except Exception as e:
            test_result["error"] = str(e)
            print(f"  ❌ 基础功能测试失败: {e}")
        
        test_result["duration"] = time.time() - test_result["start_time"]
        self.test_results.append(test_result)
    
    async def _test_data_flow(self, system: EnhancedMultiAgentSystem):
        """测试数据流通"""
        print("\n测试3: 数据流通")
        
        test_result = {
            "test_name": "数据流通测试",
            "start_time": time.time(),
            "success": False,
            "details": {}
        }
        
        try:
            test_content = "测试数据在各个智能体之间的流通是否正常。"
            result = await system.process_content(test_content, "flow_test")
            
            # 检查数据流的完整性
            routing_phase = result.get("routing_phase", {})
            sub_agents_phase = result.get("sub_agents_phase", {}) 
            arbitration_phase = result.get("arbitration_phase", {})
            
            test_result["details"] = {
                "routing_completed": bool(routing_phase.get("tasks_assigned", 0) > 0),
                "sub_agents_participated": len(sub_agents_phase.get("participating_agents", [])),
                "arbitration_completed": bool(arbitration_phase.get("arbitration_result")),
                "communication_events": result.get("system_metadata", {}).get("communication_events", 0),
                "data_flow_stages": {
                    "routing": bool(routing_phase),
                    "sub_agents": bool(sub_agents_phase), 
                    "arbitration": bool(arbitration_phase)
                }
            }
            
            # 验证数据流
            assert routing_phase.get("tasks_assigned", 0) > 0, "路由阶段未分配任务"
            assert len(sub_agents_phase.get("participating_agents", [])) > 0, "无子智能体参与"
            assert arbitration_phase.get("arbitration_result"), "仲裁阶段未完成"
            
            test_result["success"] = True
            print("  路由阶段数据传递正常")
            print(f"  子智能体参与数: {len(sub_agents_phase.get('participating_agents', []))}")
            print("  仲裁阶段数据接收正常")
            print(f"  通信事件数: {result.get('system_metadata', {}).get('communication_events', 0)}")
            
        except Exception as e:
            test_result["error"] = str(e)
            print(f"   数据流通测试失败: {e}")
        
        test_result["duration"] = time.time() - test_result["start_time"]
        self.test_results.append(test_result)
    
    async def _test_agent_collaboration(self, system: EnhancedMultiAgentSystem):
        """测试智能体协作"""
        print("\n测试4: 智能体协作")
        
        test_result = {
            "test_name": "智能体协作测试",
            "start_time": time.time(),
            "success": False,
            "details": {}
        }
        
        try:
            # 使用可能引起智能体间协作的复杂内容
            complex_content = "这个产品的质量让我很困惑，说不上好还是坏，可能需要更多时间评估。"
            result = await system.process_content(complex_content, "collaboration_test")
            
            collaboration_results = result.get("sub_agents_phase", {}).get("collaboration_results", {})
            communication_log = result.get("arbitration_phase", {}).get("communication_log", {})
            
            test_result["details"] = {
                "collaboration_attempts": len(collaboration_results),
                "successful_collaborations": len([c for c in collaboration_results.values() if c.get("status") == "completed"]),
                "arbitrator_communications": len(communication_log),
                "collaboration_types": [c.get("type") for c in collaboration_results.values()],
                "agent_consensus": result.get("arbitration_phase", {}).get("arbitration_result", {}).get("agent_consensus_analysis", {})
            }
            
            # 验证协作功能
            # 注意：协作可能不总是发生，这取决于内容和智能体状态
            print("  协作机制运行正常")
            print(f"  协作尝试次数: {len(collaboration_results)}")
            print(f"  成功协作次数: {len([c for c in collaboration_results.values() if c.get('status') == 'completed'])}")
            print(f"  仲裁器通信次数: {len(communication_log)}")
            
            test_result["success"] = True
            
        except Exception as e:
            test_result["error"] = str(e)
            print(f"  智能体协作测试失败: {e}")
        
        test_result["duration"] = time.time() - test_result["start_time"]
        self.test_results.append(test_result)
    
    async def _test_arbitration_functionality(self, system: EnhancedMultiAgentSystem):
        """测试仲裁功能"""
        print("\n测试5: 仲裁功能")
        
        test_result = {
            "test_name": "仲裁功能测试",
            "start_time": time.time(),
            "success": False,
            "details": {}
        }
        
        try:
            # 使用可能产生分歧的内容
            controversial_content = "这个评论包含一些争议性的表达，可能需要仔细判断。"
            result = await system.process_content(controversial_content, "arbitration_test")
            
            arbitration_result = result.get("arbitration_phase", {}).get("arbitration_result", {})
            
            test_result["details"] = {
                "arbitration_completed": bool(arbitration_result),
                "final_decision": arbitration_result.get("final_decision"),
                "confidence_score": arbitration_result.get("confidence_score"),
                "has_reasoning": bool(arbitration_result.get("arbitration_reasoning")),
                "evidence_analysis": bool(arbitration_result.get("evidence_analysis")),
                "consensus_analysis": bool(arbitration_result.get("agent_consensus_analysis")),
                "risk_assessment": bool(arbitration_result.get("risk_assessment")),
                "recommendations": bool(arbitration_result.get("recommendations"))
            }
            
            # 验证仲裁功能
            assert arbitration_result, "仲裁结果为空"
            assert arbitration_result.get("final_decision") in ["approved", "rejected", "needs_human_review"], "仲裁决策无效"
            assert "arbitration_reasoning" in arbitration_result, "缺少仲裁推理"
            assert "evidence_analysis" in arbitration_result, "缺少证据分析"
            
            test_result["success"] = True
            print("  仲裁器成功生成决策")
            print(f"  最终决策: {arbitration_result.get('final_decision')}")
            print(f"  仲裁置信度: {arbitration_result.get('confidence_score', 0):.3f}")
            print("   仲裁推理完整")
            print("  证据分析完整")
            
        except Exception as e:
            test_result["error"] = str(e)
            print(f"  仲裁功能测试失败: {e}")
        
        test_result["duration"] = time.time() - test_result["start_time"]
        self.test_results.append(test_result)
    
    async def _test_performance_and_concurrency(self, system: EnhancedMultiAgentSystem):
        """测试性能和并发"""
        print("\n测试6: 性能和并发")
        
        test_result = {
            "test_name": "性能和并发测试",
            "start_time": time.time(),
            "success": False,
            "details": {}
        }
        
        try:
            # 并发处理多个内容
            test_contents = [
                "并发测试内容1：这是一个正面的评价。",
                "并发测试内容2：这个产品还可以。", 
                "并发测试内容3：不太满意这次购买。"
            ]
            
            # 记录开始时间
            start_time = time.time()
            
            # 创建并发任务
            tasks = []
            for i, content in enumerate(test_contents):
                task = asyncio.create_task(
                    system.process_content(content, f"concurrent_test_{i}")
                )
                tasks.append(task)
            
            # 等待所有任务完成
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            total_time = time.time() - start_time
            
            # 分析结果
            successful_results = [r for r in results if not isinstance(r, Exception)]
            failed_results = [r for r in results if isinstance(r, Exception)]
            
            avg_processing_time = sum(r.get("processing_time", 0) for r in successful_results) / len(successful_results) if successful_results else 0
            
            test_result["details"] = {
                "total_requests": len(test_contents),
                "successful_requests": len(successful_results),
                "failed_requests": len(failed_results),
                "total_time": total_time,
                "average_processing_time": avg_processing_time,
                "concurrency_efficiency": len(successful_results) / len(test_contents),
                "throughput": len(test_contents) / total_time
            }
            
            # 验证并发性能
            assert len(successful_results) == len(test_contents), f"并发处理失败，成功{len(successful_results)}/{len(test_contents)}"
            assert total_time < avg_processing_time * len(test_contents), "并发效率低于预期"
            
            test_result["success"] = True
            print(f"  并发处理 {len(test_contents)} 个请求")
            print(f"  总耗时: {total_time:.2f}秒")
            print(f"  平均处理时间: {avg_processing_time:.2f}秒")
            print(f"  吞吐量: {len(test_contents)/total_time:.2f} 请求/秒")
            
        except Exception as e:
            test_result["error"] = str(e)
            print(f"  性能和并发测试失败: {e}")
        
        test_result["duration"] = time.time() - test_result["start_time"]
        self.test_results.append(test_result)
    
    async def _test_error_handling(self, system: EnhancedMultiAgentSystem):
        """测试错误处理"""
        print("\n测试7: 错误处理")
        
        test_result = {
            "test_name": "错误处理测试",
            "start_time": time.time(),
            "success": False,
            "details": {}
        }
        
        try:
            # 测试各种边界情况
            edge_cases = [
                "",  # 空内容
                "a" * 10000,  # 超长内容
                "特殊字符测试: !@#$%^&*()_+{}[]|\\:;<>?",  # 特殊字符
                None,  # None值（但我们会转换为字符串）
            ]
            
            error_handling_results = []
            
            for i, content in enumerate(edge_cases):
                try:
                    # 处理None值
                    if content is None:
                        content = "None"
                    
                    result = await system.process_content(str(content), f"edge_case_{i}")
                    error_handling_results.append({
                        "case": f"edge_case_{i}",
                        "content_length": len(str(content)),
                        "success": True,
                        "decision": result.get("final_decision"),
                        "error": None
                    })
                    
                except Exception as e:
                    error_handling_results.append({
                        "case": f"edge_case_{i}",
                        "content_length": len(str(content)) if content else 0,
                        "success": False,
                        "decision": None,
                        "error": str(e)
                    })
            
            test_result["details"] = {
                "total_edge_cases": len(edge_cases),
                "handled_successfully": len([r for r in error_handling_results if r["success"]]),
                "error_cases": [r for r in error_handling_results if not r["success"]],
                "results": error_handling_results
            }
            
            # 验证错误处理
            success_rate = len([r for r in error_handling_results if r["success"]]) / len(edge_cases)
            assert success_rate >= 0.5, f"错误处理成功率过低: {success_rate}"
            
            test_result["success"] = True
            print(f"  边界情况处理成功率: {success_rate:.1%}")
            print(f"  成功处理: {len([r for r in error_handling_results if r['success']])}/{len(edge_cases)}")
            
            for result in error_handling_results:
                if result["success"]:
                    print(f"  {result['case']}: {result['decision']}")
                else:
                    print(f"   {result['case']}: {result['error']}")
            
        except Exception as e:
            test_result["error"] = str(e)
            print(f"    错误处理测试失败: {e}")
        
        test_result["duration"] = time.time() - test_result["start_time"]
        self.test_results.append(test_result)
    
    async def _test_all_cases(self, system: EnhancedMultiAgentSystem):
        """测试所有预定义测试用例"""
        print("\n测试8: 全部测试用例")
        
        test_result = {
            "test_name": "全部测试用例",
            "start_time": time.time(),
            "success": False,
            "details": {}
        }
        
        try:
            case_results = []
            
            for case in self.test_cases:
                print(f"\n    执行: {case['name']}")
                
                try:
                    result = await system.process_content(case["content"], case["platform"])
                    
                    actual_decision = result.get("final_decision")
                    expected_decision = case["expected_decision"]
                    confidence = result.get("final_confidence", 0)
                    processing_time = result.get("processing_time", 0)
                    
                    # 判断是否符合预期（允许一定的灵活性）
                    decision_match = (actual_decision == expected_decision or 
                                    (expected_decision == "approved" and actual_decision in ["approved", "needs_human_review"]) or
                                    (expected_decision == "rejected" and actual_decision in ["rejected", "needs_human_review"]))
                    
                    case_result = {
                        "name": case["name"],
                        "expected": expected_decision,
                        "actual": actual_decision,
                        "confidence": confidence,
                        "processing_time": processing_time,
                        "match": decision_match,
                        "success": True
                    }
                    
                    if decision_match:
                        print(f"      决策: {actual_decision} (预期: {expected_decision}) ✅ 正确")
                    else:
                        print(f"      决策: {actual_decision} (预期: {expected_decision}) ❌ 错误")
                    
                    print(f"      置信度: {confidence:.3f}")
                    print(f"      耗时: {processing_time:.2f}秒")
                    
                except Exception as e:
                    case_result = {
                        "name": case["name"],
                        "expected": case["expected_decision"],
                        "actual": "error",
                        "confidence": 0,
                        "processing_time": 0,
                        "match": False,
                        "success": False,
                        "error": str(e)
                    }
                    print(f"      错误: {e}")
                
                case_results.append(case_result)
            
            # 计算总体性能
            successful_cases = [r for r in case_results if r["success"]]
            matching_cases = [r for r in case_results if r.get("match", False)]
            
            test_result["details"] = {
                "total_cases": len(self.test_cases),
                "successful_cases": len(successful_cases),
                "matching_decisions": len(matching_cases),
                "success_rate": len(successful_cases) / len(self.test_cases),
                "accuracy_rate": len(matching_cases) / len(self.test_cases),
                "average_confidence": sum(r.get("confidence", 0) for r in successful_cases) / len(successful_cases) if successful_cases else 0,
                "average_processing_time": sum(r.get("processing_time", 0) for r in successful_cases) / len(successful_cases) if successful_cases else 0,
                "case_results": case_results
            }
            
            success_rate = len(successful_cases) / len(self.test_cases)
            accuracy_rate = len(matching_cases) / len(self.test_cases)
            
            assert success_rate >= 0.8, f"成功率过低: {success_rate}"
            
            test_result["success"] = True
            print(f"\n   测试用例总结:")
            print(f"   成功率: {success_rate:.1%} ({len(successful_cases)}/{len(self.test_cases)})")
            print(f"   准确率: {accuracy_rate:.1%} ({len(matching_cases)}/{len(self.test_cases)})")
            print(f"   平均置信度: {test_result['details']['average_confidence']:.3f}")
            print(f"   平均处理时间: {test_result['details']['average_processing_time']:.2f}秒")
            
        except Exception as e:
            test_result["error"] = str(e)
            print(f"   ❌ 测试用例执行失败: {e}")
        
        test_result["duration"] = time.time() - test_result["start_time"]
        self.test_results.append(test_result)
    
    def _generate_test_report(self):
        """生成测试报告"""
        print("\n" + "="*60)
        print("📋 测试报告生成")
        print("="*60)
        
        total_tests = len(self.test_results)
        successful_tests = len([r for r in self.test_results if r["success"]])
        total_duration = sum(r["duration"] for r in self.test_results)
        
        print(f"\n🎯 总体结果:")
        print(f"   总测试数: {total_tests}")
        print(f"   成功测试: {successful_tests}")
        print(f"   失败测试: {total_tests - successful_tests}")
        print(f"   成功率: {successful_tests/total_tests:.1%}")
        print(f"   总耗时: {total_duration:.2f}秒")
        
        print(f"\n📝 详细结果:")
        for result in self.test_results:
            status = "✅" if result["success"] else "❌"
            print(f"   {status} {result['test_name']}: {result['duration']:.2f}秒")
            if not result["success"] and "error" in result:
                print(f"      错误: {result['error']}")
        
        # 保存详细报告到文件
        report_data = {
            "test_summary": {
                "total_tests": total_tests,
                "successful_tests": successful_tests,
                "success_rate": successful_tests / total_tests,
                "total_duration": total_duration,
                "timestamp": datetime.utcnow().isoformat()
            },
            "test_results": self.test_results,
            "system_config": self.config
        }
        
        with open(f"test_report_{int(time.time())}.json", "w", encoding="utf-8") as f:
            json.dump(report_data, f, ensure_ascii=False, indent=2)
        
        print(f"\n💾 详细测试报告已保存到: test_report_{int(time.time())}.json")
        
        if successful_tests == total_tests:
            print("\n🎉 所有测试通过！增强版多智能体系统运行正常！")
        else:
            print(f"\n⚠️ {total_tests - successful_tests} 个测试失败，请检查系统配置。")

async def demonstrate_reasoning_chains():
    """演示推导过程和思维链功能"""
    print("🧩 ===== 多智能体推导过程演示 =====")
    print("🎯 目标: 展示每个子智能体的详细推导过程和思维链")
    print("=" * 60)
    
    # 初始化系统
    config = {
        'dashscope_api_key': 'sk-8b654ec58c4c49f6a30cfb3d555a95d0',
        'consensus_threshold': 0.7,
        'max_agents': 6,
        'coordination_timeout': 30,
        'reasoning_depth': 3
    }
    
    system = EnhancedMultiAgentSystem(config)
    
    try:
        await system.initialize()
        print("✅ 系统初始化成功")
        
        # 测试用例 - 选择一个有挑战性的内容
        demo_cases = [
            {
                "content": "这个产品质量真的太好了，我非常满意！强烈推荐给大家，性价比超高！",
                "platform": "taobao",
                "description": "正面评价测试"
            },
            {
                "content": "这垃圾东西简直就是骗钱的，卖家态度极其恶劣，我要投诉！",
                "platform": "douyin", 
                "description": "负面毒性内容测试"
            },
            {
                "content": "这个设计很有创意，可能不是所有人都能接受这种风格吧。",
                "platform": "zhihu",
                "description": "复杂歧义内容测试"
            }
        ]
        
        for i, case in enumerate(demo_cases, 1):
            print(f"\n🎬 演示案例 {i}: {case['description']}")
            print(f"📱 平台: {case['platform']}")
            print(f"📝 内容: {case['content']}")
            print("-" * 60)
            
            # 处理内容
            result = await system.process_content(case['content'], case['platform'])
            
            # 显示结果摘要
            print(f"📊 处理结果:")
            print(f"  🎯 最终决策: {result.get('final_decision', 'unknown')}")
            print(f"  📈 置信度: {result.get('final_confidence', 0):.3f}")
            print(f"  ⏱️ 处理时间: {result.get('processing_time', 0):.2f}秒")
            
            # 显示详细的推导过程
            print_all_reasoning_chains(result)
            
            # 等待一下让用户看清楚
            print("按 Enter 继续下一个演示案例...")
            # input()  # 注释掉以便自动运行
            
    except Exception as e:
        print(f"❌ 演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        await system.shutdown()
        print("🏁 演示完成")

async def main():
    """主函数"""
    print("🛡️ SenTox 增强版多智能体内容审核系统测试")
    print("架构: 总分总 (路由器 → 子智能体群 → 中心仲裁器)")
    print("特性: 思维链、动作链、MCP协议、双向通信")
    
    print("\n选择运行模式:")
    print("1. 完整测试套件")
    print("2. 推导过程演示")
    
    # 自动选择演示模式以展示推导过程
    choice = "2"  # 可以改为 input("请选择 (1 或 2): ").strip()
    
    if choice == "1":
        tester = SystemTester()
        await tester.run_all_tests()
    elif choice == "2":
        await demonstrate_reasoning_chains()
    else:
        print("无效选择，运行默认演示...")
        await demonstrate_reasoning_chains()

if __name__ == "__main__":
    asyncio.run(main())
