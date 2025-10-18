/**
 * 多智能体推理过程可视化模块
 * 用于在网站中实时显示推理进度和各个智能体的工作状态
 */

class MultiAgentProgressVisualizer {
    constructor(containerId) {
        this.container = document.getElementById(containerId);
        this.currentSession = null;
        this.stages = [
            { id: 'router', name: '中心路由器', icon: '🎯', color: '#3498db' },
            { id: 'subagents', name: '子智能体群', icon: '🤖', color: '#2ecc71' },
            { id: 'collaboration', name: '智能体协作', icon: '🤝', color: '#f39c12' },
            { id: 'arbitration', name: '中心仲裁', icon: '⚖️', color: '#e74c3c' },
            { id: 'communication', name: '双向沟通', icon: '💬', color: '#9b59b6' }
        ];
        this.subAgents = [
            { id: 'content_analyzer', name: '内容分析', color: '#3498db' },
            { id: 'semantic_analyzer', name: '语义分析', color: '#2ecc71' },
            { id: 'sentiment_analyzer', name: '情感分析', color: '#f39c12' },
            { id: 'toxicity_detector', name: '毒性检测', color: '#e74c3c' },
            { id: 'context_analyzer', name: '上下文分析', color: '#9b59b6' },
            { id: 'risk_assessor', name: '风险评估', color: '#34495e' }
        ];
        this.init();
    }

    init() {
        this.container.innerHTML = `
            <div class="progress-container">
                <div class="progress-header">
                    <h3>多智能体推理进度</h3>
                    <div class="session-info">
                        <span id="session-id">等待处理...</span>
                        <span id="processing-time">0.00s</span>
                    </div>
                </div>
                
                <div class="main-progress">
                    <div class="progress-bar-container">
                        <div class="progress-bar" id="main-progress-bar">
                            <div class="progress-fill" style="width: 0%"></div>
                        </div>
                        <div class="progress-text" id="main-progress-text">等待开始...</div>
                    </div>
                </div>

                <div class="stages-container" id="stages-container">
                    ${this.renderStages()}
                </div>

                <div class="sub-agents-container" id="sub-agents-container">
                    <h4>子智能体状态</h4>
                    <div class="agents-grid">
                        ${this.renderSubAgents()}
                    </div>
                </div>

                <div class="results-container" id="results-container" style="display: none;">
                    <h4>处理结果</h4>
                    <div class="result-summary" id="result-summary"></div>
                </div>

                <div class="logs-container" id="logs-container">
                    <h4>实时日志</h4>
                    <div class="logs" id="logs"></div>
                </div>
            </div>
        `;
    }

    renderStages() {
        return this.stages.map(stage => `
            <div class="stage" id="stage-${stage.id}" data-stage="${stage.id}">
                <div class="stage-icon" style="background-color: ${stage.color}">${stage.icon}</div>
                <div class="stage-name">${stage.name}</div>
                <div class="stage-status">待处理</div>
                <div class="stage-progress">
                    <div class="mini-progress-bar">
                        <div class="mini-progress-fill" style="width: 0%"></div>
                    </div>
                </div>
            </div>
        `).join('');
    }

    renderSubAgents() {
        return this.subAgents.map(agent => `
            <div class="agent-card" id="agent-${agent.id}" data-agent="${agent.id}">
                <div class="agent-header" style="border-left: 4px solid ${agent.color}">
                    <div class="agent-name">${agent.name}</div>
                    <div class="agent-status">待机</div>
                </div>
                <div class="agent-progress">
                    <div class="mini-progress-bar">
                        <div class="mini-progress-fill" style="width: 0%"></div>
                    </div>
                </div>
                <div class="agent-details">
                    <div class="agent-confidence">置信度: --</div>
                    <div class="agent-time">耗时: --</div>
                </div>
            </div>
        `).join('');
    }

    startProcessing(sessionId, content) {
        this.currentSession = sessionId;
        this.startTime = Date.now();
        
        document.getElementById('session-id').textContent = `会话: ${sessionId}`;
        this.updateMainProgress(0, '开始处理...');
        this.addLog('info', `开始处理内容: ${content.substring(0, 50)}...`);
        
        // 重置所有状态
        this.resetAllStates();
        document.getElementById('results-container').style.display = 'none';
    }

    updateStage(stageId, status, progress = 0, details = '') {
        const stageElement = document.getElementById(`stage-${stageId}`);
        if (!stageElement) return;

        const statusElement = stageElement.querySelector('.stage-status');
        const progressElement = stageElement.querySelector('.mini-progress-fill');
        
        statusElement.textContent = status;
        progressElement.style.width = `${progress}%`;
        
        // 更新样式
        stageElement.className = 'stage';
        if (status === '处理中') {
            stageElement.classList.add('processing');
        } else if (status === '完成') {
            stageElement.classList.add('completed');
        } else if (status === '失败') {
            stageElement.classList.add('failed');
        }

        this.addLog('stage', `${this.getStageNameById(stageId)}: ${status} ${details}`);
    }

    updateSubAgent(agentId, status, progress = 0, confidence = null, processingTime = null) {
        const agentElement = document.getElementById(`agent-${agentId}`);
        if (!agentElement) return;

        const statusElement = agentElement.querySelector('.agent-status');
        const progressElement = agentElement.querySelector('.mini-progress-fill');
        const confidenceElement = agentElement.querySelector('.agent-confidence');
        const timeElement = agentElement.querySelector('.agent-time');
        
        statusElement.textContent = status;
        progressElement.style.width = `${progress}%`;
        
        if (confidence !== null) {
            confidenceElement.textContent = `置信度: ${(confidence * 100).toFixed(1)}%`;
        }
        
        if (processingTime !== null) {
            timeElement.textContent = `耗时: ${processingTime.toFixed(2)}s`;
        }

        // 更新样式
        agentElement.className = 'agent-card';
        if (status === '处理中') {
            agentElement.classList.add('processing');
        } else if (status === '完成') {
            agentElement.classList.add('completed');
        } else if (status === '失败') {
            agentElement.classList.add('failed');
        }

        this.addLog('agent', `${this.getAgentNameById(agentId)}: ${status}`);
    }

    updateMainProgress(percentage, text) {
        const progressFill = document.querySelector('#main-progress-bar .progress-fill');
        const progressText = document.getElementById('main-progress-text');
        
        progressFill.style.width = `${percentage}%`;
        progressText.textContent = text;
        
        // 更新处理时间
        if (this.startTime) {
            const elapsed = (Date.now() - this.startTime) / 1000;
            document.getElementById('processing-time').textContent = `${elapsed.toFixed(2)}s`;
        }
    }

    updateResults(results) {
        const resultsContainer = document.getElementById('results-container');
        const resultSummary = document.getElementById('result-summary');
        
        resultSummary.innerHTML = `
            <div class="result-item">
                <strong>最终决策:</strong> 
                <span class="decision ${results.final_decision}">${this.formatDecision(results.final_decision)}</span>
            </div>
            <div class="result-item">
                <strong>置信度:</strong> 
                <span class="confidence">${(results.final_confidence * 100).toFixed(1)}%</span>
            </div>
            <div class="result-item">
                <strong>处理时间:</strong> 
                <span class="time">${results.processing_time.toFixed(2)}秒</span>
            </div>
            <div class="result-item">
                <strong>参与智能体:</strong> 
                <span class="agents">${results.system_metadata?.total_agents_involved || 0}个</span>
            </div>
        `;
        
        resultsContainer.style.display = 'block';
        this.addLog('result', `处理完成: ${this.formatDecision(results.final_decision)} (置信度: ${(results.final_confidence * 100).toFixed(1)}%)`);
    }

    addLog(type, message) {
        const logsContainer = document.getElementById('logs');
        const timestamp = new Date().toLocaleTimeString();
        
        const logEntry = document.createElement('div');
        logEntry.className = `log-entry log-${type}`;
        logEntry.innerHTML = `
            <span class="log-time">[${timestamp}]</span>
            <span class="log-message">${message}</span>
        `;
        
        logsContainer.appendChild(logEntry);
        logsContainer.scrollTop = logsContainer.scrollHeight;
        
        // 限制日志条数
        const logEntries = logsContainer.querySelectorAll('.log-entry');
        if (logEntries.length > 50) {
            logEntries[0].remove();
        }
    }

    resetAllStates() {
        // 重置阶段状态
        this.stages.forEach(stage => {
            this.updateStage(stage.id, '待处理', 0);
        });
        
        // 重置智能体状态
        this.subAgents.forEach(agent => {
            this.updateSubAgent(agent.id, '待机', 0);
        });
        
        // 清空日志
        document.getElementById('logs').innerHTML = '';
    }

    getStageNameById(stageId) {
        const stage = this.stages.find(s => s.id === stageId);
        return stage ? stage.name : stageId;
    }

    getAgentNameById(agentId) {
        const agent = this.subAgents.find(a => a.id === agentId);
        return agent ? agent.name : agentId;
    }

    formatDecision(decision) {
        const decisions = {
            'approved': '通过',
            'rejected': '拒绝',
            'needs_human_review': '需人工审核',
            'error': '错误'
        };
        return decisions[decision] || decision;
    }

    // 模拟处理进度的方法
    simulateProgress() {
        let currentStage = 0;
        const stageProgress = [
            { stage: 'router', duration: 2000 },
            { stage: 'subagents', duration: 5000 },
            { stage: 'collaboration', duration: 3000 },
            { stage: 'arbitration', duration: 4000 },
            { stage: 'communication', duration: 2000 }
        ];

        const processStage = (index) => {
            if (index >= stageProgress.length) return;

            const { stage, duration } = stageProgress[index];
            this.updateStage(stage, '处理中', 0);
            
            const steps = 20;
            const stepDuration = duration / steps;
            let step = 0;

            const updateProgress = () => {
                step++;
                const progress = (step / steps) * 100;
                this.updateStage(stage, '处理中', progress);
                this.updateMainProgress((index + step / steps) / stageProgress.length * 100, `正在${this.getStageNameById(stage)}...`);

                if (step < steps) {
                    setTimeout(updateProgress, stepDuration);
                } else {
                    this.updateStage(stage, '完成', 100);
                    setTimeout(() => processStage(index + 1), 500);
                }
            };

            updateProgress();
        };

        processStage(0);
    }
}

// 全局初始化
let progressVisualizer;

document.addEventListener('DOMContentLoaded', function() {
    if (document.getElementById('progress-container')) {
        progressVisualizer = new MultiAgentProgressVisualizer('progress-container');
    }
});

// 导出给其他模块使用
window.MultiAgentProgressVisualizer = MultiAgentProgressVisualizer;
