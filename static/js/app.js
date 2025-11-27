let conversationHistory = [];
let currentSessionId = null;  // Track current chat session

document.addEventListener('DOMContentLoaded', function() {
    // Set current session ID from template
    if (typeof CURRENT_SESSION_ID !== 'undefined' && CURRENT_SESSION_ID !== null) {
        currentSessionId = CURRENT_SESSION_ID;
    }
    
    checkConfiguration();
    loadChatSessions();
    
    // Load existing session messages if session ID is present
    if (currentSessionId) {
        loadCurrentSessionMessages();
    }
    
    const chatForm = document.getElementById('chatForm');
    chatForm.addEventListener('submit', handleSubmit);
    
    // 送信/中止ボタンのクリックハンドラ
    const sendBtn = document.getElementById('sendBtn');
    if (sendBtn) {
        sendBtn.addEventListener('click', async (e) => {
            if (isProcessing) {
                e.preventDefault();
                e.stopPropagation();
                await cancelCurrentTask();
            }
        });
    }
    
    // テキストエリアのEnterキーハンドリング
    const questionInput = document.getElementById('questionInput');
    if (questionInput) {
        // Shift+Enterで改行、Enterのみで送信
        questionInput.addEventListener('keydown', function(e) {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault(); // デフォルトの改行を防止
                chatForm.dispatchEvent(new Event('submit', { cancelable: true }));
            }
        });
        
        // テキストエリアの自動高さ調整
        questionInput.addEventListener('input', function() {
            this.style.height = 'auto';
            this.style.height = Math.min(this.scrollHeight, 200) + 'px';
        });
    }
});

async function checkConfiguration() {
    const statusDiv = document.getElementById('configStatus');
    
    try {
        const response = await fetch('/api/settings');
        
        if (response.status === 404) {
            // No active project
            statusDiv.className = 'config-status mt-3 p-3 rounded error';
            statusDiv.innerHTML = `
                <i class="bi bi-exclamation-triangle-fill text-warning"></i>
                <strong>プロジェクトが選択されていません</strong><br>
                <a href="/projects" class="text-decoration-none">プロジェクト管理</a> ページでプロジェクトを作成・選択してください。
            `;
            // Disable chat input
            document.getElementById('questionInput').disabled = true;
            document.getElementById('sendBtn').disabled = true;
            return;
        }
        
        const config = await response.json();
        
        if (!config.success) {
            if (config.no_project) {
                statusDiv.className = 'config-status mt-3 p-3 rounded error';
                statusDiv.innerHTML = `
                    <i class="bi bi-exclamation-triangle-fill text-warning"></i>
                    <strong>プロジェクトが選択されていません</strong><br>
                    <a href="/projects" class="text-decoration-none">プロジェクト管理</a> ページでプロジェクトを作成・選択してください。
                `;
                // Disable chat input
                document.getElementById('questionInput').disabled = true;
                document.getElementById('sendBtn').disabled = true;
                return;
            }
            throw new Error(config.error || 'Failed to load configuration');
        }
        
        // Set AI provider from project settings
        const aiProviderSelect = document.getElementById('aiProviderSelect');
        if (aiProviderSelect && config.ai_provider) {
            aiProviderSelect.value = config.ai_provider;
        }
        
        // Check if at least one AI API key is configured (OpenAI OR Gemini)
        const hasAnyApiKey = config.has_api_key || config.has_gemini_key;
        
        if (hasAnyApiKey && config.has_service_account && config.project_id && config.dataset) {
            const aiProvider = config.ai_provider === 'gemini' ? 'Gemini' : 'OpenAI';
            statusDiv.className = 'config-status mt-3 p-3 rounded success';
            statusDiv.innerHTML = `
                <i class="bi bi-check-circle-fill text-success"></i>
                <strong>設定完了:</strong> プロジェクト: <strong>${config.project_name}</strong> | AI: ${aiProvider} | BigQuery: ${config.project_id} | データセット: ${config.dataset}
            `;
        } else {
            statusDiv.className = 'config-status mt-3 p-3 rounded error';
            let missing = [];
            if (!hasAnyApiKey) missing.push('API キー (OpenAI または Gemini)');
            if (!config.has_service_account) missing.push('GCP サービスアカウント');
            if (!config.project_id) missing.push('BigQueryプロジェクトID');
            if (!config.dataset) missing.push('データセット');
            statusDiv.innerHTML = `
                <i class="bi bi-exclamation-triangle-fill text-warning"></i>
                <strong>設定が不完全です:</strong> ${missing.join(', ')}が設定されていません<br>
                <a href="/settings" class="text-decoration-none">設定ページ</a> で設定してください。
            `;
        }
    } catch (error) {
        statusDiv.className = 'config-status mt-3 p-3 rounded error';
        statusDiv.innerHTML = `
            <i class="bi bi-x-circle-fill text-danger"></i>
            <strong>エラー:</strong> 設定の確認に失敗しました
        `;
    }
}

async function handleSubmit(event) {
    event.preventDefault();
    
    const input = document.getElementById('questionInput');
    const question = input.value.trim();
    
    if (!question) return;
    
    console.log('Submitting question:', question);
    console.log('Current session ID:', currentSessionId);
    
    addUserMessage(question);
    input.value = '';
    
    const statusDiv = createStatusDiv();
    const messagesDiv = document.getElementById('chatMessages');
    messagesDiv.appendChild(statusDiv);
    scrollToBottom();
    
    const assistantMessageDiv = createAssistantMessageDiv();
    messagesDiv.appendChild(assistantMessageDiv);
    scrollToBottom();
    
    try {
        // Get selected AI provider
        const aiProviderSelect = document.getElementById('aiProviderSelect');
        const selectedProvider = aiProviderSelect ? aiProviderSelect.value : 'openai';
        
        const requestBody = {
            question: question,
            history: conversationHistory,
            session_id: currentSessionId,
            provider: selectedProvider
        };
        console.log('Sending request with body:', requestBody);
        
        // Start task
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(requestBody)
        });
        
        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            throw new Error(errorData.error || 'サーバーエラー: ' + response.status);
        }
        
        const { task_id, session_id } = await response.json();
        console.log('Task started:', task_id);
        
        // 送信ボタンを中止ボタンに切り替え
        setSendButtonToCancel(task_id);
        
        // Update session ID
        if (session_id) {
            currentSessionId = session_id;
        }
        
        // Poll for progress
        let lastStepCount = 0;
        let lastReasoningLength = 0;
        let pollingActive = true;
        let reasoningDiv = null;
        
        while (pollingActive) {
            await new Promise(resolve => setTimeout(resolve, 500)); // Poll every 500ms
            
            const statusResponse = await fetch(`/api/chat/status/${task_id}`);
            if (!statusResponse.ok) {
                throw new Error('Status check failed');
            }
            
            const statusData = await statusResponse.json();
            console.log('Status update:', statusData);
            
            // Display new steps
            if (statusData.steps && statusData.steps.length > lastStepCount) {
                const newSteps = statusData.steps.slice(lastStepCount);
                for (const step of newSteps) {
                    updateStatus(statusDiv, '', step, 'info');
                }
                lastStepCount = statusData.steps.length;
                scrollToBottom();
            }
            
            // Display reasoning process in real-time
            if (statusData.reasoning && statusData.reasoning.length > lastReasoningLength) {
                const newReasoning = statusData.reasoning.substring(lastReasoningLength);
                
                if (!reasoningDiv) {
                    // Create reasoning container if it doesn't exist
                    reasoningDiv = document.createElement('div');
                    reasoningDiv.className = 'reasoning-process';
                    reasoningDiv.innerHTML = `
                        <div class="reasoning-header">
                            <i class="bi bi-lightbulb-fill"></i> 推論過程
                        </div>
                        <div class="reasoning-content"></div>
                    `;
                    statusDiv.querySelector('.status-logs-container').appendChild(reasoningDiv);
                }
                
                // Append new reasoning content
                const contentDiv = reasoningDiv.querySelector('.reasoning-content');
                contentDiv.textContent += newReasoning;
                lastReasoningLength = statusData.reasoning.length;
                scrollToBottom();
            }
            
            // Check for cancellation
            if (statusData.status === 'cancelled' || statusData.cancelled) {
                pollingActive = false;
                setSendButtonToNormal();
                updateStatus(statusDiv, '⛔', '処理がキャンセルされました', 'warning');
                assistantMessageDiv.remove();
                scrollToBottom();
            }
            // Check completion
            else if (statusData.status === 'completed') {
                pollingActive = false;
                setSendButtonToNormal();
                const result = statusData.result;
                
                // Update session ID from final result
                if (statusData.session_id) {
                    currentSessionId = statusData.session_id;
                    loadChatSessions();
                }
                
                updateStatus(statusDiv, '✅', '処理が完了しました', 'success');
                displayFinalResult(assistantMessageDiv, result);
                conversationHistory.push(
                    { role: 'user', content: question },
                    { role: 'assistant', content: result.answer }
                );
                scrollToBottom();
                
            } else if (statusData.status === 'error') {
                pollingActive = false;
                setSendButtonToNormal();
                updateStatus(statusDiv, '❌', 'エラーが発生しました', 'error');
                assistantMessageDiv.remove();
                addErrorMessage(statusData.error || 'エラーが発生しました');
            }
        }
        
    } catch (error) {
        console.error('Chat error:', error);
        setSendButtonToNormal();
        updateStatus(statusDiv, '❌', '通信エラーが発生しました', 'error');
        assistantMessageDiv.remove();
        addErrorMessage('通信エラーが発生しました: ' + error.message);
    }
}

function createStatusDiv() {
    const div = document.createElement('div');
    div.className = 'message status-message';
    div.innerHTML = `
        <div class="status-logs-container"></div>
    `;
    return div;
}

let currentTaskId = null;
let isProcessing = false;

function setSendButtonToCancel(taskId) {
    currentTaskId = taskId;
    isProcessing = true;
    const sendBtn = document.getElementById('sendBtn');
    if (sendBtn) {
        sendBtn.className = 'btn btn-danger';
        sendBtn.innerHTML = '<i class="bi bi-stop-fill"></i> 中止';
        sendBtn.type = 'button';
    }
}

function setSendButtonToNormal() {
    currentTaskId = null;
    isProcessing = false;
    const sendBtn = document.getElementById('sendBtn');
    if (sendBtn) {
        sendBtn.className = 'btn btn-primary';
        sendBtn.innerHTML = '<i class="bi bi-send-fill"></i> 送信';
        sendBtn.type = 'submit';
    }
}

async function cancelCurrentTask() {
    if (!currentTaskId || !isProcessing) return false;
    
    const sendBtn = document.getElementById('sendBtn');
    if (sendBtn) {
        sendBtn.disabled = true;
        sendBtn.innerHTML = '<i class="bi bi-hourglass-split"></i> 中止中...';
    }
    
    try {
        const response = await fetch(`/api/chat/cancel/${currentTaskId}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            }
        });
        
        if (response.ok) {
            return true;
        }
    } catch (error) {
        console.error('Cancel error:', error);
    } finally {
        if (sendBtn) {
            sendBtn.disabled = false;
        }
    }
    return false;
}

function updateStatus(statusDiv, icon, message, type = '') {
    console.log('updateStatus called:', {icon, message, type, statusDiv});
    const container = statusDiv.querySelector('.status-logs-container');
    console.log('container found:', container);
    if (container) {
        // thinkingタイプの場合は、既存のthinkingメッセージを削除
        if (type === 'thinking') {
            const existingThinking = container.querySelector('.status-indicator.thinking');
            if (existingThinking) {
                existingThinking.remove();
            }
        }
        
        // 新しいステータスログを追加
        const logEntry = document.createElement('div');
        logEntry.className = `status-indicator ${type}`;
        logEntry.innerHTML = `
            <span class="status-icon">${icon}</span>
            <span class="status-text">${message}</span>
        `;
        container.appendChild(logEntry);
        console.log('Status log added successfully');
    } else {
        console.error('Status logs container not found!');
    }
}

function createAssistantMessageDiv(timestamp = null) {
    const div = document.createElement('div');
    div.className = 'message';
    
    const jstTime = timestamp ? formatToJST(timestamp) : '';
    const timestampHtml = jstTime ? `<div class="message-timestamp-outside">${jstTime}</div>` : '';
    
    div.innerHTML = `
        <div class="message-wrapper-assistant">
            ${timestampHtml}
            <div class="message-assistant">
                <h5><i class="bi bi-robot"></i> AIアシスタント</h5>
                <div class="assistant-content"></div>
            </div>
        </div>
    `;
    return div;
}

function updateAssistantMessage(messageDiv, text, isReasoning = false) {
    const contentDiv = messageDiv.querySelector('.assistant-content');
    if (contentDiv) {
        if (isReasoning) {
            // 推論過程は専用のスタイルで表示
            contentDiv.innerHTML = `
                <div class="reasoning-process">
                    <div class="reasoning-header">
                        <i class="bi bi-lightbulb"></i> <strong>推論過程:</strong>
                    </div>
                    <div class="reasoning-content">${formatAnswer(text)}</div>
                </div>
                <span class="typing-cursor">|</span>
            `;
        } else {
            contentDiv.innerHTML = formatAnswer(text) + '<span class="typing-cursor">|</span>';
        }
    }
}

function updateCombinedMessage(messageDiv, reasoningText, normalText) {
    const contentDiv = messageDiv.querySelector('.assistant-content');
    if (contentDiv) {
        let html = '';
        
        // 推論過程がある場合は表示
        if (reasoningText) {
            html += `
                <div class="reasoning-process">
                    <div class="reasoning-header">
                        <i class="bi bi-lightbulb"></i> <strong>推論過程:</strong>
                    </div>
                    <div class="reasoning-content">${formatAnswer(reasoningText)}</div>
                </div>
            `;
        }
        
        // 通常のテキストを表示
        if (normalText) {
            html += formatAnswer(normalText);
        }
        
        html += '<span class="typing-cursor">|</span>';
        contentDiv.innerHTML = html;
    }
}

function displayFinalResult(messageDiv, result) {
    const contentDiv = messageDiv.querySelector('.assistant-content');
    if (!contentDiv) return;
    
    // タイピングカーソルを削除
    const cursor = contentDiv.querySelector('.typing-cursor');
    if (cursor) {
        cursor.remove();
    }
    
    let html = '';
    
    // AIの回答テキストを追加
    if (result.answer) {
        html += `<div class="answer-text mb-3">${formatAnswer(result.answer)}</div>`;
    }
    
    // 実行ステップを追加（デバッグ情報として）
    if (result.steps && result.steps.length > 0) {
        html += `<div class="steps-container mb-2">`;
        html += `<strong><i class="bi bi-gear-fill"></i> 実行ステップ:</strong><br>`;
        result.steps.forEach(step => {
            html += `<div class="step">${escapeHtml(step)}</div>`;
        });
        html += `</div>`;
    }
    
    // データテーブルとグラフを追加
    if (result.data && result.data.length > 0) {
        html += createDataTable(result.data);
        
        // 複数のグラフ設定がある場合は全て描画
        if (result.charts && Array.isArray(result.charts) && result.charts.length > 0) {
            result.charts.forEach(chartConfig => {
                html += createChart(result.data, chartConfig);
            });
        } else {
            // グラフ設定がない場合、デフォルト動作（後方互換性）
            html += createChart(result.data, null);
        }
    }
    
    // Python実行結果を追加
    if (result.python_results && Array.isArray(result.python_results) && result.python_results.length > 0) {
        result.python_results.forEach((pythonResult, index) => {
            html += '<div class="python-result-container mt-3 p-3" style="background: #f8f9fa; border-left: 4px solid #667eea; border-radius: 8px;">';
            html += '<h6 class="mb-2"><i class="bi bi-code-slash"></i> Python実行結果</h6>';
            
            // 出力テキスト
            if (pythonResult.output) {
                html += '<div class="python-output mb-2" style="background: #fff; padding: 10px; border-radius: 4px; font-family: monospace; white-space: pre-wrap;">';
                html += escapeHtml(pythonResult.output);
                html += '</div>';
            }
            
            // 結果変数
            if (pythonResult.result) {
                html += '<div class="python-result-value mb-2" style="background: #e3f2fd; padding: 10px; border-radius: 4px;">';
                html += '<strong>結果:</strong> ' + escapeHtml(pythonResult.result);
                html += '</div>';
            }
            
            // グラフ（base64エンコードされた画像）
            if (pythonResult.plots && Array.isArray(pythonResult.plots) && pythonResult.plots.length > 0) {
                pythonResult.plots.forEach((plotBase64, plotIndex) => {
                    html += '<div class="python-plot mt-2">';
                    html += '<div class="d-flex justify-content-between align-items-center mb-2">';
                    html += '<p class="text-muted mb-0"><i class="bi bi-graph-up"></i> Python グラフ ' + (plotIndex + 1) + '</p>';
                    html += '<button class="btn btn-sm btn-outline-primary download-btn" onclick="downloadBase64Image(\'' + plotBase64 + '\', \'python_plot_' + (plotIndex + 1) + '.png\')">';
                    html += '<i class="bi bi-download"></i> PNG</button>';
                    html += '</div>';
                    html += '<img src="data:image/png;base64,' + plotBase64 + '" style="max-width: 100%; height: auto; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);" />';
                    html += '</div>';
                });
            }
            
            html += '</div>';
        });
    }
    
    // コンテンツをセット（既存の内容をクリア）
    contentDiv.innerHTML = html;
}

function addUserMessage(text) {
    const messagesDiv = document.getElementById('chatMessages');
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message clearfix';
    
    // 現在時刻を取得
    const now = new Date().toISOString();
    const jstTime = formatToJST(now);
    
    messageDiv.innerHTML = `
        <div class="message-wrapper-user">
            <div class="message-timestamp-outside">${jstTime}</div>
            <div class="message-user">${escapeHtml(text)}</div>
        </div>
    `;
    messagesDiv.appendChild(messageDiv);
    scrollToBottom();
}

function addUserMessageWithTimestamp(text, timestamp) {
    const messagesDiv = document.getElementById('chatMessages');
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message clearfix';
    
    // 日本時間でフォーマット
    const jstTime = formatToJST(timestamp);
    
    messageDiv.innerHTML = `
        <div class="message-wrapper-user">
            <div class="message-timestamp-outside">${jstTime}</div>
            <div class="message-user">${escapeHtml(text)}</div>
        </div>
    `;
    messagesDiv.appendChild(messageDiv);
    scrollToBottom();
}

function addErrorMessage(error) {
    const messagesDiv = document.getElementById('chatMessages');
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message';
    messageDiv.innerHTML = `
        <div class="message-error">
            <h5><i class="bi bi-exclamation-triangle"></i> エラー</h5>
            <div>${escapeHtml(error)}</div>
        </div>
    `;
    messagesDiv.appendChild(messageDiv);
    scrollToBottom();
}

// Store table data for downloads
const tableDataStore = {};

function createDataTable(data) {
    if (!data || data.length === 0) return '';
    
    const tableId = 'table-' + Date.now();
    tableDataStore[tableId] = data;
    
    const keys = Object.keys(data[0]);
    const maxRows = 100;
    const displayData = data.slice(0, maxRows);
    
    let html = `<div class="data-table">`;
    html += `<div class="d-flex justify-content-between align-items-center mb-3">`;
    html += `<div>`;
    html += `<p class="text-muted mb-0"><i class="bi bi-table"></i> クエリ結果</p>`;
    html += `<small class="text-muted">全${data.length.toLocaleString()}行（表示: ${Math.min(displayData.length, data.length).toLocaleString()}行）</small>`;
    html += `</div>`;
    html += `<button class="btn btn-sm btn-primary download-btn" onclick="downloadCSV(tableDataStore['${tableId}'], 'bigquery_result.csv')" title="全${data.length.toLocaleString()}行をCSVダウンロード">`;
    html += `<i class="bi bi-download"></i> CSV ダウンロード (${data.length.toLocaleString()}行)</button>`;
    html += `</div>`;
    html += `<div class="table-responsive">`;
    html += `<table class="table table-sm table-striped table-hover">`;
    html += `<thead class="table-dark"><tr>`;
    keys.forEach(key => {
        html += `<th>${escapeHtml(key)}</th>`;
    });
    html += `</tr></thead><tbody>`;
    
    displayData.forEach(row => {
        html += `<tr>`;
        keys.forEach(key => {
            html += `<td>${escapeHtml(String(row[key] ?? ''))}</td>`;
        });
        html += `</tr>`;
    });
    
    html += `</tbody></table>`;
    html += `</div>`;
    
    if (data.length > maxRows) {
        html += `<div class="alert alert-info mt-2 mb-0">`;
        html += `<i class="bi bi-info-circle"></i> `;
        html += `画面には最初の${maxRows}行のみ表示していますが、<strong>CSVダウンロードでは全${data.length.toLocaleString()}行</strong>が含まれます。`;
        html += `</div>`;
    }
    
    html += `</div>`;
    return html;
}

// Global counter for unique chart IDs
let chartIdCounter = 0;

function createChart(data, chartConfig = null) {
    if (!data || data.length === 0) return '';
    
    // AIが明示的に"none"を指定した場合はグラフを表示しない
    if (chartConfig && chartConfig.chart_type === 'none') {
        return '';
    }
    
    // グラフ設定がない場合、デフォルトの棒グラフを表示（50行以下のみ）
    if (!chartConfig) {
        if (data.length > 50) return '';
        
        const keys = Object.keys(data[0]);
        if (keys.length < 2) return '';
        
        chartConfig = {
            chart_type: 'bar',
            x_axis: keys[0],
            y_axis: keys.find(k => !isNaN(parseFloat(data[0][k]))) || keys[1],
            title: 'データ可視化'
        };
    }
    
    const chartId = 'chart-' + Date.now() + '-' + (chartIdCounter++);
    const chartType = chartConfig.chart_type;
    const xAxis = chartConfig.x_axis;
    const yAxis = chartConfig.y_axis;
    const title = chartConfig.title || 'グラフ表示';
    
    // Store chart configuration for later initialization
    const chartData = {
        id: chartId,
        data: data,
        chartType: chartType,
        xAxis: xAxis,
        yAxis: yAxis,
        title: title
    };
    
    // Initialize chart after DOM is ready
    setTimeout(() => initializeChart(chartData), 100);
    
    const iconMap = {
        'bar': 'bi-bar-chart-fill',
        'line': 'bi-graph-up',
        'pie': 'bi-pie-chart-fill',
        'doughnut': 'bi-pie-chart',
        'scatter': 'bi-diagram-3'
    };
    const icon = iconMap[chartType] || 'bi-bar-chart-fill';
    
    return `
        <div class="chart-container">
            <div class="d-flex justify-content-between align-items-center mb-2">
                <p class="text-muted mb-0"><i class="bi ${icon}"></i> ${title}</p>
                <button class="btn btn-sm btn-outline-primary download-btn" onclick="downloadChartAsPNG('${chartId}', '${title.replace(/[^a-zA-Z0-9]/g, '_')}.png')">
                    <i class="bi bi-download"></i> PNG
                </button>
            </div>
            <canvas id="${chartId}"></canvas>
        </div>
    `;
}

function initializeChart(chartData) {
    const canvas = document.getElementById(chartData.id);
    if (!canvas) {
        console.warn('Canvas not found for chart:', chartData.id);
        return;
    }
    
    const { data, chartType, xAxis, yAxis, title } = chartData;
    
    const labels = data.map(row => String(row[xAxis] ?? ''));
    const values = data.map(row => parseFloat(row[yAxis]) || 0);
    
    // グラフタイプに応じた設定
    let chartConfig = {};
    let chartOptions = {
            responsive: true,
            maintainAspectRatio: true,
            plugins: {
                legend: {
                    display: true
                },
                title: {
                    display: true,
                    text: title,
                    font: {
                        size: 16
                    }
                }
            }
        };
        
    if (chartType === 'pie' || chartType === 'doughnut') {
        // 円グラフ・ドーナツグラフ
        chartConfig = {
            labels: labels,
            datasets: [{
                label: yAxis,
                data: values,
                backgroundColor: [
                    'rgba(255, 99, 132, 0.7)',
                    'rgba(54, 162, 235, 0.7)',
                    'rgba(255, 206, 86, 0.7)',
                    'rgba(75, 192, 192, 0.7)',
                    'rgba(153, 102, 255, 0.7)',
                    'rgba(255, 159, 64, 0.7)',
                    'rgba(199, 199, 199, 0.7)',
                    'rgba(83, 102, 255, 0.7)',
                    'rgba(255, 99, 255, 0.7)',
                    'rgba(99, 255, 132, 0.7)'
                ],
                borderWidth: 1
            }]
        };
    } else if (chartType === 'line') {
        // 折れ線グラフ
        chartConfig = {
            labels: labels,
            datasets: [{
                label: yAxis,
                data: values,
                backgroundColor: 'rgba(75, 192, 192, 0.2)',
                borderColor: 'rgba(75, 192, 192, 1)',
                borderWidth: 2,
                fill: true,
                tension: 0.3
            }]
        };
        chartOptions.scales = {
            y: {
                beginAtZero: true
            }
        };
    } else if (chartType === 'scatter') {
        // 散布図
        const scatterData = data.map(row => ({
            x: parseFloat(row[xAxis]) || 0,
            y: parseFloat(row[yAxis]) || 0
        }));
        chartConfig = {
            datasets: [{
                label: `${xAxis} vs ${yAxis}`,
                data: scatterData,
                backgroundColor: 'rgba(255, 99, 132, 0.5)',
                borderColor: 'rgba(255, 99, 132, 1)',
                borderWidth: 1
            }]
        };
        chartOptions.scales = {
            x: {
                type: 'linear',
                position: 'bottom',
                title: {
                    display: true,
                    text: xAxis
                }
            },
            y: {
                title: {
                    display: true,
                    text: yAxis
                }
            }
        };
    } else {
        // 棒グラフ（デフォルト）
        chartConfig = {
            labels: labels,
            datasets: [{
                label: yAxis,
                data: values,
                backgroundColor: 'rgba(54, 162, 235, 0.5)',
                borderColor: 'rgba(54, 162, 235, 1)',
                borderWidth: 1
            }]
        };
        chartOptions.scales = {
            y: {
                beginAtZero: true
            }
        };
    }
        
    new Chart(canvas, {
        type: chartType === 'scatter' ? 'scatter' : (chartType === 'doughnut' ? 'doughnut' : (chartType === 'pie' ? 'pie' : (chartType === 'line' ? 'line' : 'bar'))),
        data: chartConfig,
        options: chartOptions
    });
}

function formatAnswer(text) {
    if (!text) return '';
    
    text = escapeHtml(text);
    text = text.replace(/\n/g, '<br>');
    text = text.replace(/```sql([\s\S]*?)```/g, '<pre class="bg-light p-2 rounded"><code>$1</code></pre>');
    text = text.replace(/```([\s\S]*?)```/g, '<pre class="bg-light p-2 rounded"><code>$1</code></pre>');
    text = text.replace(/`([^`]+)`/g, '<code class="bg-light px-1">$1</code>');
    
    return text;
}

function escapeHtml(text) {
    const map = {
        '&': '&amp;',
        '<': '&lt;',
        '>': '&gt;',
        '"': '&quot;',
        "'": '&#039;'
    };
    return String(text).replace(/[&<>"']/g, m => map[m]);
}

function scrollToBottom() {
    const messagesDiv = document.getElementById('chatMessages');
    messagesDiv.scrollTop = messagesDiv.scrollHeight;
}

// ============================================
// Chat Session Management
// ============================================

async function loadChatSessions() {
    const container = document.getElementById('chatHistoryContainer');
    if (!container) return;  // Not on chat page
    
    try {
        const response = await fetch('/api/chat-sessions');
        const data = await response.json();
        
        if (data.success && data.sessions && data.sessions.length > 0) {
            renderChatSessions(data.sessions);
        } else {
            container.innerHTML = '<p class="text-muted text-center small">チャット履歴がありません</p>';
        }
    } catch (error) {
        console.error('Error loading chat sessions:', error);
        container.innerHTML = '<p class="text-danger text-center small">読み込みエラー</p>';
    }
}

function renderChatSessions(sessions) {
    const container = document.getElementById('chatHistoryContainer');
    
    container.innerHTML = sessions.map(session => {
        const isActive = session.id === currentSessionId;
        const date = new Date(session.updated_at);
        const dateStr = formatDateRelative(date);
        const escapedTitle = session.title.replace(/\\/g, '\\\\').replace(/'/g, "\\'");
        
        return `
            <div class="chat-session-item ${isActive ? 'active' : ''}" data-session-id="${session.id}" data-session-title="${escapeHtml(session.title)}">
                <a href="/agent-chat/${session.id}" class="session-link text-decoration-none">
                    <div class="d-flex align-items-start">
                        <i class="bi bi-chat-left-text me-2 text-muted"></i>
                        <div class="flex-grow-1" style="min-width: 0;">
                            <div class="session-title text-truncate">${escapeHtml(session.title)}</div>
                            <div class="session-meta text-muted small">${dateStr}</div>
                        </div>
                        ${isActive ? '<i class="bi bi-check-circle-fill text-success"></i>' : ''}
                    </div>
                </a>
                <div class="session-actions">
                    <button class="btn btn-sm btn-link p-0 me-2 rename-btn" data-session-id="${session.id}" title="名前を変更">
                        <i class="bi bi-pencil"></i>
                    </button>
                    <button class="btn btn-sm btn-link p-0 text-danger delete-btn" data-session-id="${session.id}" title="削除">
                        <i class="bi bi-trash"></i>
                    </button>
                </div>
            </div>
        `;
    }).join('');
    
    // Add event listeners after rendering
    container.querySelectorAll('.rename-btn').forEach(btn => {
        btn.addEventListener('click', (e) => {
            e.stopPropagation();
            const sessionId = parseInt(btn.dataset.sessionId);
            const sessionItem = btn.closest('.chat-session-item');
            const currentTitle = sessionItem.dataset.sessionTitle;
            renameSession(sessionId, currentTitle);
        });
    });
    
    container.querySelectorAll('.delete-btn').forEach(btn => {
        btn.addEventListener('click', (e) => {
            e.stopPropagation();
            const sessionId = parseInt(btn.dataset.sessionId);
            deleteSession(sessionId);
        });
    });
}

async function loadCurrentSessionMessages() {
    try {
        if (!currentSessionId) return;
        
        // Clear current chat
        const messagesDiv = document.getElementById('chatMessages');
        messagesDiv.innerHTML = '';
        conversationHistory = [];
        
        // Load messages
        const response = await fetch(`/api/chat-sessions/${currentSessionId}/messages`);
        const data = await response.json();
        
        if (data.success && data.messages) {
            // Render messages
            data.messages.forEach(msg => {
                // Add user message with timestamp
                addUserMessageWithTimestamp(msg.user_message, msg.created_at);
                
                // Add AI response
                if (msg.ai_response) {
                    const assistantDiv = createAssistantMessageDiv(msg.created_at);
                    messagesDiv.appendChild(assistantDiv);
                    
                    const msgContainer = assistantDiv.querySelector('.assistant-content');
                    if (msgContainer) {
                        let html = '';
                        
                        // Add reasoning process if available
                        if (msg.reasoning_process && msg.reasoning_process.trim()) {
                            html += `
                                <div class="reasoning-process">
                                    <div class="reasoning-header">
                                        <i class="bi bi-lightbulb"></i> <strong>推論過程:</strong>
                                    </div>
                                    <div class="reasoning-content">${escapeHtml(msg.reasoning_process)}</div>
                                </div>
                            `;
                        }
                        
                        html += `<div class="answer-text mb-3">${formatAnswer(msg.ai_response)}</div>`;
                        
                        // Render data table and charts if query result exists
                        if (msg.query_result) {
                            try {
                                const result = typeof msg.query_result === 'string' 
                                    ? JSON.parse(msg.query_result) 
                                    : msg.query_result;
                                
                                // Add data table
                                if (result && result.data && Array.isArray(result.data) && result.data.length > 0) {
                                    html += createDataTable(result.data);
                                    
                                    // Add charts
                                    if (result.charts && Array.isArray(result.charts) && result.charts.length > 0) {
                                        result.charts.forEach(chartConfig => {
                                            html += createChart(result.data, chartConfig);
                                        });
                                    }
                                }
                            } catch (e) {
                                console.error('Error parsing query result:', e, msg.query_result);
                            }
                        }
                        
                        // Add processing metadata
                        if (msg.steps_count || msg.processing_time) {
                            html += createProcessingMetadata(msg.steps_count, msg.processing_time);
                        }
                        
                        msgContainer.innerHTML = html;
                    }
                }
                
                // Update conversation history
                conversationHistory.push({
                    role: 'user',
                    content: msg.user_message
                });
                if (msg.ai_response) {
                    conversationHistory.push({
                        role: 'assistant',
                        content: msg.ai_response
                    });
                }
            });
            
            scrollToBottom();
        }
    } catch (error) {
        console.error('Error loading chat session messages:', error);
    }
}

async function deleteSession(sessionId) {
    if (!confirm('この会話を削除してもよろしいですか？')) {
        return;
    }
    
    try {
        const response = await fetch(`/api/chat-sessions/${sessionId}`, {
            method: 'DELETE'
        });
        
        const data = await response.json();
        
        if (data.success) {
            // If we deleted the current session, redirect to main chat page
            if (sessionId === currentSessionId) {
                window.location.href = '/agent-chat';
            } else {
                // Just refresh the session list
                loadChatSessions();
            }
        } else {
            alert('削除に失敗しました: ' + (data.error || '不明なエラー'));
        }
    } catch (error) {
        console.error('Error deleting session:', error);
        alert('削除中にエラーが発生しました');
    }
}

async function renameSession(sessionId, currentTitle) {
    const newTitle = prompt('新しい会話名を入力してください:', currentTitle);
    
    if (!newTitle || newTitle === currentTitle) {
        return;
    }
    
    try {
        const response = await fetch(`/api/chat-sessions/${sessionId}/title`, {
            method: 'PUT',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ title: newTitle })
        });
        
        const data = await response.json();
        
        if (data.success) {
            // Refresh the session list to show new title
            loadChatSessions();
        } else {
            alert('名前変更に失敗しました: ' + (data.error || '不明なエラー'));
        }
    } catch (error) {
        console.error('Error renaming session:', error);
        alert('名前変更中にエラーが発生しました');
    }
}

async function createNewChat() {
    try {
        const response = await fetch('/api/chat-sessions', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            }
        });
        
        const data = await response.json();
        
        if (data.success && data.session) {
            // Redirect to the new session
            window.location.href = `/agent-chat/${data.session.id}`;
        } else {
            alert('新しいチャットの作成に失敗しました: ' + (data.error || '不明なエラー'));
        }
    } catch (error) {
        console.error('Error creating new chat:', error);
        alert('新しいチャットの作成中にエラーが発生しました');
    }
}

function formatDateRelative(date) {
    const now = new Date();
    const diffMs = now - date;
    const diffMins = Math.floor(diffMs / 60000);
    const diffHours = Math.floor(diffMs / 3600000);
    const diffDays = Math.floor(diffMs / 86400000);
    
    if (diffMins < 1) return '今';
    if (diffMins < 60) return `${diffMins}分前`;
    if (diffHours < 24) return `${diffHours}時間前`;
    if (diffDays < 7) return `${diffDays}日前`;
    
    return date.toLocaleDateString('ja-JP', { month: 'short', day: 'numeric' });
}

function formatToJST(timestamp) {
    if (!timestamp) return '';
    
    const date = new Date(timestamp);
    // 日本時間（JST = UTC+9）でフォーマット
    const options = {
        timeZone: 'Asia/Tokyo',
        year: 'numeric',
        month: '2-digit',
        day: '2-digit',
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit',
        hour12: false
    };
    
    const formatter = new Intl.DateTimeFormat('ja-JP', options);
    const parts = formatter.formatToParts(date);
    
    // 年-月-日 時:分:秒の形式で組み立て
    const year = parts.find(p => p.type === 'year').value;
    const month = parts.find(p => p.type === 'month').value;
    const day = parts.find(p => p.type === 'day').value;
    const hour = parts.find(p => p.type === 'hour').value;
    const minute = parts.find(p => p.type === 'minute').value;
    const second = parts.find(p => p.type === 'second').value;
    
    return `${year}-${month}-${day} ${hour}:${minute}:${second}`;
}

function createProcessingMetadata(stepsCount, processingTime) {
    const timeStr = processingTime ? `${processingTime.toFixed(2)}秒` : '不明';
    const stepsStr = stepsCount || 0;
    
    return `
        <div class="processing-metadata mt-3">
            <div class="metadata-item">
                <i class="bi bi-list-check"></i>
                <span>実行ステップ: <strong>${stepsStr}</strong></span>
            </div>
            <div class="metadata-item">
                <i class="bi bi-clock-history"></i>
                <span>処理時間: <strong>${timeStr}</strong></span>
            </div>
        </div>
    `;
}

// ============================================
// Download Utilities
// ============================================

function downloadCSV(data, filename = 'data.csv') {
    if (!data || data.length === 0) {
        alert('ダウンロードするデータがありません');
        return;
    }
    
    const keys = Object.keys(data[0]);
    let csv = keys.join(',') + '\n';
    
    data.forEach(row => {
        const values = keys.map(key => {
            const value = row[key] ?? '';
            // CSVエスケープ処理
            const stringValue = String(value);
            if (stringValue.includes(',') || stringValue.includes('"') || stringValue.includes('\n')) {
                return '"' + stringValue.replace(/"/g, '""') + '"';
            }
            return stringValue;
        });
        csv += values.join(',') + '\n';
    });
    
    // UTF-8 BOMを追加（Excelで日本語を正しく表示するため）
    const BOM = '\uFEFF';
    const csvWithBOM = BOM + csv;
    
    // タイムスタンプを追加したファイル名を生成
    const timestamp = new Date().toISOString().slice(0, 19).replace(/:/g, '-');
    const filenameWithTimestamp = filename.replace('.csv', `_${timestamp}.csv`);
    
    const blob = new Blob([csvWithBOM], { type: 'text/csv;charset=utf-8;' });
    const link = document.createElement('a');
    link.href = URL.createObjectURL(blob);
    link.download = filenameWithTimestamp;
    link.click();
    URL.revokeObjectURL(link.href);
    
    // ダウンロード成功のフィードバック
    console.log(`CSV downloaded: ${filenameWithTimestamp} (${data.length}行)`);
}

function downloadChartAsPNG(canvasId, filename = 'chart.png') {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;
    
    canvas.toBlob(function(blob) {
        const link = document.createElement('a');
        link.href = URL.createObjectURL(blob);
        link.download = filename;
        link.click();
        URL.revokeObjectURL(link.href);
    });
}

function downloadBase64Image(base64Data, filename = 'plot.png') {
    const link = document.createElement('a');
    link.href = 'data:image/png;base64,' + base64Data;
    link.download = filename;
    link.click();
}

// Event listener for new chat button
document.addEventListener('DOMContentLoaded', function() {
    const newChatBtn = document.getElementById('newChatBtn');
    if (newChatBtn) {
        newChatBtn.addEventListener('click', createNewChat);
    }
});
