let projects = [];
let activeProject = null;

document.addEventListener('DOMContentLoaded', () => {
    loadProjects();
    setupEventListeners();
});

function setupEventListeners() {
    document.getElementById('createProjectBtn').addEventListener('click', createProject);
    document.getElementById('updateProjectBtn').addEventListener('click', updateProject);
    document.getElementById('confirmDeleteBtn').addEventListener('click', confirmDelete);
}

async function loadProjects() {
    try {
        const response = await fetch('/api/projects');
        const data = await response.json();
        
        if (data.success) {
            projects = data.projects;
            renderProjects();
            loadActiveProject();
        } else {
            showError('プロジェクトの読み込みに失敗しました');
        }
    } catch (error) {
        console.error('Error loading projects:', error);
        showError('プロジェクトの読み込み中にエラーが発生しました');
    }
}

async function loadActiveProject() {
    try {
        const response = await fetch('/api/projects/active');
        const data = await response.json();
        
        if (data.success && data.project) {
            activeProject = data.project;
            document.getElementById('activeProjectAlert').style.display = 'block';
            document.getElementById('activeProjectName').textContent = data.project.name;
        } else {
            document.getElementById('activeProjectAlert').style.display = 'none';
        }
    } catch (error) {
        console.error('Error loading active project:', error);
    }
}

function renderProjects() {
    const container = document.getElementById('projectsContainer');
    
    if (projects.length === 0) {
        container.innerHTML = `
            <div class="text-center py-5">
                <i class="bi bi-folder-x" style="font-size: 4rem; color: #ccc;"></i>
                <p class="mt-3 text-muted">プロジェクトがありません</p>
                <p class="text-muted">「新規プロジェクト」ボタンからプロジェクトを作成してください</p>
            </div>
        `;
        return;
    }
    
    container.innerHTML = `
        <div class="row">
            ${projects.map(project => `
                <div class="col-md-6 col-lg-4 mb-4">
                    <div class="card h-100 ${project.is_active ? 'border-primary' : ''}">
                        <div class="card-body">
                            <div class="d-flex justify-content-between align-items-start mb-2">
                                <h5 class="card-title">
                                    <i class="bi bi-folder"></i> ${escapeHtml(project.name)}
                                    ${project.is_active ? '<span class="badge bg-primary ms-2">アクティブ</span>' : ''}
                                </h5>
                                <div class="dropdown">
                                    <button class="btn btn-sm btn-link text-secondary" type="button" data-bs-toggle="dropdown">
                                        <i class="bi bi-three-dots-vertical"></i>
                                    </button>
                                    <ul class="dropdown-menu dropdown-menu-end">
                                        ${!project.is_active ? `
                                            <li><a class="dropdown-item" href="#" onclick="activateProject(${project.id}); return false;">
                                                <i class="bi bi-check-circle"></i> アクティブ化
                                            </a></li>
                                        ` : ''}
                                        <li><a class="dropdown-item" href="#" onclick="editProject(${project.id}); return false;">
                                            <i class="bi bi-pencil"></i> 編集
                                        </a></li>
                                        <li><hr class="dropdown-divider"></li>
                                        <li><a class="dropdown-item text-danger" href="#" onclick="deleteProject(${project.id}, '${escapeHtml(project.name)}'); return false;">
                                            <i class="bi bi-trash"></i> 削除
                                        </a></li>
                                    </ul>
                                </div>
                            </div>
                            
                            <p class="card-text text-muted small">
                                ${project.description || '説明なし'}
                            </p>
                            
                            <div class="mt-3">
                                <small class="text-muted">
                                    ${project.bigquery_project_id ? 
                                        `<i class="bi bi-database"></i> ${escapeHtml(project.bigquery_project_id)}` : 
                                        '<i class="bi bi-exclamation-circle text-warning"></i> BigQuery未設定'
                                    }
                                </small>
                            </div>
                            
                            <div class="mt-2">
                                <small class="text-muted">
                                    作成日: ${formatDate(project.created_at)}
                                </small>
                            </div>
                        </div>
                        <div class="card-footer bg-transparent">
                            <a href="/settings?project=${project.id}" class="btn btn-sm btn-outline-primary">
                                <i class="bi bi-gear"></i> 設定
                            </a>
                        </div>
                    </div>
                </div>
            `).join('')}
        </div>
    `;
}

async function createProject() {
    const name = document.getElementById('projectName').value.trim();
    const description = document.getElementById('projectDescription').value.trim();
    
    if (!name) {
        alert('プロジェクト名を入力してください');
        return;
    }
    
    try {
        const response = await fetch('/api/projects', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ name, description })
        });
        
        const data = await response.json();
        
        if (data.success) {
            bootstrap.Modal.getInstance(document.getElementById('createProjectModal')).hide();
            document.getElementById('createProjectForm').reset();
            showSuccess(data.message);
            loadProjects();
        } else {
            showError(data.error || 'プロジェクトの作成に失敗しました');
        }
    } catch (error) {
        console.error('Error creating project:', error);
        showError('プロジェクトの作成中にエラーが発生しました');
    }
}

function editProject(projectId) {
    const project = projects.find(p => p.id === projectId);
    if (!project) return;
    
    document.getElementById('editProjectId').value = project.id;
    document.getElementById('editProjectName').value = project.name;
    document.getElementById('editProjectDescription').value = project.description || '';
    
    new bootstrap.Modal(document.getElementById('editProjectModal')).show();
}

async function updateProject() {
    const projectId = document.getElementById('editProjectId').value;
    const name = document.getElementById('editProjectName').value.trim();
    const description = document.getElementById('editProjectDescription').value.trim();
    
    if (!name) {
        alert('プロジェクト名を入力してください');
        return;
    }
    
    try {
        const response = await fetch(`/api/projects/${projectId}`, {
            method: 'PUT',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ name, description })
        });
        
        const data = await response.json();
        
        if (data.success) {
            bootstrap.Modal.getInstance(document.getElementById('editProjectModal')).hide();
            showSuccess(data.message);
            loadProjects();
        } else {
            showError(data.error || 'プロジェクトの更新に失敗しました');
        }
    } catch (error) {
        console.error('Error updating project:', error);
        showError('プロジェクトの更新中にエラーが発生しました');
    }
}

function deleteProject(projectId, projectName) {
    document.getElementById('deleteProjectId').value = projectId;
    document.getElementById('deleteProjectName').textContent = projectName;
    new bootstrap.Modal(document.getElementById('deleteProjectModal')).show();
}

async function confirmDelete() {
    const projectId = document.getElementById('deleteProjectId').value;
    
    try {
        const response = await fetch(`/api/projects/${projectId}`, {
            method: 'DELETE'
        });
        
        const data = await response.json();
        
        if (data.success) {
            bootstrap.Modal.getInstance(document.getElementById('deleteProjectModal')).hide();
            showSuccess(data.message);
            loadProjects();
        } else {
            showError(data.error || 'プロジェクトの削除に失敗しました');
        }
    } catch (error) {
        console.error('Error deleting project:', error);
        showError('プロジェクトの削除中にエラーが発生しました');
    }
}

async function activateProject(projectId) {
    try {
        const response = await fetch(`/api/projects/${projectId}/activate`, {
            method: 'POST'
        });
        
        const data = await response.json();
        
        if (data.success) {
            showSuccess(data.message);
            loadProjects();
            loadActiveProject();
        } else {
            showError(data.error || 'プロジェクトのアクティブ化に失敗しました');
        }
    } catch (error) {
        console.error('Error activating project:', error);
        showError('プロジェクトのアクティブ化中にエラーが発生しました');
    }
}

function formatDate(dateString) {
    if (!dateString) return '不明';
    const date = new Date(dateString);
    return date.toLocaleDateString('ja-JP', { 
        year: 'numeric', 
        month: '2-digit', 
        day: '2-digit',
        hour: '2-digit',
        minute: '2-digit'
    });
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function showSuccess(message) {
    const alertHtml = `
        <div class="alert alert-success alert-dismissible fade show" role="alert">
            <i class="bi bi-check-circle"></i> ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        </div>
    `;
    const container = document.querySelector('.container-fluid');
    container.insertAdjacentHTML('afterbegin', alertHtml);
    
    setTimeout(() => {
        const alert = container.querySelector('.alert');
        if (alert) alert.remove();
    }, 5000);
}

function showError(message) {
    const alertHtml = `
        <div class="alert alert-danger alert-dismissible fade show" role="alert">
            <i class="bi bi-exclamation-triangle"></i> ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        </div>
    `;
    const container = document.querySelector('.container-fluid');
    container.insertAdjacentHTML('afterbegin', alertHtml);
    
    setTimeout(() => {
        const alert = container.querySelector('.alert');
        if (alert) alert.remove();
    }, 5000);
}
