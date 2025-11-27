-- BigQuery AI Data Analysis Agent - CloudSQL Full Migration
-- This migration creates all tables from scratch for Google Cloud SQL (PostgreSQL)
-- Run this on a fresh database to recreate the complete schema

-- ============================================
-- 1. USERS TABLE
-- ============================================
CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(100) NOT NULL UNIQUE,
    password_hash VARCHAR(255) NOT NULL,
    email VARCHAR(255),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Index for username lookups
CREATE INDEX IF NOT EXISTS idx_users_username ON users(username);

-- ============================================
-- 2. PROJECTS TABLE
-- ============================================
CREATE TABLE IF NOT EXISTS projects (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    openai_api_key TEXT,
    gemini_api_key TEXT,
    ai_provider VARCHAR(50) DEFAULT 'openai',
    bigquery_project_id VARCHAR(255),
    bigquery_dataset_id VARCHAR(255),
    service_account_json TEXT,
    original_json_filename VARCHAR(255),
    use_env_json BOOLEAN DEFAULT FALSE,
    is_active BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for projects
CREATE INDEX IF NOT EXISTS idx_projects_user_id ON projects(user_id);
CREATE INDEX IF NOT EXISTS idx_projects_is_active ON projects(user_id, is_active);

-- ============================================
-- 3. CHAT SESSIONS TABLE
-- ============================================
CREATE TABLE IF NOT EXISTS chat_sessions (
    id SERIAL PRIMARY KEY,
    project_id INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    title VARCHAR(500) DEFAULT 'New Chat',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Index for session lookups by project
CREATE INDEX IF NOT EXISTS idx_chat_sessions_project_id ON chat_sessions(project_id, updated_at DESC);

-- ============================================
-- 4. CHAT HISTORY TABLE
-- ============================================
CREATE TABLE IF NOT EXISTS chat_history (
    id SERIAL PRIMARY KEY,
    project_id INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    session_id INTEGER NOT NULL REFERENCES chat_sessions(id) ON DELETE CASCADE,
    user_message TEXT NOT NULL,
    ai_response TEXT,
    query_result JSONB,
    reasoning_process TEXT,
    steps_count INTEGER DEFAULT 0,
    processing_time DOUBLE PRECISION DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for chat history
CREATE INDEX IF NOT EXISTS idx_chat_history_project_id ON chat_history(project_id);
CREATE INDEX IF NOT EXISTS idx_chat_history_session_id ON chat_history(session_id, created_at);
CREATE INDEX IF NOT EXISTS idx_chat_history_created_at ON chat_history(project_id, created_at DESC);

-- ============================================
-- 5. PROJECT MEMORIES TABLE
-- ============================================
CREATE TABLE IF NOT EXISTS project_memories (
    id SERIAL PRIMARY KEY,
    project_id INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    memory_key VARCHAR(255) NOT NULL,
    memory_value TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for project memories
CREATE INDEX IF NOT EXISTS idx_project_memories_project_id ON project_memories(project_id);
CREATE INDEX IF NOT EXISTS idx_project_memories_user_id ON project_memories(user_id);

-- Unique constraint to prevent duplicate memory keys per project
CREATE UNIQUE INDEX IF NOT EXISTS idx_project_memories_unique_key 
ON project_memories(project_id, memory_key);

-- ============================================
-- 6. CHAT TASKS TABLE (Cloud Run Compatible)
-- ============================================
-- Stores running chat task status for Cloud Run compatibility (stateless)
-- This replaces the in-memory task dictionary to support multiple instances
CREATE TABLE IF NOT EXISTS chat_tasks (
    task_id VARCHAR(36) PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    project_id INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    status VARCHAR(20) NOT NULL DEFAULT 'running',
    steps JSONB DEFAULT '[]'::jsonb,
    result JSONB,
    error TEXT,
    reasoning TEXT DEFAULT '',
    cancelled BOOLEAN DEFAULT false,
    session_id INTEGER,
    traceback TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for chat tasks
CREATE INDEX IF NOT EXISTS idx_chat_tasks_user_id ON chat_tasks(user_id);
CREATE INDEX IF NOT EXISTS idx_chat_tasks_status ON chat_tasks(status);
CREATE INDEX IF NOT EXISTS idx_chat_tasks_created_at ON chat_tasks(created_at);

-- ============================================
-- COMMENTS
-- ============================================
COMMENT ON TABLE users IS 'User accounts for authentication';
COMMENT ON TABLE projects IS 'BigQuery projects with API keys and configuration';
COMMENT ON TABLE chat_sessions IS 'Chat sessions for organizing conversations';
COMMENT ON TABLE chat_history IS 'Individual chat messages and AI responses';
COMMENT ON TABLE project_memories IS 'Persistent memory storage for AI context per project';
COMMENT ON TABLE chat_tasks IS 'Running chat task status for Cloud Run compatibility (stateless architecture)';

COMMENT ON COLUMN projects.use_env_json IS 'When true, use Application Default Credentials (ADC) instead of uploaded JSON file';
COMMENT ON COLUMN projects.ai_provider IS 'AI provider to use: openai or gemini';
COMMENT ON COLUMN projects.original_json_filename IS 'Original filename of uploaded service account JSON';
COMMENT ON COLUMN chat_history.reasoning_process IS 'AI reasoning/thinking process for transparency';
COMMENT ON COLUMN chat_history.steps_count IS 'Number of steps taken by AI agent';
COMMENT ON COLUMN chat_history.processing_time IS 'Total processing time in seconds';
COMMENT ON COLUMN chat_tasks.status IS 'Task status: running, completed, error, cancelled';
COMMENT ON COLUMN chat_tasks.steps IS 'Array of progress step messages';
