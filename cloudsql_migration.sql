-- ================================================================
-- CloudSQL (PostgreSQL) Migration Script
-- BigQuery AI Data Analysis Agent
-- ================================================================
-- 
-- このスクリプトは既存のデータベーススキーマをCloudSQL (PostgreSQL 15)に
-- マイグレーションするために使用します。
--
-- 実行方法:
-- psql -h <CLOUDSQL_IP> -U <USERNAME> -d <DATABASE_NAME> -f cloudsql_migration.sql
--
-- または Cloud SQL Proxy経由:
-- psql "host=127.0.0.1 port=5432 dbname=bigquery_ai user=postgres" -f cloudsql_migration.sql
-- ================================================================

-- トランザクション開始
BEGIN;

-- ================================================================
-- 1. Users Table
-- ================================================================
-- ユーザー認証情報を格納するテーブル
CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(100) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ================================================================
-- 2. Projects Table
-- ================================================================
-- BigQueryプロジェクトの設定情報を格納するテーブル
-- 各ユーザーは複数のプロジェクトを持つことができる
CREATE TABLE IF NOT EXISTS projects (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    openai_api_key TEXT,
    bigquery_project_id VARCHAR(255),
    bigquery_dataset_id VARCHAR(255),
    service_account_json TEXT,
    is_active BOOLEAN DEFAULT false,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ================================================================
-- 3. Chat Sessions Table
-- ================================================================
-- プロジェクトごとのチャットセッションを格納するテーブル
-- ChatGPTのような会話履歴の管理に使用
CREATE TABLE IF NOT EXISTS chat_sessions (
    id SERIAL PRIMARY KEY,
    project_id INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    title VARCHAR(500) DEFAULT 'New Chat',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ================================================================
-- 4. Chat History Table
-- ================================================================
-- セッションごとの会話履歴を格納するテーブル
-- ユーザーメッセージ、AI応答、クエリ結果を保存
CREATE TABLE IF NOT EXISTS chat_history (
    id SERIAL PRIMARY KEY,
    project_id INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    session_id INTEGER NOT NULL REFERENCES chat_sessions(id) ON DELETE CASCADE,
    user_message TEXT NOT NULL,
    ai_response TEXT,
    query_result JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ================================================================
-- 5. Indexes for Performance Optimization
-- ================================================================

-- Users table indexes
CREATE INDEX IF NOT EXISTS idx_users_username ON users(username);
CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);

-- Projects table indexes
CREATE INDEX IF NOT EXISTS idx_projects_user_id ON projects(user_id);
CREATE INDEX IF NOT EXISTS idx_projects_is_active ON projects(user_id, is_active);

-- Chat sessions table indexes
CREATE INDEX IF NOT EXISTS idx_chat_sessions_project_id ON chat_sessions(project_id, updated_at DESC);

-- Chat history table indexes
CREATE INDEX IF NOT EXISTS idx_chat_history_project_id ON chat_history(project_id);
CREATE INDEX IF NOT EXISTS idx_chat_history_session_id ON chat_history(session_id, created_at ASC);
CREATE INDEX IF NOT EXISTS idx_chat_history_created_at ON chat_history(project_id, created_at DESC);

-- ================================================================
-- 6. Additional Constraints and Comments
-- ================================================================

-- テーブルコメント
COMMENT ON TABLE users IS 'ユーザー認証情報';
COMMENT ON TABLE projects IS 'BigQueryプロジェクト設定（ユーザーごと）';
COMMENT ON TABLE chat_sessions IS 'チャットセッション（プロジェクトごと）';
COMMENT ON TABLE chat_history IS '会話履歴（セッションごと）';

-- カラムコメント - Users
COMMENT ON COLUMN users.id IS 'ユーザーID（主キー）';
COMMENT ON COLUMN users.username IS 'ユーザー名（一意）';
COMMENT ON COLUMN users.password_hash IS 'パスワードハッシュ（bcrypt）';
COMMENT ON COLUMN users.email IS 'メールアドレス（一意）';
COMMENT ON COLUMN users.created_at IS 'アカウント作成日時';

-- カラムコメント - Projects
COMMENT ON COLUMN projects.id IS 'プロジェクトID（主キー）';
COMMENT ON COLUMN projects.user_id IS 'ユーザーID（外部キー）';
COMMENT ON COLUMN projects.name IS 'プロジェクト名';
COMMENT ON COLUMN projects.description IS 'プロジェクト説明';
COMMENT ON COLUMN projects.openai_api_key IS 'OpenAI APIキー';
COMMENT ON COLUMN projects.bigquery_project_id IS 'BigQuery プロジェクトID';
COMMENT ON COLUMN projects.bigquery_dataset_id IS 'BigQuery データセットID';
COMMENT ON COLUMN projects.service_account_json IS 'GCPサービスアカウントJSON';
COMMENT ON COLUMN projects.is_active IS 'アクティブフラグ（ユーザーごとに1つのみ）';
COMMENT ON COLUMN projects.created_at IS 'プロジェクト作成日時';
COMMENT ON COLUMN projects.updated_at IS '最終更新日時';

-- カラムコメント - Chat Sessions
COMMENT ON COLUMN chat_sessions.id IS 'セッションID（主キー）';
COMMENT ON COLUMN chat_sessions.project_id IS 'プロジェクトID（外部キー）';
COMMENT ON COLUMN chat_sessions.title IS 'セッションタイトル';
COMMENT ON COLUMN chat_sessions.created_at IS 'セッション作成日時';
COMMENT ON COLUMN chat_sessions.updated_at IS '最終更新日時';

-- カラムコメント - Chat History
COMMENT ON COLUMN chat_history.id IS '履歴ID（主キー）';
COMMENT ON COLUMN chat_history.project_id IS 'プロジェクトID（外部キー）';
COMMENT ON COLUMN chat_history.session_id IS 'セッションID（外部キー）';
COMMENT ON COLUMN chat_history.user_message IS 'ユーザーメッセージ';
COMMENT ON COLUMN chat_history.ai_response IS 'AI応答';
COMMENT ON COLUMN chat_history.query_result IS 'クエリ結果（JSON形式）';
COMMENT ON COLUMN chat_history.created_at IS 'メッセージ作成日時';

-- トランザクションコミット
COMMIT;

-- ================================================================
-- マイグレーション完了
-- ================================================================

-- テーブル一覧を確認
SELECT 
    schemaname,
    tablename,
    tableowner
FROM pg_tables 
WHERE schemaname = 'public'
ORDER BY tablename;

-- インデックス一覧を確認
SELECT
    tablename,
    indexname,
    indexdef
FROM pg_indexes
WHERE schemaname = 'public'
ORDER BY tablename, indexname;
