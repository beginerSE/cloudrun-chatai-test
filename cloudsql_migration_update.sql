-- ================================================================
-- CloudSQL (PostgreSQL) Update Migration Script
-- BigQuery AI Data Analysis Agent
-- ================================================================
-- 
-- このスクリプトは既存のデータベースを最新のスキーマに更新します。
-- 新規インストールにはcloudsql_migration.sqlを使用してください。
--
-- 実行方法:
-- psql -h <CLOUDSQL_IP> -U <USERNAME> -d <DATABASE_NAME> -f cloudsql_migration_update.sql
-- ================================================================

-- トランザクション開始
BEGIN;

-- ================================================================
-- 1. Projects Table - 新規カラム追加
-- ================================================================

-- ai_provider カラム追加
DO $$ 
BEGIN
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                   WHERE table_name = 'projects' AND column_name = 'ai_provider') THEN
        ALTER TABLE projects ADD COLUMN ai_provider VARCHAR(50) DEFAULT 'openai';
        COMMENT ON COLUMN projects.ai_provider IS 'AIプロバイダー（openai/gemini）';
    END IF;
END $$;

-- gemini_api_key カラム追加
DO $$ 
BEGIN
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                   WHERE table_name = 'projects' AND column_name = 'gemini_api_key') THEN
        ALTER TABLE projects ADD COLUMN gemini_api_key TEXT;
        COMMENT ON COLUMN projects.gemini_api_key IS 'Gemini APIキー';
    END IF;
END $$;

-- use_env_json カラム追加
DO $$ 
BEGIN
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                   WHERE table_name = 'projects' AND column_name = 'use_env_json') THEN
        ALTER TABLE projects ADD COLUMN use_env_json BOOLEAN DEFAULT false;
        COMMENT ON COLUMN projects.use_env_json IS '環境変数からJSON使用フラグ';
    END IF;
END $$;

-- original_json_filename カラム追加
DO $$ 
BEGIN
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                   WHERE table_name = 'projects' AND column_name = 'original_json_filename') THEN
        ALTER TABLE projects ADD COLUMN original_json_filename VARCHAR(255);
        COMMENT ON COLUMN projects.original_json_filename IS 'アップロードされたJSONファイル名';
    END IF;
END $$;

-- ================================================================
-- 2. Chat History Table - 新規カラム追加
-- ================================================================

-- steps_count カラム追加
DO $$ 
BEGIN
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                   WHERE table_name = 'chat_history' AND column_name = 'steps_count') THEN
        ALTER TABLE chat_history ADD COLUMN steps_count INTEGER DEFAULT 0;
        COMMENT ON COLUMN chat_history.steps_count IS '処理ステップ数';
    END IF;
END $$;

-- processing_time カラム追加
DO $$ 
BEGIN
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                   WHERE table_name = 'chat_history' AND column_name = 'processing_time') THEN
        ALTER TABLE chat_history ADD COLUMN processing_time REAL DEFAULT 0;
        COMMENT ON COLUMN chat_history.processing_time IS '処理時間（秒）';
    END IF;
END $$;

-- reasoning_process カラム追加
DO $$ 
BEGIN
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                   WHERE table_name = 'chat_history' AND column_name = 'reasoning_process') THEN
        ALTER TABLE chat_history ADD COLUMN reasoning_process TEXT;
        COMMENT ON COLUMN chat_history.reasoning_process IS 'AI推論過程';
    END IF;
END $$;

-- ================================================================
-- 3. Project Memories Table - 新規テーブル作成
-- ================================================================

CREATE TABLE IF NOT EXISTS project_memories (
    id SERIAL PRIMARY KEY,
    project_id INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    memory_key VARCHAR(255) NOT NULL,
    memory_value TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Project memories table indexes
CREATE INDEX IF NOT EXISTS idx_project_memories_project_id ON project_memories(project_id);
CREATE INDEX IF NOT EXISTS idx_project_memories_user_id ON project_memories(user_id);
CREATE INDEX IF NOT EXISTS idx_project_memories_key ON project_memories(project_id, memory_key);

-- テーブルコメント
COMMENT ON TABLE project_memories IS 'プロジェクト共有メモリ（AI参照用）';

-- カラムコメント
COMMENT ON COLUMN project_memories.id IS 'メモリID（主キー）';
COMMENT ON COLUMN project_memories.project_id IS 'プロジェクトID（外部キー）';
COMMENT ON COLUMN project_memories.user_id IS 'ユーザーID（外部キー）';
COMMENT ON COLUMN project_memories.memory_key IS 'メモリキー';
COMMENT ON COLUMN project_memories.memory_value IS 'メモリ値';
COMMENT ON COLUMN project_memories.created_at IS '作成日時';
COMMENT ON COLUMN project_memories.updated_at IS '更新日時';

-- トランザクションコミット
COMMIT;

-- ================================================================
-- マイグレーション完了確認
-- ================================================================

-- 追加されたカラムを確認
SELECT 
    table_name,
    column_name,
    data_type,
    column_default
FROM information_schema.columns
WHERE table_name IN ('projects', 'chat_history', 'project_memories')
  AND table_schema = 'public'
ORDER BY table_name, ordinal_position;
