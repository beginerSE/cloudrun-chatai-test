# BigQuery AI データ分析エージェント デプロイ手順書

**ドキュメントバージョン**: 1.0  
**作成日**: 2025年11月

---

## 1. 概要

本ドキュメントは、BigQuery AI データ分析エージェントをGoogle Cloud Platform（Cloud Run + Cloud SQL + IAP）環境にデプロイするための手順を説明します。

### 1.1 前提条件

- Google Cloud Platform アカウント
- Google Cloud SDK（gcloud）インストール済み
- Docker インストール済み
- 適切なIAM権限を持つユーザー
- Google Workspace 組織（IAP使用のため）

### 1.2 必要な権限

デプロイ担当者には以下のIAMロールが必要です：

| ロール | 用途 |
|--------|------|
| roles/run.admin | Cloud Run管理 |
| roles/cloudsql.admin | Cloud SQL管理 |
| roles/iap.admin | IAP設定 |
| roles/secretmanager.admin | Secret Manager管理 |
| roles/artifactregistry.admin | コンテナレジストリ管理 |
| roles/iam.serviceAccountAdmin | サービスアカウント管理 |

---

## 2. 事前準備

### 2.1 GCPプロジェクト設定

```bash
# プロジェクトIDを設定
export PROJECT_ID="your-project-id"
export REGION="asia-northeast1"

# gcloud設定
gcloud config set project $PROJECT_ID
gcloud config set run/region $REGION

# 必要なAPIを有効化
gcloud services enable \
  run.googleapis.com \
  sqladmin.googleapis.com \
  iap.googleapis.com \
  secretmanager.googleapis.com \
  artifactregistry.googleapis.com \
  cloudresourcemanager.googleapis.com \
  bigquery.googleapis.com
```

### 2.2 Artifact Registry リポジトリ作成

```bash
# Dockerリポジトリを作成
gcloud artifacts repositories create bigquery-ai-agent \
  --repository-format=docker \
  --location=$REGION \
  --description="BigQuery AI Agent Docker images"

# Docker認証設定
gcloud auth configure-docker ${REGION}-docker.pkg.dev
```

### 2.3 サービスアカウント作成

```bash
# Cloud Run用サービスアカウント
gcloud iam service-accounts create bq-ai-agent-sa \
  --display-name="BigQuery AI Agent Service Account"

# 必要なロールを付与
SA_EMAIL="bq-ai-agent-sa@${PROJECT_ID}.iam.gserviceaccount.com"

# Cloud SQL接続権限
gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:${SA_EMAIL}" \
  --role="roles/cloudsql.client"

# BigQuery権限
gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:${SA_EMAIL}" \
  --role="roles/bigquery.dataViewer"

gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:${SA_EMAIL}" \
  --role="roles/bigquery.jobUser"

# Secret Manager権限
gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:${SA_EMAIL}" \
  --role="roles/secretmanager.secretAccessor"
```

---

## 3. Cloud SQL セットアップ

### 3.1 インスタンス作成

```bash
# Cloud SQLインスタンス作成（本番環境向け）
gcloud sql instances create bq-ai-agent-db \
  --database-version=POSTGRES_15 \
  --tier=db-custom-2-4096 \
  --region=$REGION \
  --storage-type=SSD \
  --storage-size=10GB \
  --storage-auto-increase \
  --backup-start-time=03:00 \
  --availability-type=REGIONAL \
  --maintenance-window-day=SUN \
  --maintenance-window-hour=04

# 開発環境向け（低コスト）
# gcloud sql instances create bq-ai-agent-db-dev \
#   --database-version=POSTGRES_15 \
#   --tier=db-f1-micro \
#   --region=$REGION \
#   --storage-type=HDD \
#   --storage-size=10GB
```

### 3.2 データベース・ユーザー作成

```bash
# データベース作成
gcloud sql databases create bq_ai_agent --instance=bq-ai-agent-db

# パスワード生成
DB_PASSWORD=$(openssl rand -base64 32)

# ユーザー作成
gcloud sql users create bq_ai_user \
  --instance=bq-ai-agent-db \
  --password="${DB_PASSWORD}"

# 接続名を取得
INSTANCE_CONNECTION_NAME=$(gcloud sql instances describe bq-ai-agent-db \
  --format='value(connectionName)')

echo "Instance Connection Name: ${INSTANCE_CONNECTION_NAME}"
echo "Database Password: ${DB_PASSWORD}"
```

### 3.3 Secret Manager に認証情報を保存

```bash
# データベースURLをシークレットとして保存
echo -n "postgresql://bq_ai_user:${DB_PASSWORD}@/${DB_NAME}?host=/cloudsql/${INSTANCE_CONNECTION_NAME}" | \
  gcloud secrets create database-url --data-file=-

# セッションシークレット作成
SESSION_SECRET=$(openssl rand -base64 32)
echo -n "${SESSION_SECRET}" | \
  gcloud secrets create session-secret --data-file=-
```

### 3.4 データベーススキーマ初期化

```bash
# Cloud SQL Auth Proxyで接続（ローカルから）
cloud_sql_proxy -instances=${INSTANCE_CONNECTION_NAME}=tcp:5432 &

# スキーマ適用
psql "postgresql://bq_ai_user:${DB_PASSWORD}@localhost:5432/bq_ai_agent" < database/schema.sql
```

**database/schema.sql の内容:**

```sql
-- ユーザーテーブル
CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- プロジェクトテーブル
CREATE TABLE IF NOT EXISTS projects (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
    name VARCHAR(100) NOT NULL,
    description TEXT,
    bigquery_project_id VARCHAR(100),
    bigquery_dataset_id VARCHAR(100),
    openai_api_key TEXT,
    gemini_api_key TEXT,
    ai_provider VARCHAR(20) DEFAULT 'openai',
    service_account_json TEXT,
    use_env_json BOOLEAN DEFAULT FALSE,
    is_active BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- チャットセッションテーブル
CREATE TABLE IF NOT EXISTS chat_sessions (
    id SERIAL PRIMARY KEY,
    project_id INTEGER REFERENCES projects(id) ON DELETE CASCADE,
    title VARCHAR(200) DEFAULT 'New Chat',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- チャット履歴テーブル
CREATE TABLE IF NOT EXISTS chat_history (
    id SERIAL PRIMARY KEY,
    project_id INTEGER REFERENCES projects(id) ON DELETE CASCADE,
    session_id INTEGER REFERENCES chat_sessions(id) ON DELETE CASCADE,
    user_message TEXT NOT NULL,
    ai_response TEXT,
    query_result JSONB,
    reasoning_process TEXT,
    steps_count INTEGER DEFAULT 0,
    processing_time FLOAT DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- チャットタスクテーブル（Cloud Run対応）
CREATE TABLE IF NOT EXISTS chat_tasks (
    id VARCHAR(36) PRIMARY KEY,
    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
    project_id INTEGER,
    status VARCHAR(20) DEFAULT 'pending',
    steps JSONB DEFAULT '[]',
    result JSONB,
    reasoning TEXT,
    error TEXT,
    cancelled BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- プロジェクトメモリテーブル
CREATE TABLE IF NOT EXISTS project_memories (
    id SERIAL PRIMARY KEY,
    project_id INTEGER REFERENCES projects(id) ON DELETE CASCADE,
    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
    memory_key VARCHAR(100) NOT NULL,
    memory_value TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(project_id, memory_key)
);

-- インデックス作成
CREATE INDEX IF NOT EXISTS idx_projects_user_id ON projects(user_id);
CREATE INDEX IF NOT EXISTS idx_projects_is_active ON projects(user_id, is_active);
CREATE INDEX IF NOT EXISTS idx_chat_sessions_project_id ON chat_sessions(project_id);
CREATE INDEX IF NOT EXISTS idx_chat_history_session_id ON chat_history(session_id);
CREATE INDEX IF NOT EXISTS idx_chat_tasks_user_id ON chat_tasks(user_id);
CREATE INDEX IF NOT EXISTS idx_chat_tasks_status ON chat_tasks(status);
CREATE INDEX IF NOT EXISTS idx_project_memories_project_id ON project_memories(project_id);
```

---

## 4. Dockerイメージのビルド・プッシュ

### 4.1 Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# システム依存関係
RUN apt-get update && apt-get install -y \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Python依存関係
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# アプリケーションコード
COPY . .

# 非rootユーザーで実行
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

# ポート設定
ENV PORT=8080
EXPOSE 8080

# Gunicornで起動
CMD exec gunicorn --bind :$PORT --workers 2 --threads 8 --timeout 300 app:app
```

### 4.2 requirements.txt

```
flask>=3.0.0
flask-login>=0.6.3
gunicorn>=21.0.0
psycopg2-binary>=2.9.9
bcrypt>=4.1.2
openai>=1.12.0
google-generativeai>=0.4.0
google-cloud-bigquery>=3.17.0
mcp>=1.0.0
mcp-server-bigquery>=0.1.0
pandas>=2.2.0
numpy>=1.26.0
matplotlib>=3.8.0
seaborn>=0.13.0
scikit-learn>=1.4.0
scipy>=1.12.0
statsmodels>=0.14.0
python-dotenv>=1.0.0
```

### 4.3 ビルド・プッシュ

```bash
# イメージビルド
docker build -t ${REGION}-docker.pkg.dev/${PROJECT_ID}/bigquery-ai-agent/app:latest .

# プッシュ
docker push ${REGION}-docker.pkg.dev/${PROJECT_ID}/bigquery-ai-agent/app:latest
```

---

## 5. Cloud Run デプロイ

### 5.1 サービスデプロイ

```bash
gcloud run deploy bq-ai-agent \
  --image=${REGION}-docker.pkg.dev/${PROJECT_ID}/bigquery-ai-agent/app:latest \
  --platform=managed \
  --region=$REGION \
  --service-account=${SA_EMAIL} \
  --add-cloudsql-instances=${INSTANCE_CONNECTION_NAME} \
  --set-secrets=DATABASE_URL=database-url:latest,SESSION_SECRET=session-secret:latest \
  --set-env-vars="INSTANCE_CONNECTION_NAME=${INSTANCE_CONNECTION_NAME}" \
  --memory=2Gi \
  --cpu=2 \
  --timeout=300 \
  --concurrency=80 \
  --min-instances=1 \
  --max-instances=10 \
  --no-allow-unauthenticated \
  --iap
```

### 5.2 デプロイ確認

```bash
# サービスURL取得
SERVICE_URL=$(gcloud run services describe bq-ai-agent \
  --region=$REGION \
  --format='value(status.url)')

echo "Service URL: ${SERVICE_URL}"
```

---

## 6. IAP 設定

### 6.1 IAPサービスエージェント作成

```bash
# IAPサービスエージェント作成
gcloud beta services identity create \
  --service=iap.googleapis.com \
  --project=$PROJECT_ID

# プロジェクト番号取得
PROJECT_NUMBER=$(gcloud projects describe $PROJECT_ID --format='value(projectNumber)')

# IAPサービスアカウントにCloud Run invoker権限付与
gcloud run services add-iam-policy-binding bq-ai-agent \
  --region=$REGION \
  --member="serviceAccount:service-${PROJECT_NUMBER}@gcp-sa-iap.iam.gserviceaccount.com" \
  --role="roles/run.invoker"
```

### 6.2 ユーザーアクセス権限設定

```bash
# 特定ユーザーにアクセス権限付与
gcloud beta iap web add-iam-policy-binding \
  --resource-type=cloud-run \
  --service=bq-ai-agent \
  --region=$REGION \
  --member="user:user@example.com" \
  --role="roles/iap.httpsResourceAccessor" \
  --condition=None

# Google Groupにアクセス権限付与（推奨）
gcloud beta iap web add-iam-policy-binding \
  --resource-type=cloud-run \
  --service=bq-ai-agent \
  --region=$REGION \
  --member="group:bq-ai-users@example.com" \
  --role="roles/iap.httpsResourceAccessor" \
  --condition=None

# ドメイン全体にアクセス権限付与
gcloud beta iap web add-iam-policy-binding \
  --resource-type=cloud-run \
  --service=bq-ai-agent \
  --region=$REGION \
  --member="domain:example.com" \
  --role="roles/iap.httpsResourceAccessor" \
  --condition=None
```

### 6.3 OAuth同意画面設定（必要に応じて）

1. Google Cloud Console → 「APIとサービス」→「OAuth同意画面」
2. ユーザーの種類: 「内部」を選択
3. アプリ名、サポートメール等を入力
4. スコープは最小限（email, profile, openid）

---

## 7. BigQuery アクセス設定

### 7.1 サービスアカウントへのBigQuery権限付与

分析対象のBigQueryプロジェクトに対して権限を付与：

```bash
# 分析対象のプロジェクトID
TARGET_BQ_PROJECT="your-bigquery-project-id"

# BigQuery Data Viewer
gcloud projects add-iam-policy-binding $TARGET_BQ_PROJECT \
  --member="serviceAccount:${SA_EMAIL}" \
  --role="roles/bigquery.dataViewer"

# BigQuery Job User
gcloud projects add-iam-policy-binding $TARGET_BQ_PROJECT \
  --member="serviceAccount:${SA_EMAIL}" \
  --role="roles/bigquery.jobUser"
```

### 7.2 データセットレベルの権限（より細かい制御）

```bash
# 特定データセットのみにアクセス許可
bq update --source /dev/stdin ${TARGET_BQ_PROJECT}:dataset_name <<EOF
{
  "access": [
    {
      "role": "READER",
      "userByEmail": "${SA_EMAIL}"
    }
  ]
}
EOF
```

---

## 8. 動作確認

### 8.1 ヘルスチェック

```bash
# IAP経由でアクセス（ブラウザでアクセス）
echo "Open in browser: ${SERVICE_URL}"

# または gcloud でアクセステスト
gcloud run services proxy bq-ai-agent --region=$REGION
```

### 8.2 ログ確認

```bash
# Cloud Runログ確認
gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=bq-ai-agent" \
  --limit=50 \
  --format="table(timestamp,severity,textPayload)"

# エラーログのみ
gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=bq-ai-agent AND severity>=ERROR" \
  --limit=20
```

### 8.3 Cloud SQLログ確認

```bash
gcloud logging read "resource.type=cloudsql_database AND resource.labels.database_id=${PROJECT_ID}:bq-ai-agent-db" \
  --limit=20
```

---

## 9. 本番環境チェックリスト

### 9.1 デプロイ前

- [ ] GCPプロジェクトが組織に属している
- [ ] 必要なAPIがすべて有効化されている
- [ ] サービスアカウントに適切な権限が付与されている
- [ ] Cloud SQL インスタンスが作成・設定されている
- [ ] Secret Manager にシークレットが登録されている
- [ ] Dockerイメージがビルド・プッシュされている

### 9.2 デプロイ後

- [ ] Cloud Run サービスが正常に起動している
- [ ] IAP が有効化されている
- [ ] 許可されたユーザーがアクセスできる
- [ ] 許可されていないユーザーがアクセスできない
- [ ] ログインが正常に動作する
- [ ] BigQuery への接続が成功する
- [ ] AI チャットが正常に動作する

### 9.3 セキュリティ確認

- [ ] HTTPS のみでアクセス可能
- [ ] APIキーがログに出力されていない
- [ ] セッションが正しく管理されている
- [ ] SQLインジェクションが防止されている

---

## 10. トラブルシューティング

### 10.1 よくある問題

| 問題 | 原因 | 解決策 |
|------|------|--------|
| IAP認証後に403エラー | IAP Service AgentにRun Invoker権限がない | セクション6.1の手順を実行 |
| データベース接続エラー | Cloud SQL接続設定が不正 | INSTANCE_CONNECTION_NAMEを確認 |
| BigQueryアクセス拒否 | サービスアカウント権限不足 | BigQuery権限を再確認 |
| コールドスタートが遅い | min-instancesが0 | min-instances=1に設定 |
| タイムアウトエラー | タイムアウト設定が短い | --timeout=300に設定 |

### 10.2 デバッグコマンド

```bash
# サービス詳細確認
gcloud run services describe bq-ai-agent --region=$REGION

# リビジョン確認
gcloud run revisions list --service=bq-ai-agent --region=$REGION

# IAMポリシー確認
gcloud run services get-iam-policy bq-ai-agent --region=$REGION

# Cloud SQL接続テスト
gcloud sql connect bq-ai-agent-db --user=bq_ai_user
```

---

## 11. 更新・ロールバック

### 11.1 アプリケーション更新

```bash
# 新バージョンビルド・プッシュ
docker build -t ${REGION}-docker.pkg.dev/${PROJECT_ID}/bigquery-ai-agent/app:v2.0.0 .
docker push ${REGION}-docker.pkg.dev/${PROJECT_ID}/bigquery-ai-agent/app:v2.0.0

# デプロイ（段階的ロールアウト）
gcloud run deploy bq-ai-agent \
  --image=${REGION}-docker.pkg.dev/${PROJECT_ID}/bigquery-ai-agent/app:v2.0.0 \
  --region=$REGION \
  --no-traffic

# トラフィック切り替え（100%）
gcloud run services update-traffic bq-ai-agent \
  --region=$REGION \
  --to-latest
```

### 11.2 ロールバック

```bash
# リビジョン一覧確認
gcloud run revisions list --service=bq-ai-agent --region=$REGION

# 特定リビジョンにロールバック
gcloud run services update-traffic bq-ai-agent \
  --region=$REGION \
  --to-revisions=bq-ai-agent-00001-abc=100
```

---

## 12. 付録

### 12.1 環境変数一覧

| 変数名 | 説明 | 設定方法 |
|--------|------|----------|
| DATABASE_URL | Cloud SQL接続URL | Secret Manager |
| SESSION_SECRET | セッション暗号化キー | Secret Manager |
| INSTANCE_CONNECTION_NAME | Cloud SQL接続名 | 環境変数 |
| PORT | リスニングポート | 自動設定（8080） |
| GOOGLE_CLOUD_PROJECT | GCPプロジェクトID | 自動設定 |

### 12.2 関連ドキュメント

- [Cloud Run ドキュメント](https://cloud.google.com/run/docs)
- [Cloud SQL ドキュメント](https://cloud.google.com/sql/docs)
- [IAP ドキュメント](https://cloud.google.com/iap/docs)
- [Secret Manager ドキュメント](https://cloud.google.com/secret-manager/docs)
