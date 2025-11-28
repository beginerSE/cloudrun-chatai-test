# BigQuery AI データ分析エージェント 運用マニュアル

**ドキュメントバージョン**: 1.0  
**作成日**: 2025年11月

---

## 1. 概要

本ドキュメントは、BigQuery AI データ分析エージェントの日常運用、監視、トラブルシューティング手順を説明します。

---

## 2. システム監視

### 2.1 監視ダッシュボード

Google Cloud Console から以下を監視します：

1. **Cloud Run ダッシュボード**
   - URL: `https://console.cloud.google.com/run`
   - 監視項目: リクエスト数、レイテンシ、エラー率、インスタンス数

2. **Cloud SQL ダッシュボード**
   - URL: `https://console.cloud.google.com/sql`
   - 監視項目: CPU使用率、メモリ使用率、接続数、ストレージ

3. **Cloud Logging**
   - URL: `https://console.cloud.google.com/logs`
   - アプリケーションログ、エラーログの確認

### 2.2 重要メトリクス

| メトリクス | 正常値 | 警告閾値 | 緊急閾値 |
|-----------|--------|----------|----------|
| Cloud Run レスポンス時間 (P95) | < 2秒 | > 5秒 | > 10秒 |
| Cloud Run エラー率 | < 1% | > 3% | > 5% |
| Cloud Run インスタンス数 | 1-3 | > 8 | = 10 (上限) |
| Cloud SQL CPU使用率 | < 50% | > 70% | > 90% |
| Cloud SQL 接続数 | < 50 | > 80 | > 95 |
| Cloud SQL ストレージ使用率 | < 70% | > 85% | > 95% |

### 2.3 アラート設定

```bash
# Cloud Runエラー率アラート
gcloud alpha monitoring policies create \
  --display-name="Cloud Run High Error Rate" \
  --condition-display-name="Error rate > 5%" \
  --condition-filter='resource.type="cloud_run_revision" AND metric.type="run.googleapis.com/request_count" AND metric.labels.response_code_class!="2xx"' \
  --condition-threshold-value=0.05 \
  --condition-threshold-comparison=COMPARISON_GT \
  --notification-channels="projects/${PROJECT_ID}/notificationChannels/YOUR_CHANNEL_ID"
```

---

## 3. 日常運用タスク

### 3.1 日次タスク

| タスク | 内容 | 確認方法 |
|--------|------|----------|
| ログ確認 | ERRORレベルのログがないか確認 | Cloud Logging |
| バックアップ確認 | Cloud SQL自動バックアップが成功しているか | Cloud SQL コンソール |
| リソース使用状況 | CPU/メモリ/ストレージの使用状況 | Cloud Monitoring |

### 3.2 週次タスク

| タスク | 内容 |
|--------|------|
| パフォーマンスレビュー | 過去1週間のレスポンス時間推移を確認 |
| コスト確認 | Cloud Billing で費用推移を確認 |
| セキュリティログ確認 | IAP認証ログで不審なアクセスがないか確認 |

### 3.3 月次タスク

| タスク | 内容 |
|--------|------|
| インスタンスタイプ見直し | 使用状況に応じてCloud Run/Cloud SQLのスペック調整 |
| 古いリビジョン削除 | 不要なCloud Runリビジョンの削除 |
| チャット履歴クリーンアップ | 古いチャットタスク（chat_tasks）の削除 |
| セキュリティアップデート | ベースイメージ、パッケージのアップデート確認 |

---

## 4. ユーザー管理

### 4.1 IAP アクセス権限の追加

```bash
# 個別ユーザーを追加
gcloud beta iap web add-iam-policy-binding \
  --resource-type=cloud-run \
  --service=bq-ai-agent \
  --region=$REGION \
  --member="user:newuser@example.com" \
  --role="roles/iap.httpsResourceAccessor" \
  --condition=None

# Google Groupに追加（推奨）
# Google Workspace Admin Console でグループメンバーを管理
```

### 4.2 IAP アクセス権限の削除

```bash
gcloud beta iap web remove-iam-policy-binding \
  --resource-type=cloud-run \
  --service=bq-ai-agent \
  --region=$REGION \
  --member="user:removeduser@example.com" \
  --role="roles/iap.httpsResourceAccessor"
```

### 4.3 アプリケーション内ユーザー管理

データベースに直接接続してユーザーを管理：

```bash
# Cloud SQL Proxy経由で接続
cloud_sql_proxy -instances=${INSTANCE_CONNECTION_NAME}=tcp:5432 &

# psql接続
psql "postgresql://bq_ai_user:${DB_PASSWORD}@localhost:5432/bq_ai_agent"
```

```sql
-- ユーザー一覧確認
SELECT id, username, email, created_at FROM users ORDER BY created_at DESC;

-- 特定ユーザーの無効化（パスワードを無効なハッシュに変更）
UPDATE users SET password_hash = 'DISABLED' WHERE email = 'user@example.com';

-- ユーザー削除（関連データも削除される）
DELETE FROM users WHERE email = 'user@example.com';
```

---

## 5. バックアップと復旧

### 5.1 Cloud SQL バックアップ

#### 自動バックアップの確認

```bash
gcloud sql backups list --instance=bq-ai-agent-db
```

#### 手動バックアップ作成

```bash
gcloud sql backups create --instance=bq-ai-agent-db \
  --description="Manual backup before update"
```

#### バックアップからの復元

```bash
# 復元用の新インスタンスを作成
gcloud sql instances restore-backup bq-ai-agent-db \
  --restore-instance=bq-ai-agent-db-restored \
  --backup-id=BACKUP_ID
```

### 5.2 ポイントインタイムリカバリ

```bash
# 特定時点への復元
gcloud sql instances clone bq-ai-agent-db bq-ai-agent-db-recovered \
  --point-in-time "2025-01-15T10:30:00.000Z"
```

### 5.3 復旧手順

1. **データベース障害時**
   ```bash
   # 最新バックアップから復元
   gcloud sql backups list --instance=bq-ai-agent-db
   gcloud sql instances restore-backup bq-ai-agent-db-restored \
     --restore-instance=bq-ai-agent-db \
     --backup-id=<BACKUP_ID>
   
   # Cloud Run再デプロイ（接続先更新が必要な場合）
   gcloud run services update bq-ai-agent --region=$REGION
   ```

2. **アプリケーション障害時**
   ```bash
   # 前のリビジョンにロールバック
   gcloud run revisions list --service=bq-ai-agent --region=$REGION
   gcloud run services update-traffic bq-ai-agent \
     --region=$REGION \
     --to-revisions=<PREVIOUS_REVISION>=100
   ```

---

## 6. トラブルシューティング

### 6.1 よくある問題と対処法

#### 問題1: ユーザーがログインできない

**症状**: IAP認証後、アプリケーションのログインページでエラー

**確認手順**:
```bash
# Cloud Runログ確認
gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=bq-ai-agent AND textPayload:login" --limit=20

# データベース接続確認
gcloud sql connect bq-ai-agent-db --user=bq_ai_user
```

**対処法**:
- セッションシークレットが正しく設定されているか確認
- データベース接続が正常か確認
- ユーザーが正しく登録されているか確認

#### 問題2: AI チャットが応答しない

**症状**: 質問を送信しても応答がない、またはタイムアウト

**確認手順**:
```bash
# エラーログ確認
gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=bq-ai-agent AND severity>=ERROR" --limit=20

# BigQuery権限確認
gcloud projects get-iam-policy $TARGET_BQ_PROJECT \
  --flatten="bindings[].members" \
  --filter="bindings.members:${SA_EMAIL}"
```

**対処法**:
- AIプロバイダーのAPIキーが有効か確認
- BigQueryへのアクセス権限を確認
- Cloud Run のタイムアウト設定を確認（300秒推奨）

#### 問題3: パフォーマンス低下

**症状**: ページ読み込みが遅い、レスポンスタイムが増加

**確認手順**:
```bash
# Cloud Run メトリクス確認
gcloud monitoring metrics list --filter="metric.type:run.googleapis.com"

# Cloud SQL メトリクス確認
gcloud sql instances describe bq-ai-agent-db
```

**対処法**:
- Cloud Run インスタンス数を増加
- Cloud SQL のスペックを上げる
- 古いチャットタスクをクリーンアップ

```sql
-- 24時間以上前の完了済みタスクを削除
DELETE FROM chat_tasks 
WHERE status IN ('completed', 'error', 'cancelled') 
AND created_at < NOW() - INTERVAL '24 hours';
```

#### 問題4: メモリ不足エラー

**症状**: Cloud Run が頻繁にクラッシュ、OOM (Out of Memory) エラー

**対処法**:
```bash
# メモリを増加してデプロイ
gcloud run services update bq-ai-agent \
  --region=$REGION \
  --memory=4Gi
```

### 6.2 ログ調査コマンド

```bash
# 直近1時間のエラーログ
gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=bq-ai-agent AND severity>=ERROR AND timestamp>=\"$(date -u -d '1 hour ago' '+%Y-%m-%dT%H:%M:%SZ')\"" --format="table(timestamp,severity,textPayload)"

# 特定ユーザーのアクティビティ
gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=bq-ai-agent AND textPayload:user@example.com" --limit=50

# IAP認証ログ
gcloud logging read "resource.type=iap_request" --limit=20
```

### 6.3 緊急連絡先

| 状況 | 連絡先 |
|------|--------|
| システム障害 | インフラチーム: infra@example.com |
| セキュリティインシデント | セキュリティチーム: security@example.com |
| GCP サポート | Google Cloud サポートコンソール |

---

## 7. 定期メンテナンス

### 7.1 データベースメンテナンス

```sql
-- テーブル統計更新
ANALYZE;

-- 不要なチャットタスク削除（週次）
DELETE FROM chat_tasks 
WHERE created_at < NOW() - INTERVAL '7 days';

-- インデックス再構築（月次）
REINDEX DATABASE bq_ai_agent;
```

### 7.2 アプリケーション更新

```bash
# 1. 新バージョンをビルド・プッシュ
docker build -t ${REGION}-docker.pkg.dev/${PROJECT_ID}/bigquery-ai-agent/app:v2.0.0 .
docker push ${REGION}-docker.pkg.dev/${PROJECT_ID}/bigquery-ai-agent/app:v2.0.0

# 2. ステージング環境でテスト
gcloud run deploy bq-ai-agent-staging \
  --image=${REGION}-docker.pkg.dev/${PROJECT_ID}/bigquery-ai-agent/app:v2.0.0 \
  --region=$REGION

# 3. 本番環境にデプロイ（段階的ロールアウト）
gcloud run deploy bq-ai-agent \
  --image=${REGION}-docker.pkg.dev/${PROJECT_ID}/bigquery-ai-agent/app:v2.0.0 \
  --region=$REGION \
  --no-traffic

# 4. トラフィックを段階的に切り替え
gcloud run services update-traffic bq-ai-agent \
  --region=$REGION \
  --to-revisions=LATEST=10  # 10%

# 問題なければ100%に
gcloud run services update-traffic bq-ai-agent \
  --region=$REGION \
  --to-latest
```

### 7.3 セキュリティパッチ適用

```bash
# ベースイメージを更新してリビルド
docker pull python:3.11-slim
docker build --no-cache -t ${REGION}-docker.pkg.dev/${PROJECT_ID}/bigquery-ai-agent/app:latest .
docker push ${REGION}-docker.pkg.dev/${PROJECT_ID}/bigquery-ai-agent/app:latest

# 再デプロイ
gcloud run deploy bq-ai-agent \
  --image=${REGION}-docker.pkg.dev/${PROJECT_ID}/bigquery-ai-agent/app:latest \
  --region=$REGION
```

---

## 8. コスト管理

### 8.1 コスト内訳目安（月額）

| サービス | 仕様 | 概算費用 |
|----------|------|----------|
| Cloud Run | min=1, max=10, 2GB RAM | $20-100 |
| Cloud SQL | db-custom-2-4096 | $50-80 |
| BigQuery | 従量課金（クエリ量による） | $5-50 |
| Artifact Registry | イメージ保存 | $1-5 |
| Cloud Logging | ログ保存 | $5-20 |
| **合計** | - | **$80-250** |

### 8.2 コスト最適化のヒント

1. **Cloud Run**
   - `min-instances=0` に設定（許容できる場合）
   - `max-instances` を適切に制限
   - 不要なリビジョンを削除

2. **Cloud SQL**
   - 開発環境は `db-f1-micro` を使用
   - オフピーク時のインスタンス停止（開発環境のみ）

3. **BigQuery**
   - クエリ結果のキャッシュを活用
   - パーティションテーブルを使用

### 8.3 コスト監視

```bash
# 課金アラート設定
gcloud billing budgets create \
  --billing-account=BILLING_ACCOUNT_ID \
  --display-name="BQ AI Agent Budget" \
  --budget-amount=300USD \
  --threshold-rule=percent=80 \
  --threshold-rule=percent=100
```

---

## 9. 監査ログ

### 9.1 アクセスログ確認

```bash
# IAP認証ログ
gcloud logging read 'resource.type="iap_request"' \
  --format="table(timestamp,protoPayload.authenticationInfo.principalEmail,protoPayload.requestMetadata.callerIp,httpRequest.requestUrl)"

# Cloud Run アクセスログ
gcloud logging read 'resource.type="cloud_run_revision" AND httpRequest.requestMethod!=""' \
  --format="table(timestamp,httpRequest.requestMethod,httpRequest.requestUrl,httpRequest.status)"
```

### 9.2 管理操作ログ

```bash
# IAM変更ログ
gcloud logging read 'protoPayload.methodName:"SetIamPolicy"' \
  --format="table(timestamp,protoPayload.authenticationInfo.principalEmail,protoPayload.methodName)"

# リソース変更ログ
gcloud logging read 'resource.type="cloud_run_revision" AND protoPayload.methodName:"Deploy"' \
  --format="table(timestamp,protoPayload.authenticationInfo.principalEmail)"
```

---

## 10. 付録

### 10.1 重要なURLリスト

| 名称 | URL |
|------|-----|
| Cloud Run コンソール | https://console.cloud.google.com/run |
| Cloud SQL コンソール | https://console.cloud.google.com/sql |
| IAP コンソール | https://console.cloud.google.com/security/iap |
| Cloud Logging | https://console.cloud.google.com/logs |
| Cloud Monitoring | https://console.cloud.google.com/monitoring |
| Billing | https://console.cloud.google.com/billing |

### 10.2 重要なコマンドチートシート

```bash
# サービス状態確認
gcloud run services describe bq-ai-agent --region=$REGION

# ログ確認（エラーのみ）
gcloud logging read "resource.type=cloud_run_revision AND severity>=ERROR" --limit=20

# データベース接続
cloud_sql_proxy -instances=${INSTANCE_CONNECTION_NAME}=tcp:5432 &
psql "postgresql://bq_ai_user:${DB_PASSWORD}@localhost:5432/bq_ai_agent"

# ロールバック
gcloud run services update-traffic bq-ai-agent --region=$REGION --to-revisions=REVISION_NAME=100

# インスタンス数調整
gcloud run services update bq-ai-agent --region=$REGION --max-instances=5

# メモリ増加
gcloud run services update bq-ai-agent --region=$REGION --memory=4Gi
```
