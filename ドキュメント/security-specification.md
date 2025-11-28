# BigQuery AI データ分析エージェント セキュリティ仕様書

**ドキュメントバージョン**: 1.0  
**作成日**: 2025年11月  
**機密レベル**: 社内限定

---

## 1. 概要

本ドキュメントは、BigQuery AI データ分析エージェントのセキュリティ設計、認証・認可方式、およびセキュリティ対策について説明します。

---

## 2. セキュリティアーキテクチャ

### 2.1 多層防御モデル

```
┌──────────────────────────────────────────────────────────────────────┐
│                        セキュリティレイヤー                            │
├──────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │ Layer 1: ネットワークセキュリティ                                 │ │
│  │ ・HTTPS/TLS 1.2+強制                                            │ │
│  │ ・Cloud Armor DDoS保護                                          │ │
│  │ ・VPC Service Controls（オプション）                             │ │
│  └─────────────────────────────────────────────────────────────────┘ │
│                              ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │ Layer 2: Identity-Aware Proxy (IAP)                             │ │
│  │ ・Google Workspace認証必須                                       │ │
│  │ ・組織ドメイン制限                                               │ │
│  │ ・コンテキストアウェアアクセス（オプション）                       │ │
│  └─────────────────────────────────────────────────────────────────┘ │
│                              ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │ Layer 3: アプリケーション認証                                     │ │
│  │ ・Flask-Login セッション管理                                     │ │
│  │ ・bcrypt パスワードハッシュ                                      │ │
│  │ ・署名付きCookieセッション                                       │ │
│  └─────────────────────────────────────────────────────────────────┘ │
│                              ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │ Layer 4: データアクセス制御                                       │ │
│  │ ・ユーザー単位のデータ分離                                       │ │
│  │ ・BigQuery読み取り専用強制                                       │ │
│  │ ・SQLインジェクション防止                                        │ │
│  └─────────────────────────────────────────────────────────────────┘ │
│                              ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │ Layer 5: 実行環境セキュリティ                                     │ │
│  │ ・Pythonサンドボックス                                          │ │
│  │ ・制限されたライブラリのみ許可                                   │ │
│  │ ・実行タイムアウト                                               │ │
│  └─────────────────────────────────────────────────────────────────┘ │
│                                                                       │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 3. 認証・認可

### 3.1 IAP（Identity-Aware Proxy）認証

#### 概要
IAPはGoogle Cloudの認証プロキシサービスで、アプリケーションへのアクセスをGoogle Workspace認証で保護します。

#### 設定要件

| 項目 | 設定値 |
|------|--------|
| 認証方式 | Google Workspace アカウント |
| 組織制限 | 社内ドメインのみ許可 |
| アクセス制御 | IAMロールベース |
| ログ記録 | Cloud Audit Logs |

#### 必要なIAMロール

```
roles/iap.httpsResourceAccessor
```

このロールを持つユーザー/グループのみがアプリケーションにアクセス可能です。

#### アクセス許可の設定例

```bash
# Google Groupにアクセス権限付与（推奨）
gcloud beta iap web add-iam-policy-binding \
  --resource-type=cloud-run \
  --service=bq-ai-agent \
  --region=asia-northeast1 \
  --member="group:bq-ai-users@example.com" \
  --role="roles/iap.httpsResourceAccessor"

# ドメイン全体にアクセス権限付与
gcloud beta iap web add-iam-policy-binding \
  --resource-type=cloud-run \
  --service=bq-ai-agent \
  --region=asia-northeast1 \
  --member="domain:example.com" \
  --role="roles/iap.httpsResourceAccessor"
```

### 3.2 アプリケーション認証

IAP認証をパスした後、アプリケーション固有の認証が行われます。

#### パスワードポリシー

| 項目 | 要件 |
|------|------|
| ハッシュアルゴリズム | bcrypt |
| ソルト | 自動生成（bcrypt内蔵） |
| ストレッチング | 12ラウンド |
| 最小文字数 | 8文字（推奨） |

#### セッション管理

| 項目 | 設定値 |
|------|--------|
| セッションストレージ | 署名付きCookie |
| 有効期限 | 24時間 |
| 暗号化 | HMAC-SHA256署名 |
| HttpOnly | 有効 |
| Secure | 有効（HTTPS必須） |
| SameSite | Lax |

### 3.3 サービスアカウント認証

BigQueryへのアクセスにはサービスアカウントを使用します。

#### 認証方式

| 方式 | 説明 | 使用場面 |
|------|------|----------|
| ADC (Application Default Credentials) | Cloud Run環境で自動的に利用可能 | 本番環境（推奨） |
| サービスアカウントJSONキー | JSONファイルをアップロード | 開発環境、外部プロジェクトアクセス |

---

## 4. BigQuery アクセス権限

### 4.1 必要な権限一覧

本システムのサービスアカウントには以下のIAMロールが必要です：

| ロール | ロールID | スコープ | 用途 |
|--------|----------|----------|------|
| BigQuery データ閲覧者 | `roles/bigquery.dataViewer` | データセットまたはプロジェクト | テーブルデータの読み取り |
| BigQuery ジョブユーザー | `roles/bigquery.jobUser` | プロジェクト（必須） | クエリジョブの実行 |

### 4.2 詳細な権限リスト

#### bigquery.dataViewer に含まれる権限

```
bigquery.datasets.get
bigquery.datasets.getIamPolicy
bigquery.models.export
bigquery.models.getData
bigquery.models.getMetadata
bigquery.models.list
bigquery.routines.get
bigquery.routines.list
bigquery.tables.export
bigquery.tables.get
bigquery.tables.getData
bigquery.tables.getIamPolicy
bigquery.tables.list
resourcemanager.projects.get
```

#### bigquery.jobUser に含まれる権限

```
bigquery.jobs.create
bigquery.jobs.list
```

### 4.3 最小権限の原則

推奨される権限設定：

```bash
# プロジェクトレベルでジョブ実行権限を付与
gcloud projects add-iam-policy-binding $TARGET_PROJECT \
  --member="serviceAccount:${SA_EMAIL}" \
  --role="roles/bigquery.jobUser"

# データセットレベルで読み取り権限を付与（推奨）
bq update --source /dev/stdin ${PROJECT}:${DATASET} <<EOF
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

### 4.4 クロスプロジェクトアクセス

分析対象のBigQueryプロジェクトが本アプリケーションのGCPプロジェクトと異なる場合：

1. 分析対象プロジェクトで本アプリケーションのサービスアカウントに権限を付与
2. 必要なロール: `bigquery.dataViewer` + `bigquery.jobUser`

---

## 5. データ保護

### 5.1 通信の暗号化

| 通信経路 | 暗号化方式 |
|----------|------------|
| クライアント ⇔ Cloud Run | TLS 1.2+ (HTTPS必須) |
| Cloud Run ⇔ Cloud SQL | Unix Socket (内部通信) |
| Cloud Run ⇔ BigQuery | gRPC over TLS |
| Cloud Run ⇔ OpenAI API | HTTPS |
| Cloud Run ⇔ Gemini API | HTTPS |

### 5.2 保存データの暗号化

| データ種別 | 暗号化方式 |
|-----------|------------|
| Cloud SQL データ | Google管理の暗号化キー（デフォルト） |
| Secret Manager | Google管理の暗号化キー |
| サービスアカウントJSON | ファイルシステム保存（アクセス制限） |
| セッションCookie | HMAC-SHA256署名 |
| パスワード | bcryptハッシュ |

### 5.3 機密データの取り扱い

| データ種別 | 保存場所 | 表示方法 |
|-----------|----------|----------|
| OpenAI APIキー | データベース | マスク表示（最初と最後の4文字のみ） |
| Gemini APIキー | データベース | マスク表示 |
| パスワード | データベース（ハッシュ化） | 表示しない |
| サービスアカウントJSON | ファイルシステム | パス名のみ表示 |

---

## 6. SQL インジェクション対策

### 6.1 防止策

本システムでは以下の対策を実施：

1. **読み取り専用クエリの強制**
   - SELECT文のみ許可
   - INSERT, UPDATE, DELETE, CREATE, DROP等は拒否

2. **パラメータ化クエリ**
   - アプリケーション内部のSQLはパラメータ化クエリを使用
   - ユーザー入力を直接SQL文字列に結合しない

3. **AIによるクエリ生成**
   - ユーザー入力は自然言語として処理
   - AIがSQLを生成し、構文チェック後に実行

### 6.2 クエリバリデーション

```python
def validate_query(query: str) -> bool:
    """SQLクエリが読み取り専用かチェック"""
    query_upper = query.upper().strip()
    
    # 禁止されたキーワード
    forbidden_keywords = [
        'INSERT', 'UPDATE', 'DELETE', 'DROP', 'CREATE', 
        'ALTER', 'TRUNCATE', 'MERGE', 'GRANT', 'REVOKE'
    ]
    
    for keyword in forbidden_keywords:
        if keyword in query_upper:
            return False
    
    # SELECT文で始まることを確認
    if not query_upper.startswith('SELECT'):
        return False
    
    return True
```

---

## 7. Python サンドボックス

### 7.1 セキュリティ設計

AIが生成したPythonコードは制限された環境で実行されます。

#### 許可されたライブラリ

| ライブラリ | バージョン | 用途 |
|-----------|-----------|------|
| pandas | 2.2+ | データ操作 |
| numpy | 1.26+ | 数値計算 |
| matplotlib | 3.8+ | グラフ作成 |
| seaborn | 0.13+ | 統計可視化 |
| scikit-learn | 1.4+ | 機械学習 |
| scipy | 1.12+ | 科学技術計算 |
| statsmodels | 0.14+ | 統計分析 |

#### 禁止されている操作

```python
# 禁止される操作の例
- ファイルシステムアクセス（open, os.*, pathlib）
- ネットワークアクセス（socket, requests, urllib）
- プロセス実行（subprocess, os.system）
- モジュールの動的インポート（__import__, importlib）
- コード実行（exec, eval, compile）
- システム情報取得（os.environ, sys.modules）
```

### 7.2 実行制限

| 制限項目 | 設定値 |
|----------|--------|
| 実行タイムアウト | 30秒 |
| メモリ制限 | 512MB |
| CPU制限 | シングルスレッド |
| 出力サイズ制限 | 10MB |

### 7.3 サンドボックス実装

```python
RESTRICTED_GLOBALS = {
    '__builtins__': {
        'abs': abs, 'all': all, 'any': any, 'bool': bool,
        'dict': dict, 'enumerate': enumerate, 'filter': filter,
        'float': float, 'int': int, 'len': len, 'list': list,
        'map': map, 'max': max, 'min': min, 'print': print,
        'range': range, 'round': round, 'set': set, 'sorted': sorted,
        'str': str, 'sum': sum, 'tuple': tuple, 'type': type,
        'zip': zip, 'True': True, 'False': False, 'None': None,
    },
    'pd': pd,
    'np': np,
    'plt': plt,
    'sns': sns,
}
```

---

## 8. ログとモニタリング

### 8.1 セキュリティログ

| ログ種別 | 保存先 | 保持期間 |
|----------|--------|----------|
| IAP認証ログ | Cloud Audit Logs | 400日 |
| アプリケーションログ | Cloud Logging | 30日 |
| Cloud SQLログ | Cloud Logging | 30日 |
| BigQueryクエリログ | BigQuery INFORMATION_SCHEMA | 180日 |

### 8.2 監視すべきイベント

| イベント | 重要度 | 対応 |
|----------|--------|------|
| 認証失敗（連続5回以上） | 高 | アラート通知 |
| 不正なSQLクエリ試行 | 高 | ログ記録、アラート |
| 大量データ取得 | 中 | ログ記録 |
| Pythonサンドボックス違反 | 高 | 実行拒否、ログ記録 |
| 異常なAPIコール頻度 | 中 | Rate limiting検討 |

### 8.3 セキュリティアラート設定

```bash
# 認証失敗アラート
gcloud alpha monitoring policies create \
  --display-name="IAP Auth Failures" \
  --condition-display-name="Multiple auth failures" \
  --condition-filter='resource.type="iap_request" AND protoPayload.status.code!=0' \
  --condition-threshold-value=10 \
  --condition-threshold-comparison=COMPARISON_GT \
  --aggregations-alignment-period=300s \
  --notification-channels="YOUR_CHANNEL_ID"
```

---

## 9. インシデント対応

### 9.1 セキュリティインシデント分類

| レベル | 説明 | 対応時間目標 |
|--------|------|------------|
| Critical | データ漏洩、不正アクセス成功 | 1時間以内 |
| High | 認証突破試行、SQLインジェクション試行 | 4時間以内 |
| Medium | 異常なアクセスパターン | 24時間以内 |
| Low | ポリシー違反、設定ミス | 1週間以内 |

### 9.2 対応手順

#### Critical/High インシデント

1. **即時対応**
   - サービスの一時停止（必要に応じて）
   - 影響範囲の特定
   - セキュリティチームへ連絡

2. **調査**
   - ログ分析
   - 攻撃ベクトルの特定
   - 被害範囲の確認

3. **復旧**
   - 脆弱性の修正
   - 認証情報のローテーション
   - サービス再開

4. **事後対応**
   - インシデントレポート作成
   - 再発防止策の実施

### 9.3 緊急連絡先

| 役割 | 連絡先 |
|------|--------|
| セキュリティチーム | security@example.com |
| インフラチーム | infra@example.com |
| 責任者 | manager@example.com |

---

## 10. コンプライアンス

### 10.1 準拠規格・法令

| 規格・法令 | 対応状況 | 備考 |
|-----------|----------|------|
| GDPR | 対応要 | 個人データ取り扱い時 |
| 個人情報保護法 | 対応要 | 国内運用時 |
| ISO 27001 | 参考 | GCPが認証取得済み |
| SOC 2 | 参考 | GCPが認証取得済み |

### 10.2 データ分類

| 分類 | 説明 | 取り扱い |
|------|------|----------|
| 機密 | APIキー、パスワード | 暗号化保存、マスク表示 |
| 社内限定 | チャット履歴、分析結果 | アクセス制御 |
| 公開可能 | ドキュメント | 制限なし |

---

## 11. 定期レビュー

### 11.1 レビュー項目

| 項目 | 頻度 | 担当 |
|------|------|------|
| IAMポリシー確認 | 四半期 | セキュリティチーム |
| アクセス権限棚卸 | 四半期 | 運用チーム |
| ログレビュー | 月次 | セキュリティチーム |
| 脆弱性スキャン | 月次 | インフラチーム |
| ペネトレーションテスト | 年次 | 外部ベンダー |

### 11.2 チェックリスト

- [ ] 不要なユーザーアカウントが削除されているか
- [ ] サービスアカウントの権限が最小限か
- [ ] APIキーが定期的にローテーションされているか
- [ ] セキュリティパッチが適用されているか
- [ ] バックアップが正常に取得されているか
- [ ] ログが適切に保存されているか

---

## 12. 付録

### 12.1 セキュリティ設定一覧

```yaml
# セキュリティ設定サマリー
authentication:
  iap:
    enabled: true
    audience: "Google Workspace users only"
    mfa: "Google Account MFA"
  application:
    password_hash: "bcrypt (12 rounds)"
    session_timeout: "24 hours"
    cookie_secure: true
    cookie_httponly: true

authorization:
  bigquery:
    roles:
      - "roles/bigquery.dataViewer"
      - "roles/bigquery.jobUser"
    query_type: "SELECT only"

encryption:
  in_transit: "TLS 1.2+"
  at_rest: "Google managed keys"

sandbox:
  python:
    timeout: "30 seconds"
    memory_limit: "512MB"
    allowed_libraries:
      - pandas
      - numpy
      - matplotlib
      - seaborn
      - scikit-learn
      - scipy
      - statsmodels
```

### 12.2 関連ドキュメント

- requirements.md - 要件定義書
- architecture.md - システムアーキテクチャ
- deployment-guide.md - デプロイ手順書
- operations-manual.md - 運用マニュアル
