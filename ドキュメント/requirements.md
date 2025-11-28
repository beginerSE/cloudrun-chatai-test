# BigQuery AI データ分析エージェント 要件定義書

**ドキュメントバージョン**: 1.0  
**作成日**: 2025年11月  
**最終更新日**: 2025年11月

---

## 1. はじめに

### 1.1 目的
本ドキュメントは、BigQuery AI データ分析エージェント（以下、本システム）のソフトウェア要件を定義するものです。本システムは、自然言語によるBigQueryデータ分析を可能にするWebアプリケーションであり、非技術者でもセルフサービスでデータ分析を行えることを目的としています。

### 1.2 スコープ
本システムは以下の機能を提供します：
- 自然言語からBigQuery SQLクエリへの自動変換・実行
- クエリ結果のテーブル表示およびグラフ可視化
- 複数ターンの対話型分析
- Pythonコードによる高度な統計分析・機械学習
- マルチプロジェクト・マルチユーザー管理

### 1.3 対象読者
- プロジェクトマネージャー
- 開発者・エンジニア
- インフラ管理者
- セキュリティ担当者

### 1.4 用語定義

| 用語 | 定義 |
|------|------|
| BigQuery | Google Cloudのサーバーレスデータウェアハウス |
| MCP | Model Context Protocol - LLMと外部ツールを連携するフレームワーク |
| ADC | Application Default Credentials - GCPの自動認証メカニズム |
| IAP | Identity-Aware Proxy - Googleの認証プロキシサービス |
| Cloud Run | Googleのサーバーレスコンテナ実行環境 |
| Cloud SQL | Googleのマネージドリレーショナルデータベース |

---

## 2. システム概要

### 2.1 製品概要
本システムは、ChatGPT/Gemini APIとBigQuery MCP Serverを統合し、自然言語ベースのデータ分析を実現するWebアプリケーションです。

### 2.2 主要機能
1. **自然言語クエリ**: ユーザーの質問をSQLに変換し実行
2. **スキーマ自動検出**: テーブル構造を自動取得しAIに提供
3. **データ可視化**: 棒グラフ、折れ線グラフ、円グラフ等を自動生成
4. **Python分析**: 統計分析・機械学習をセキュアに実行
5. **セッション管理**: チャット履歴の保存・再開
6. **プロジェクト管理**: 複数BigQueryプロジェクトの切り替え

### 2.3 ユーザークラス

| ユーザークラス | 説明 | 権限 |
|---------------|------|------|
| 一般ユーザー | データ分析を行う社内ユーザー | AI チャット、プロジェクト設定 |
| 管理者 | システム管理を行うユーザー | 上記 + ユーザー管理、システム設定 |

### 2.4 動作環境

| 項目 | 仕様 |
|------|------|
| 本番環境 | Google Cloud Run |
| データベース | Cloud SQL (PostgreSQL 15) |
| 認証 | Identity-Aware Proxy (IAP) |
| 対応ブラウザ | Chrome, Firefox, Safari, Edge（最新2バージョン） |
| 画面サイズ | デスクトップ（1280px以上推奨） |

---

## 3. 機能要件

### 3.1 ユーザー認証・認可

#### REQ-AUTH-001: IAP認証
- システムはIAPを通じて社内ユーザーのみアクセスを許可すること
- IAP認証をパスしたユーザーのみがアプリケーションにアクセス可能

#### REQ-AUTH-002: アプリケーション内ログイン
- IAP認証後、アプリケーション固有のユーザー登録・ログインが必要
- パスワードはbcryptでハッシュ化して保存
- セッションは暗号化されたCookieで管理

#### REQ-AUTH-003: セッション管理
- セッションタイムアウト: 24時間
- 複数デバイスからの同時ログイン: 許可
- セッション無効化時は自動的にログインページへリダイレクト

### 3.2 プロジェクト管理

#### REQ-PROJ-001: プロジェクト作成
- ユーザーは複数のBigQueryプロジェクト設定を作成可能
- 各プロジェクトに以下の情報を設定:
  - プロジェクト名（表示名）
  - プロジェクト説明
  - BigQuery プロジェクトID
  - デフォルトデータセット（任意）
  - AIプロバイダー（OpenAI / Gemini）
  - APIキー
  - 認証方法（サービスアカウントJSON / ADC）

#### REQ-PROJ-002: プロジェクト切り替え
- ダッシュボードからワンクリックでアクティブプロジェクトを切り替え可能
- 切り替え時は関連するチャットセッションも連動

#### REQ-PROJ-003: サービスアカウント接続テスト
- 設定保存前にBigQuery接続をテスト可能
- テスト結果にサービスアカウントメール、権限、アクセス可能データセットを表示

### 3.3 AIチャット機能

#### REQ-CHAT-001: 自然言語クエリ
- ユーザーが日本語または英語で質問を入力
- AIが適切なSQLクエリを生成・実行
- 結果をテーブル形式で表示

#### REQ-CHAT-002: リアルタイム進捗表示
- 処理ステップをリアルタイムで表示
- 思考プロセス（Reasoning）を紫色ボーダーで表示
- ポーリング間隔: 500ms

#### REQ-CHAT-003: データ可視化
- AIが最適なグラフタイプを提案
- 対応グラフ: 棒グラフ、折れ線グラフ、円グラフ、散布図、面グラフ
- 複数グラフの同時表示に対応
- グラフはPNG形式でダウンロード可能

#### REQ-CHAT-004: Python実行
- AIがPythonコードを生成し、サーバーサイドで実行
- 利用可能ライブラリ: pandas, numpy, matplotlib, seaborn, scikit-learn, scipy, statsmodels
- 実行タイムアウト: 30秒
- メモリ制限: 512MB

#### REQ-CHAT-005: CSVダウンロード
- クエリ結果をCSVファイルとしてダウンロード可能
- 文字コード: UTF-8 (BOM付き)

#### REQ-CHAT-006: セッション管理
- チャット履歴をセッション単位で保存
- セッションタイトルは最初の質問から自動生成
- 過去のセッションを選択して会話を再開可能

#### REQ-CHAT-007: プロジェクトメモリ
- プロジェクト横断で重要な情報を記憶
- 記憶した情報は全チャットセッションで参照可能

### 3.4 ダッシュボード

#### REQ-DASH-001: スキーマツリー表示
- BigQueryのデータセット・テーブル・カラム構造をツリー表示
- テーブルクリックでスキーマ詳細を表示

#### REQ-DASH-002: プロジェクト一覧
- ユーザーの全プロジェクトをカード形式で表示
- アクティブプロジェクトをハイライト

### 3.5 設定画面

#### REQ-SET-001: AIプロバイダー設定
- OpenAI / Gemini の切り替え
- APIキーの入力・検証・保存
- APIキーはマスク表示（最初と最後の4文字のみ表示）

#### REQ-SET-002: BigQuery設定
- プロジェクトID入力（サービスアカウントJSONから自動取得可能）
- デフォルトデータセット選択（ドロップダウン）
- 認証方法選択:
  - サービスアカウントJSONファイルアップロード
  - ADC（Cloud Run環境用）

#### REQ-SET-003: 接続テスト
- AI API接続テスト
- BigQuery接続テスト
- サービスアカウント権限確認

---

## 4. 非機能要件

### 4.1 パフォーマンス

| 項目 | 要件 |
|------|------|
| ページ読み込み時間 | 3秒以内 |
| AI応答開始時間 | 5秒以内（進捗表示開始まで） |
| 同時接続ユーザー数 | 100ユーザー |
| BigQueryクエリタイムアウト | 120秒 |

### 4.2 可用性

| 項目 | 要件 |
|------|------|
| 稼働率目標 | 99.5% |
| 計画メンテナンス | 事前通知の上、月1回まで |
| バックアップ | Cloud SQLの自動バックアップ（日次） |

### 4.3 セキュリティ

| 項目 | 要件 |
|------|------|
| 通信暗号化 | HTTPS/TLS 1.2以上必須 |
| 認証 | IAP + アプリケーション認証の二重認証 |
| SQLインジェクション対策 | パラメータ化クエリ + 読み取り専用クエリ強制 |
| APIキー保護 | データベースに暗号化保存 |
| Pythonサンドボックス | 制限されたグローバル変数、許可リスト方式 |

### 4.4 スケーラビリティ

| 項目 | 要件 |
|------|------|
| 水平スケーリング | Cloud Runの自動スケーリング対応 |
| 最小インスタンス数 | 1（コールドスタート回避） |
| 最大インスタンス数 | 10（調整可能） |
| Cloud SQL接続プール | インスタンスあたり最大100接続 |

### 4.5 保守性

| 項目 | 要件 |
|------|------|
| ログ出力 | Cloud Loggingへ構造化ログ出力 |
| エラー通知 | Cloud Monitoring アラート設定 |
| デプロイ | Dockerコンテナによるイミュータブルデプロイ |

---

## 5. 外部インターフェース要件

### 5.1 ユーザーインターフェース
- レスポンシブデザイン（デスクトップ優先）
- モダンなホワイトテーマ（紫/青のグラデーション）
- Bootstrapベースのコンポーネント
- Interフォント使用

### 5.2 外部API

| API | 用途 | 認証方式 |
|-----|------|----------|
| OpenAI API | GPT-4/GPT-4oによる自然言語処理 | APIキー |
| Google Gemini API | Gemini 2.0による自然言語処理 | APIキー |
| BigQuery API | データクエリ・スキーマ取得 | サービスアカウント / ADC |
| MCP Server | BigQueryとLLMの連携 | ローカル通信 |

### 5.3 データベーススキーマ

#### users テーブル
```sql
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

#### projects テーブル
```sql
CREATE TABLE projects (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id),
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
```

#### chat_sessions テーブル
```sql
CREATE TABLE chat_sessions (
    id SERIAL PRIMARY KEY,
    project_id INTEGER REFERENCES projects(id),
    title VARCHAR(200) DEFAULT 'New Chat',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

#### chat_history テーブル
```sql
CREATE TABLE chat_history (
    id SERIAL PRIMARY KEY,
    project_id INTEGER REFERENCES projects(id),
    session_id INTEGER REFERENCES chat_sessions(id),
    user_message TEXT NOT NULL,
    ai_response TEXT,
    query_result JSONB,
    reasoning_process TEXT,
    steps_count INTEGER DEFAULT 0,
    processing_time FLOAT DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

#### chat_tasks テーブル（Cloud Run対応）
```sql
CREATE TABLE chat_tasks (
    id VARCHAR(36) PRIMARY KEY,
    user_id INTEGER REFERENCES users(id),
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
```

#### project_memories テーブル
```sql
CREATE TABLE project_memories (
    id SERIAL PRIMARY KEY,
    project_id INTEGER REFERENCES projects(id),
    user_id INTEGER REFERENCES users(id),
    memory_key VARCHAR(100) NOT NULL,
    memory_value TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

---

## 6. 制約事項

### 6.1 技術的制約
- Python 3.11以上が必要
- PostgreSQL 15以上が必要
- Docker対応環境が必要
- GCPプロジェクトへのアクセス権限が必要

### 6.2 運用制約
- BigQueryへのクエリは読み取り専用（SELECT文のみ）
- Python実行は制限されたサンドボックス内
- ファイルアップロードサイズ: 10MB以下
- セッションあたりのチャット履歴: 1000件まで

### 6.3 前提条件
- ユーザーはIAPで認証済みの社内ユーザーであること
- BigQueryにアクセス可能なサービスアカウントが存在すること
- OpenAIまたはGemini APIの有効なAPIキーを保有していること

---

## 7. 受入基準

### 7.1 機能テスト
- [ ] ユーザー登録・ログインが正常に動作すること
- [ ] プロジェクト作成・編集・削除が正常に動作すること
- [ ] 自然言語クエリでBigQueryからデータ取得できること
- [ ] グラフが正しく表示されること
- [ ] チャット履歴が保存・復元されること
- [ ] CSV/PNGダウンロードが動作すること

### 7.2 非機能テスト
- [ ] Cloud Run環境でアプリケーションが起動すること
- [ ] Cloud SQLへの接続が正常に動作すること
- [ ] IAP認証が正しく機能すること
- [ ] 100同時ユーザーでの負荷テストをパスすること

### 7.3 セキュリティテスト
- [ ] IAP未認証ユーザーがアクセスできないこと
- [ ] SQLインジェクションが防止されていること
- [ ] APIキーがログに出力されないこと

---

## 8. 付録

### 8.1 関連ドキュメント
- architecture.md - システムアーキテクチャ図
- deployment-guide.md - デプロイ手順書
- operations-manual.md - 運用マニュアル
- security-specification.md - セキュリティ仕様書

### 8.2 変更履歴

| バージョン | 日付 | 変更内容 | 作成者 |
|-----------|------|----------|--------|
| 1.0 | 2025/11 | 初版作成 | - |
