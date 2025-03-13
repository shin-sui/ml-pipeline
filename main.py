import mlflow
from mlflow.tracking import MlflowClient

TRACKING_URI = "file:///Users/suizushinsaku/develop/patent/logs/mlflow/mlruns"
print(TRACKING_URI)
ARTIFACT_LOCATION = '/Users/suizushinsaku/develop/patent/logs//mlflow/mlruns/920544941022305717/cb8608b21d1e41dd8d87c55dfa85ac85/artifacts'
EXPERIMENT_NAME = 'demo'
# トラッキングサーバ（バックエンド）の場所を指定
mlflow.set_tracking_uri(TRACKING_URI)
xperiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)

# 特定のrun_idを指定
run_id = 'cb8608b21d1e41dd8d87c55dfa85ac85'

# MlflowClientのインスタンスを作成
client = MlflowClient()

# ラン情報を取得
run = client.get_run(run_id)

# メトリクス、パラメータ、タグを表示
print("Metrics:")
for key, value in run.data.metrics.items():
    print(f"  {key}: {value}")

print("\nParameters:")
for key, value in run.data.params.items():
    print(f"  {key}: {value}")

print("\nTags:")
for key, value in run.data.tags.items():
    print(f"  {key}: {value}")

# アーティファクトのリストを取得し、パスを表示
print("\nArtifacts:")
artifacts = client.list_artifacts(run_id)
for artifact in artifacts:
    print(f"  {artifact.path}")
