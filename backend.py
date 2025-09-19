# ===================== ライブラリのインポート =====================
import torch                       # PyTorch のメインライブラリ
import torch.nn as nn               # ニューラルネットワーク用モジュール
import torchvision.models as models # 事前学習済みモデル（ResNet など）を使用するため
import torchvision.transforms as transforms  # 画像の前処理用
from PIL import Image               # 画像読み込み用ライブラリ
from fastapi import FastAPI, UploadFile, File  # FastAPI の Web アプリ用
import io                           # バイトデータを扱うための標準ライブラリ

# ===================== FastAPI アプリケーションの生成 =====================
app = FastAPI()  # FastAPI のインスタンス作成。これがアプリ全体のベース

# ===================== 画像前処理の定義 =====================
# ResNet18 に入力する画像サイズと正規化を設定
transform = transforms.Compose([
    transforms.Resize((128, 128)),       # 画像サイズを 128x128 に統一
    transforms.ToTensor(),               # PIL Image を Tensor に変換
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # -1〜1 に正規化
])

# ===================== モデルの定義 =====================
# 事前学習済みの ResNet18 をベースに使用
model = models.resnet18(weights=None)  # 事前学習済みの重みは使わない
num_features = model.fc.in_features        # 最終全結合層の入力特徴数を取得
model.fc = nn.Linear(num_features, 2)     # 2クラス分類用に全結合層を置き換え

# 学習済みモデルのロード
model_path = "waste_classifier_resnet18.pth"  # 保存済みモデルのファイルパス
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))  # CPU 用にロード
model.eval()  # 推論モードに切り替え（Dropout などを無効化）

# ===================== 推論用関数 =====================
def predict_image(image_bytes):
    """
    アップロードされた画像の推論を行う関数
    image_bytes: アップロードされた画像データ（バイト型）
    return: 分類結果（文字列）
    """
    # バイトデータを PIL Image に変換
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    
    # 画像前処理を適用
    img_t = transform(img)
    
    # バッチ次元を追加 (1, C, H, W)
    batch_t = torch.unsqueeze(img_t, 0)
    
    # 推論
    with torch.no_grad():  # 勾配計算を無効化（高速化＆メモリ節約）
        out = model(batch_t)
    
    # 最大値のインデックスを取得
    _, predicted = torch.max(out, 1)
    
    # クラス名に変換
    class_names = ['NonRecyclable', 'Recyclable']  # 0 → NonRecyclable, 1 → Recyclable
    # 予測後のラベル取得
    predicted_class = class_names[predicted.item()]  # predicted は outputs.argmax(1)
    return predicted_class 

# ===================== API エンドポイント =====================
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    """
    /predict/ に画像を POST すると分類結果を返す
    file: アップロードされた画像ファイル
    return: JSON 形式で分類結果
    """
    # 画像データを読み込み
    image_bytes = await file.read()
    
    # 推論関数を呼び出して結果を取得
    prediction = predict_image(image_bytes)
    
    # JSON 形式で返却
    return {"prediction": prediction}

# このスクリプトが直接実行された場合のみ、以下を実行する
if __name__ == "__main__":     
    import uvicorn               # uvicorn: FastAPI アプリを起動するサーバーを提供するライブラリ
    # uvicorn.run() で FastAPI アプリを起動
    uvicorn.run(
        "backend:app",           # 起動するアプリを指定 → backend.py 内の app インスタンス
        host="127.0.0.1",        # サーバーのホスト IP（ローカルホスト）
        port=8000,               # サーバーのポート番号
        reload=True              # True にするとコード変更を監視して自動リロード
    )

