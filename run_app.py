"""
株式予測AIアプリケーション起動スクリプト
Streamlitアプリを起動します
"""

import subprocess
import sys
import os
from pathlib import Path


def check_dependencies():
    """
    必要な依存関係をチェック
    """
    required_packages = ['streamlit', 'plotly', 'yfinance', 'torch', 'pandas']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    return missing_packages


def install_dependencies():
    """
    依存関係をインストール
    """
    print("必要な依存関係をインストールしています...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], 
                      check=True, cwd=Path(__file__).parent)
        print("✅ 依存関係のインストールが完了しました")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ 依存関係のインストールに失敗しました: {e}")
        return False


def main():
    """
    メイン実行関数
    """
    print("=" * 60)
    print("Stock Predictor - 株式予測AIシステム")
    print("Stock Prediction AI System")
    print("=" * 60)
    
    # カレントディレクトリを確認
    current_dir = Path(__file__).parent
    os.chdir(current_dir)
    
    # 依存関係をチェック
    missing = check_dependencies()
    if missing:
        print(f"不足している依存関係: {', '.join(missing)}")
        install_choice = input("依存関係を自動インストールしますか? (y/n): ").lower().strip()
        
        if install_choice == 'y':
            if not install_dependencies():
                print("依存関係のインストールに失敗しました。手動でインストールしてください:")
                print("pip install -r requirements.txt")
                return
        else:
            print("手動で依存関係をインストールしてください:")
            print("pip install -r requirements.txt")
            return
    
    # モデルの存在確認
    model_path = Path("models/best_model.pth")
    if not model_path.exists():
        print("\n⚠️  警告: 訓練済みモデルが見つかりません")
        print("アプリケーションを起動する前に、以下のコマンドでモデルを訓練してください:")
        print("python main.py")
        print("または、選択肢1（完全なパイプライン）を実行してください。")
        print()
        
        continue_choice = input("モデルなしでアプリケーションを起動しますか? (y/n): ").lower().strip()
        if continue_choice != 'y':
            print("アプリケーションの起動をキャンセルしました。")
            return
    
    # Streamlitアプリを起動
    print("\n🚀 Streamlitアプリケーションを起動しています...")
    print("ブラウザで http://localhost:8501 を開いてください")
    print("アプリケーションを停止するには Ctrl+C を押してください")
    
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "app.py",
            "--server.port=8501",
            "--server.address=localhost",
            "--browser.gatherUsageStats=false"
        ], cwd=current_dir)
    except KeyboardInterrupt:
        print("\n✅ アプリケーションを終了しました")
    except Exception as e:
        print(f"\n❌ アプリケーションの起動に失敗しました: {e}")


if __name__ == "__main__":
    main()