"""
Setup script for stock prediction AI system
Handles initial environment setup and verification
"""

import subprocess
import sys
from pathlib import Path


def install_requirements():
    """Install required packages"""
    print("Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✓ Requirements installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to install requirements: {e}")
        return False


def create_directories():
    """Create necessary directories"""
    directories = [
        "data",
        "models", 
        "checkpoints",
        "results"
    ]
    
    print("Creating project directories...")
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✓ Created directory: {directory}")


def verify_installation():
    """Verify that all packages are properly installed"""
    required_packages = [
        'yfinance', 'torch', 'pandas', 'numpy', 
        'matplotlib', 'seaborn', 'sklearn', 'tqdm', 'plotly'
    ]
    
    print("\nVerifying package installation...")
    all_good = True
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package} - MISSING")
            all_good = False
    
    return all_good


def main():
    """Main setup process"""
    print("=" * 60)
    print("株式予測AI システムセットアップ")
    print("Stock Prediction AI System Setup")
    print("=" * 60)
    
    # Create directories
    create_directories()
    
    # Install requirements
    if not install_requirements():
        print("\nSetup failed. Please check error messages above.")
        return
    
    # Verify installation
    if not verify_installation():
        print("\nSome packages are missing. Please run:")
        print("pip install -r requirements.txt")
        return
    
    print("\n" + "=" * 60)
    print("✓ セットアップ完了！ (Setup Complete!)")
    print("=" * 60)
    print("\n次のステップ (Next Steps):")
    print("1. python main.py を実行してシステムを開始")
    print("   Run: python main.py to start the system")
    print("\n2. 完全パイプライン（選択肢1）で全自動実行")
    print("   Choose option 1 for complete automated pipeline")
    print("\n3. README.md で詳細な使用方法を確認")
    print("   Check README.md for detailed usage instructions")


if __name__ == "__main__":
    main()