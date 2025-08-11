"""
Main execution script for stock prediction AI
Unified interface for all system components
"""

import os

import config


def print_banner():
    """Print system banner"""
    print("=" * 70)
    print("株式予測AI - トヨタ自動車デイトレード戦略システム")
    print("Stock Prediction AI - Toyota Day Trading Strategy System")
    print("=" * 70)
    print(f"Target: {config.TARGET_SYMBOL} (Toyota Motor Corporation)")
    print(f"Market Reference: {config.MARKET_SYMBOL} (Nikkei 225 ETF)")
    print(f"Prediction Horizon: {config.PREDICTION_HORIZON} step(s) ahead")
    print(f"Data Interval: {config.DATA_INTERVAL}")
    print("=" * 70)


def check_requirements():
    """Check if required packages are installed"""
    required_packages = [
        'yfinance', 'torch', 'pandas', 'numpy', 
        'matplotlib', 'seaborn', 'sklearn', 'tqdm'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("Missing required packages:")
        for package in missing_packages:
            print(f"  - {package}")
        print("\nPlease install requirements: pip install -r requirements.txt")
        return False
    
    return True


def prepare_data():
    """Data preparation pipeline"""
    from data_handler import StockDataHandler
    handler = StockDataHandler()
    handler.prepare_data()
    return True


def train_model():
    """Model training pipeline"""
    from train import main as train_main
    try:
        train_main()
        return True
    except Exception as e:
        print(f"Training failed: {e}")
        return False


def evaluate_model():
    """Model evaluation and backtesting"""
    from evaluate import main as evaluate_main
    try:
        evaluate_main()
        return True
    except Exception as e:
        print(f"Evaluation failed: {e}")
        return False


def visualize_results():
    """Create visualizations"""
    from visualize import ModelVisualizer
    try:
        visualizer = ModelVisualizer()
        visualizer.create_dashboard()
        return True
    except Exception as e:
        print(f"Visualization failed: {e}")
        return False


def run_inference():
    """Run real-time inference"""
    model_path = os.path.join(config.MODEL_SAVE_PATH, "best_model.pth")
    
    if not os.path.exists(model_path):
        print("Trained model not found. Please complete training first.")
        return False
    
    from inference import main as inference_main
    try:
        inference_main()
        return True
    except Exception as e:
        print(f"Inference failed: {e}")
        return False


def main():
    """Main execution pipeline"""
    print_banner()
    
    # Check requirements
    if not check_requirements():
        return
    
    print("\nSelect execution mode:")
    print("1. Complete pipeline (data → train → evaluate → visualize)")
    print("2. Data preparation only")
    print("3. Model training only")
    print("4. Model evaluation only") 
    print("5. Visualization only")
    print("6. Real-time inference")
    print("7. Quick test (single prediction)")
    
    choice = input("\nEnter your choice (1-7): ").strip()
    
    if choice == "1":
        steps = [prepare_data, train_model, evaluate_model, visualize_results]
        
        for step_func in steps:
            if not step_func():
                return
        
        print("\nPipeline completed successfully. Ready for inference.")
        
    elif choice == "2":
        prepare_data()
    elif choice == "3":
        train_model()
    elif choice == "4":
        evaluate_model()
    elif choice == "5":
        visualize_results()
    elif choice == "6":
        run_inference()
    elif choice == "7":
        # Quick test
        model_path = os.path.join(config.MODEL_SAVE_PATH, "best_model.pth")
        if os.path.exists(model_path):
            from inference import single_prediction
            single_prediction(model_path)
        else:
            print("No trained model found. Please train the model first.")
    else:
        print("Invalid choice")


if __name__ == "__main__":
    main()