"""
株式予測AIアプリケーション - トヨタ自動車デイトレード戦略
動的データ取得、リアルタイム予測、分析データ表示、履歴参照機能を統合
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
import os
from datetime import datetime, timedelta
import time
import threading
import queue
from typing import Dict, List, Optional

import config
from data_handler import StockDataHandler
from inference import RealTimePredictor
from model import StockPredictionTransformer


class StockPredictionApp:
    """
    株式予測アプリケーションのメインクラス
    """
    def __init__(self):
        self.data_handler = StockDataHandler()
        self.predictor = None
        self.prediction_queue = queue.Queue()
        self.is_running = False
        self.prediction_thread = None
        
        # アプリケーション状態
        if 'prediction_history' not in st.session_state:
            st.session_state.prediction_history = []
        if 'market_data' not in st.session_state:
            st.session_state.market_data = None
        if 'analytics_data' not in st.session_state:
            st.session_state.analytics_data = {}
        
        # 初期化
        self.initialize_predictor()
    
    def initialize_predictor(self):
        """
        予測モデルを初期化
        """
        model_path = os.path.join(config.MODEL_SAVE_PATH, "best_model.pth")
        
        if os.path.exists(model_path):
            try:
                self.predictor = RealTimePredictor(model_path)
                return True
            except Exception as e:
                st.error(f"モデルの読み込みに失敗しました: {e}")
                return False
        else:
            st.warning("訓練済みモデルが見つかりません。まず訓練を実行してください。")
            return False
    
    def get_latest_prediction(self) -> Optional[Dict]:
        """
        最新の予測を取得
        """
        if not self.predictor:
            return None
            
        try:
            prediction = self.predictor.get_current_prediction()
            market_summary = self.predictor.get_market_summary()
            
            # セッション状態を更新
            st.session_state.prediction_history.append({
                **prediction,
                'market_summary': market_summary
            })
            
            # 履歴を最新100件に制限
            if len(st.session_state.prediction_history) > 100:
                st.session_state.prediction_history = st.session_state.prediction_history[-100:]
            
            return prediction
            
        except Exception as e:
            st.error(f"予測の取得に失敗しました: {e}")
            return None
    
    def calculate_analytics(self) -> Dict:
        """
        分析データを計算
        """
        if not st.session_state.prediction_history:
            return {}
        
        history = st.session_state.prediction_history
        recent_predictions = history[-20:] if len(history) >= 20 else history
        
        # 予測精度の計算（簡易版）
        signals = [p['signal'] for p in recent_predictions]
        confidences = [p['confidence'] for p in recent_predictions]
        price_changes = [p['price_change_pct'] for p in recent_predictions]
        
        analytics = {
            'total_predictions': len(history),
            'recent_accuracy': len([s for s in signals if s != 'HOLD']) / len(signals) if signals else 0,
            'avg_confidence': np.mean(confidences) if confidences else 0,
            'avg_price_change': np.mean(price_changes) if price_changes else 0,
            'signal_distribution': {
                'BUY': signals.count('BUY'),
                'SELL': signals.count('SELL'),
                'HOLD': signals.count('HOLD')
            },
            'last_updated': datetime.now().isoformat()
        }
        
        return analytics
    
    def get_market_data_for_chart(self, period: str = "5d") -> pd.DataFrame:
        """
        チャート用の市場データを取得
        """
        try:
            # 生データを取得（正規化前）
            raw_data = self.data_handler.get_latest_data_raw()
            
            # 期間に応じてデータをフィルタ
            end_time = raw_data.index.max()
            if period == "1d":
                start_time = end_time - timedelta(days=1)
            elif period == "5d":
                start_time = end_time - timedelta(days=5)
            elif period == "1m":
                start_time = end_time - timedelta(days=30)
            else:
                start_time = raw_data.index.min()
            
            filtered_data = raw_data[raw_data.index >= start_time]
            
            return filtered_data
            
        except Exception as e:
            st.error(f"市場データの取得に失敗しました: {e}")
            return pd.DataFrame()
    
    def create_price_chart(self, data: pd.DataFrame) -> go.Figure:
        """
        価格チャートを作成
        """
        fig = make_subplots(
            rows=2, cols=1,
            row_heights=[0.7, 0.3],
            vertical_spacing=0.1,
            subplot_titles=('価格チャート', '出来高')
        )
        
        # ローソク足チャート
        fig.add_trace(
            go.Candlestick(
                x=data.index,
                open=data['Open'],
                high=data['High'],
                low=data['Low'],
                close=data['Close'],
                name='トヨタ自動車'
            ),
            row=1, col=1
        )
        
        # 出来高
        fig.add_trace(
            go.Bar(
                x=data.index,
                y=data['Volume'],
                name='出来高',
                marker_color='rgba(0,100,80,0.6)'
            ),
            row=2, col=1
        )
        
        # 予測ポイントを追加
        if st.session_state.prediction_history:
            for pred in st.session_state.prediction_history[-10:]:  # 最新10件
                pred_time = pd.to_datetime(pred['timestamp'])
                if pred_time in data.index:
                    color = 'green' if pred['signal'] == 'BUY' else 'red' if pred['signal'] == 'SELL' else 'orange'
                    fig.add_trace(
                        go.Scatter(
                            x=[pred_time],
                            y=[pred['predicted_price']],
                            mode='markers',
                            marker=dict(size=10, color=color, symbol='diamond'),
                            name=f'{pred["signal"]} 予測',
                            showlegend=False
                        ),
                        row=1, col=1
                    )
        
        fig.update_layout(
            title=f'{config.TARGET_SYMBOL} - リアルタイム価格チャート',
            xaxis_title='時刻',
            yaxis_title='価格 (¥)',
            height=600,
            xaxis_rangeslider_visible=False
        )
        
        return fig
    
    def create_analytics_chart(self, analytics: Dict) -> go.Figure:
        """
        分析チャートを作成
        """
        if not analytics or not st.session_state.prediction_history:
            return go.Figure()
        
        # 信頼度の時系列
        history = st.session_state.prediction_history
        timestamps = [pd.to_datetime(h['timestamp']) for h in history]
        confidences = [h['confidence'] for h in history]
        price_changes = [h['price_change_pct'] * 100 for h in history]
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('予測信頼度推移', 'シグナル分布', '価格変化率推移', '予測精度'),
            specs=[[{"secondary_y": False}, {"type": "pie"}],
                   [{"secondary_y": False}, {"type": "indicator"}]]
        )
        
        # 信頼度推移
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=confidences,
                mode='lines+markers',
                name='信頼度',
                line=dict(color='blue')
            ),
            row=1, col=1
        )
        
        # シグナル分布
        if analytics['signal_distribution']:
            fig.add_trace(
                go.Pie(
                    labels=list(analytics['signal_distribution'].keys()),
                    values=list(analytics['signal_distribution'].values()),
                    name='シグナル分布'
                ),
                row=1, col=2
            )
        
        # 価格変化率推移
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=price_changes,
                mode='lines',
                name='価格変化率(%)',
                line=dict(color='green')
            ),
            row=2, col=1
        )
        
        # 予測精度インジケータ
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=analytics['recent_accuracy'] * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "予測精度 (%)"},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 50], 'color': "lightgray"},
                        {'range': [50, 80], 'color': "yellow"},
                        {'range': [80, 100], 'color': "green"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ),
            row=2, col=2
        )
        
        fig.update_layout(height=600, title_text="分析ダッシュボード")
        return fig


def main():
    """
    Streamlitアプリケーションのメイン関数
    """
    st.set_page_config(
        page_title="Stock Predictor",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # アプリケーション初期化
    if 'app' not in st.session_state:
        st.session_state.app = StockPredictionApp()
    
    app = st.session_state.app
    
    # タイトル
    st.title("📈 Stock Predictor - 株式予測AIシステム")
    st.markdown("---")
    
    # サイドバー
    with st.sidebar:
        st.header("⚙️ 設定")
        
        # 自動更新設定
        auto_refresh = st.checkbox("自動更新 (30秒間隔)", value=False)
        
        # 表示期間設定
        display_period = st.selectbox(
            "表示期間",
            options=["1d", "5d", "1m"],
            index=1,
            format_func=lambda x: {"1d": "1日", "5d": "5日", "1m": "1ヶ月"}[x]
        )
        
        # 手動更新ボタン
        if st.button("🔄 データ更新", type="primary"):
            st.rerun()
        
        st.markdown("---")
        
        # システム状態
        st.subheader("📊 システム状態")
        if app.predictor:
            st.success("✅ モデル読み込み済み")
        else:
            st.error("❌ モデル未読み込み")
        
        # 最新予測情報
        if st.session_state.prediction_history:
            latest = st.session_state.prediction_history[-1]
            st.metric(
                "最新シグナル",
                latest['signal'],
                f"{latest['confidence']:.2f}"
            )
            st.metric(
                "現在価格",
                f"¥{latest['current_price']:.0f}",
                f"{latest['price_change_pct']:.2%}"
            )
    
    # メインコンテンツ
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # 価格チャート
        st.subheader("💹 リアルタイム価格チャート")
        
        try:
            market_data = app.get_market_data_for_chart(display_period)
            if not market_data.empty:
                price_chart = app.create_price_chart(market_data)
                st.plotly_chart(price_chart, use_container_width=True)
            else:
                st.warning("市場データを取得できませんでした")
        except Exception as e:
            st.error(f"チャートの表示に失敗しました: {e}")
    
    with col2:
        # 最新予測とアクション
        st.subheader("🔮 次の5分足予測")
        
        if st.button("🎯 予測実行", type="primary", use_container_width=True):
            with st.spinner("予測を実行中..."):
                prediction = app.get_latest_prediction()
                if prediction:
                    st.success("予測完了!")
                    st.rerun()
        
        # 最新予測結果
        if st.session_state.prediction_history:
            latest_pred = st.session_state.prediction_history[-1]
            
            # シグナル表示
            signal_color = {
                'BUY': '🟢',
                'SELL': '🔴',
                'HOLD': '🟡'
            }
            
            st.markdown(f"""
            ### {signal_color.get(latest_pred['signal'], '⚪')} {latest_pred['signal']}
            
            **予測価格**: ¥{latest_pred['predicted_price']:.0f}  
            **変化率**: {latest_pred['price_change_pct']:.2%}  
            **信頼度**: {latest_pred['confidence']:.2f}  
            **予測時刻**: {pd.to_datetime(latest_pred['timestamp']).strftime('%H:%M:%S')}
            """)
        
        # 分析データ
        st.markdown("---")
        st.subheader("📈 分析データ")
        
        analytics = app.calculate_analytics()
        if analytics:
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("総予測回数", analytics['total_predictions'])
                st.metric("平均信頼度", f"{analytics['avg_confidence']:.2f}")
            with col_b:
                st.metric("予測精度", f"{analytics['recent_accuracy']:.1%}")
                st.metric("平均変化率", f"{analytics['avg_price_change']:.2%}")
    
    # 分析チャート（全幅）
    if st.session_state.prediction_history:
        st.markdown("---")
        st.subheader("📊 詳細分析")
        
        analytics = app.calculate_analytics()
        analytics_chart = app.create_analytics_chart(analytics)
        if analytics_chart.data:
            st.plotly_chart(analytics_chart, use_container_width=True)
    
    # 予測履歴テーブル
    if st.session_state.prediction_history:
        st.markdown("---")
        st.subheader("📋 予測履歴")
        
        # 履歴データをDataFrameに変換
        history_df = pd.DataFrame(st.session_state.prediction_history)
        history_df['timestamp'] = pd.to_datetime(history_df['timestamp'])
        history_df = history_df.sort_values('timestamp', ascending=False).head(20)
        
        # 表示用にフォーマット
        display_df = history_df[['timestamp', 'signal', 'current_price', 'predicted_price', 'price_change_pct', 'confidence']].copy()
        display_df.columns = ['時刻', 'シグナル', '現在価格', '予測価格', '変化率', '信頼度']
        display_df['現在価格'] = display_df['現在価格'].apply(lambda x: f"¥{x:.0f}")
        display_df['予測価格'] = display_df['予測価格'].apply(lambda x: f"¥{x:.0f}")
        display_df['変化率'] = display_df['変化率'].apply(lambda x: f"{x:.2%}")
        display_df['信頼度'] = display_df['信頼度'].apply(lambda x: f"{x:.2f}")
        
        st.dataframe(display_df, use_container_width=True)
    
    # 自動更新
    if auto_refresh:
        time.sleep(30)
        st.rerun()


if __name__ == "__main__":
    main()