"""
LLM-powered Trading Advisor using ChatGPT API
Provides professional stock analysis and investment advice based on prediction history and market context
"""

import openai
import json
import time
import os
from datetime import datetime
from typing import Dict, List, Optional, Any
import logging
from dataclasses import dataclass, asdict

import config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class AdvisoryMessage:
    """Advisory message data structure"""
    timestamp: str
    user_question: str
    advisor_response: str
    market_context: Dict[str, Any]
    prediction_context: Dict[str, Any]
    confidence_score: float


class LLMTradingAdvisor:
    """
    ChatGPT-powered trading advisor that provides investment analysis and recommendations
    """
    
    def __init__(self):
        # Initialize OpenAI client
        self.client = None
        self.initialize_openai()
        
        # Advisory history
        self.advisory_history: List[AdvisoryMessage] = []
        self.load_advisory_history()
        
        # Rate limiting
        self.last_api_call = 0
        
        # Prompt templates
        self.system_prompt = self._create_system_prompt()
    
    def initialize_openai(self):
        """Initialize OpenAI client with API key"""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.warning("OPENAI_API_KEY environment variable not set. LLM advisor will be disabled.")
            return False
        
        try:
            self.client = openai.OpenAI(api_key=api_key)
            logger.info("OpenAI client initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
            return False
    
    def _create_system_prompt(self) -> str:
        """Create the system prompt for the trading advisor"""
        return """あなたは日本の株式市場に特化した専門的な投資アナリストです。以下の特徴を持って回答してください：

【専門性】
- 日本株式市場（特にトヨタ自動車7203.T）の専門知識
- テクニカル分析とファンダメンタル分析の両方を活用
- デイトレード戦略に特化したアドバイス

【分析アプローチ】
- AI予測結果と市場データを総合的に評価
- リスク管理の重要性を常に考慮
- 具体的で実用的なアドバイスを提供

【回答スタイル】
- 簡潔で分かりやすい日本語
- 根拠を明確に示す
- リスクと機会の両面を提示
- 投資判断は最終的にユーザーに委ねる

【重要な注意】
- これは教育・研究目的の分析です
- 実際の投資判断は自己責任で行ってください
- 金融投資にはリスクが伴います

提供されるデータを基に、専門的で有用な分析とアドバイスを提供してください。"""

    def _format_prediction_context(self, prediction_history: List[Dict]) -> str:
        """Format prediction history for LLM context"""
        if not prediction_history:
            return "予測履歴: データなし"
        
        recent_predictions = prediction_history[-config.LLM_HISTORY_LIMIT:]
        
        context = "【最近の予測履歴】\n"
        
        # Summary statistics
        signals = [p.get('signal', 'HOLD') for p in recent_predictions]
        confidences = [p.get('confidence', 0) for p in recent_predictions]
        price_changes = [p.get('price_change_pct', 0) for p in recent_predictions]
        
        buy_count = signals.count('BUY')
        sell_count = signals.count('SELL')
        hold_count = signals.count('HOLD')
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
        avg_price_change = sum(price_changes) / len(price_changes) if price_changes else 0
        
        context += f"- 分析期間: 直近{len(recent_predictions)}回の予測\n"
        context += f"- シグナル分布: BUY({buy_count}), SELL({sell_count}), HOLD({hold_count})\n"
        context += f"- 平均信頼度: {avg_confidence:.2f}\n"
        context += f"- 平均価格変動率: {avg_price_change:.2%}\n\n"
        
        # Recent predictions detail
        context += "【直近5回の予測詳細】\n"
        for i, pred in enumerate(recent_predictions[-5:], 1):
            timestamp = pred.get('timestamp', 'N/A')
            signal = pred.get('signal', 'HOLD')
            current_price = pred.get('current_price', 0)
            predicted_price = pred.get('predicted_price', 0)
            confidence = pred.get('confidence', 0)
            price_change = pred.get('price_change_pct', 0)
            
            context += f"{i}. {timestamp} - {signal} (信頼度:{confidence:.2f})\n"
            context += f"   現在価格: ¥{current_price:.0f} → 予測価格: ¥{predicted_price:.0f} ({price_change:.2%})\n"
        
        return context
    
    def _format_market_context(self, market_data: Dict) -> str:
        """Format market data for LLM context"""
        if not market_data:
            return "市場データ: データなし"
        
        context = "【市場データ】\n"
        
        # Current market status
        if 'current_price' in market_data:
            context += f"- 現在価格: ¥{market_data['current_price']:.0f}\n"
        
        if 'price_change_pct' in market_data:
            change_pct = market_data['price_change_pct']
            direction = "上昇" if change_pct > 0 else "下降" if change_pct < 0 else "横ばい"
            context += f"- 価格変動: {change_pct:.2%} ({direction})\n"
        
        if 'volume' in market_data:
            context += f"- 出来高: {market_data['volume']:,}\n"
        
        if 'market_trend' in market_data:
            context += f"- 市場トレンド: {market_data['market_trend']}\n"
        
        # Technical indicators if available
        if 'volatility' in market_data:
            context += f"- ボラティリティ: {market_data['volatility']:.2%}\n"
        
        return context
    
    def _wait_for_rate_limit(self):
        """Implement rate limiting to avoid API limits"""
        time_since_last_call = time.time() - self.last_api_call
        if time_since_last_call < config.LLM_RATE_LIMIT_DELAY:
            wait_time = config.LLM_RATE_LIMIT_DELAY - time_since_last_call
            time.sleep(wait_time)
    
    def get_investment_advice(
        self,
        user_question: str,
        prediction_history: List[Dict],
        market_data: Dict,
        include_context: bool = True
    ) -> Optional[str]:
        """
        Get investment advice from ChatGPT based on current market context
        
        Args:
            user_question: User's question about investment strategy
            prediction_history: Recent prediction history
            market_data: Current market data and trends
            include_context: Whether to include prediction and market context
            
        Returns:
            Advisory response or None if API call fails
        """
        if not self.client:
            return "LLMアドバイザーが利用できません。OPENAI_API_KEYを設定してください。"
        
        try:
            # Implement rate limiting
            self._wait_for_rate_limit()
            
            # Build context message
            messages = [{"role": "system", "content": self.system_prompt}]
            
            if include_context:
                context_message = ""
                
                # Add prediction context
                pred_context = self._format_prediction_context(prediction_history)
                context_message += pred_context + "\n\n"
                
                # Add market context
                market_context = self._format_market_context(market_data)
                context_message += market_context + "\n\n"
                
                # Add context as separate message
                context_message += f"【ユーザーからの質問】\n{user_question}"
                messages.append({"role": "user", "content": context_message})
            else:
                messages.append({"role": "user", "content": user_question})
            
            # Make API call with retries
            for attempt in range(config.LLM_MAX_RETRIES):
                try:
                    response = self.client.chat.completions.create(
                        model=config.LLM_MODEL,
                        messages=messages,
                        max_tokens=config.LLM_MAX_TOKENS,
                        temperature=config.LLM_TEMPERATURE
                    )
                    
                    self.last_api_call = time.time()
                    
                    advice = response.choices[0].message.content
                    
                    # Store in advisory history
                    advisory_msg = AdvisoryMessage(
                        timestamp=datetime.now().isoformat(),
                        user_question=user_question,
                        advisor_response=advice,
                        market_context=market_data,
                        prediction_context={
                            'recent_predictions_count': len(prediction_history),
                            'avg_confidence': sum(p.get('confidence', 0) for p in prediction_history[-10:]) / min(10, len(prediction_history)) if prediction_history else 0
                        },
                        confidence_score=0.8  # Default confidence for LLM advice
                    )
                    
                    self.advisory_history.append(advisory_msg)
                    self._trim_advisory_history()
                    self.save_advisory_history()
                    
                    return advice
                    
                except openai.RateLimitError:
                    logger.warning(f"Rate limit exceeded, attempt {attempt + 1}")
                    if attempt < config.LLM_MAX_RETRIES - 1:
                        time.sleep(2 ** attempt)  # Exponential backoff
                    continue
                except openai.APITimeoutError:
                    logger.warning(f"API timeout, attempt {attempt + 1}")
                    if attempt < config.LLM_MAX_RETRIES - 1:
                        time.sleep(1)
                    continue
                except Exception as e:
                    logger.error(f"API call failed on attempt {attempt + 1}: {e}")
                    if attempt < config.LLM_MAX_RETRIES - 1:
                        time.sleep(1)
                    continue
            
            return "申し訳ございません。現在アドバイザーサービスが利用できません。しばらく後に再度お試しください。"
            
        except Exception as e:
            logger.error(f"Unexpected error in get_investment_advice: {e}")
            return "予期しないエラーが発生しました。システム管理者にお問い合わせください。"
    
    def get_quick_analysis(self, prediction_data: Dict, market_data: Dict) -> str:
        """
        Get a quick analysis of current market situation and prediction
        
        Args:
            prediction_data: Latest prediction data
            market_data: Current market data
            
        Returns:
            Quick analysis summary
        """
        question = "現在の市場状況と最新の予測結果について、簡潔な分析とアドバイスをお願いします。"
        
        response = self.get_investment_advice(
            user_question=question,
            prediction_history=[prediction_data] if prediction_data else [],
            market_data=market_data,
            include_context=True
        )
        
        return response or "分析データが不足しています。"
    
    def _trim_advisory_history(self):
        """Trim advisory history to configured limit"""
        if len(self.advisory_history) > config.ADVISORY_HISTORY_LIMIT:
            self.advisory_history = self.advisory_history[-config.ADVISORY_HISTORY_LIMIT:]
    
    def save_advisory_history(self):
        """Save advisory history to file"""
        try:
            os.makedirs(config.DATA_SAVE_PATH, exist_ok=True)
            history_file = os.path.join(config.DATA_SAVE_PATH, "advisory_history.json")
            
            # Convert to serializable format
            serializable_history = [asdict(msg) for msg in self.advisory_history]
            
            with open(history_file, 'w', encoding='utf-8') as f:
                json.dump(serializable_history, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save advisory history: {e}")
    
    def load_advisory_history(self):
        """Load advisory history from file"""
        try:
            history_file = os.path.join(config.DATA_SAVE_PATH, "advisory_history.json")
            
            if os.path.exists(history_file):
                with open(history_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                self.advisory_history = [AdvisoryMessage(**msg) for msg in data]
                logger.info(f"Loaded {len(self.advisory_history)} advisory messages")
            else:
                self.advisory_history = []
                
        except Exception as e:
            logger.error(f"Failed to load advisory history: {e}")
            self.advisory_history = []
    
    def get_advisory_history(self, limit: Optional[int] = None) -> List[AdvisoryMessage]:
        """
        Get advisory history
        
        Args:
            limit: Maximum number of messages to return
            
        Returns:
            List of advisory messages
        """
        if limit:
            return self.advisory_history[-limit:]
        return self.advisory_history
    
    def is_available(self) -> bool:
        """Check if LLM advisor is available"""
        return self.client is not None


def create_advisor_instance() -> LLMTradingAdvisor:
    """Factory function to create advisor instance"""
    return LLMTradingAdvisor()