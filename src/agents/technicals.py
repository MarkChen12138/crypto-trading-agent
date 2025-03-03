import math
import json
import numpy as np
import pandas as pd
from langchain_core.messages import HumanMessage
from src.agents.state import AgentState, show_agent_reasoning, show_workflow_status

class TechnicalAnalysisAgent:
    """
    技术分析代理 - 综合多种交易策略进行技术分析:
    1. 趋势跟踪
    2. 均值回归
    3. 动量分析
    4. 波动率分析
    5. 支撑/阻力位识别
    6. 量价关系分析
    """
    
    def __init__(self):
        """初始化技术分析代理"""
        self.prices_df = None
        self.symbol = None
        
    def analyze(self, state: AgentState):
        """执行技术分析"""
        show_workflow_status("技术分析代理")
    
        # 增加错误处理，确保即使缺少元数据也能正常工作
        if "metadata" in state and "show_reasoning" in state.get("metadata", {}):
            show_reasoning = state["metadata"]["show_reasoning"]
        else:
            show_reasoning = False  # 默认不显示推理过程
        
        data = state.get("data", {})
        
        # 获取基础数据
        self.symbol = data.get("symbol", "BTC/USDT")
        prices = data.get("prices", [])
        
        # 转换价格数据为DataFrame
        self.prices_df = self._convert_to_dataframe(prices)
        
        # 检查数据是否足够
        if self.prices_df.empty or len(self.prices_df) < 14:  # 至少需要14天数据
            # 返回中性信号
            message_content = {
                "signal": "neutral",
                "confidence": "50%",
                "reasoning": "数据不足，无法进行可靠的技术分析"
            }
            
            message = HumanMessage(
                content=json.dumps(message_content),
                name="technical_analyst_agent",
            )
            
            if show_reasoning:
                show_agent_reasoning(message_content, "技术分析代理")
                
            show_workflow_status("技术分析代理", "completed")
            return {
                "messages": [message],
                "data": data
            }
            
        # 执行各种策略分析
        strategies = {}
        
        # 1. 趋势跟踪策略
        strategies["trend_following"] = self._analyze_trend()
        
        # 2. 均值回归策略
        strategies["mean_reversion"] = self._analyze_mean_reversion()
        
        # 3. 动量策略
        strategies["momentum"] = self._analyze_momentum()
        
        # 4. 波动率策略
        strategies["volatility"] = self._analyze_volatility()
        
        # 5. 支撑阻力位分析
        strategies["support_resistance"] = self._analyze_support_resistance()
        
        # 6. 量价关系分析
        strategies["volume_price"] = self._analyze_volume_price()
        
        # 计算综合信号
        overall_signal, confidence = self._calculate_combined_signal(strategies)
        
        # 准备详细分析报告
        analysis_report = {
            "signal": overall_signal,
            "confidence": f"{round(confidence * 100)}%",
            "strategy_signals": {
                "trend_following": {
                    "signal": strategies["trend_following"]["signal"],
                    "confidence": f"{round(strategies['trend_following']['confidence'] * 100)}%",
                    "metrics": self._normalize_metrics(strategies["trend_following"]["metrics"])
                },
                "mean_reversion": {
                    "signal": strategies["mean_reversion"]["signal"],
                    "confidence": f"{round(strategies['mean_reversion']['confidence'] * 100)}%",
                    "metrics": self._normalize_metrics(strategies["mean_reversion"]["metrics"])
                },
                "momentum": {
                    "signal": strategies["momentum"]["signal"],
                    "confidence": f"{round(strategies['momentum']['confidence'] * 100)}%",
                    "metrics": self._normalize_metrics(strategies["momentum"]["metrics"])
                },
                "volatility": {
                    "signal": strategies["volatility"]["signal"],
                    "confidence": f"{round(strategies['volatility']['confidence'] * 100)}%",
                    "metrics": self._normalize_metrics(strategies["volatility"]["metrics"])
                },
                "support_resistance": {
                    "signal": strategies["support_resistance"]["signal"],
                    "confidence": f"{round(strategies['support_resistance']['confidence'] * 100)}%",
                    "metrics": self._normalize_metrics(strategies["support_resistance"]["metrics"])
                },
                "volume_price": {
                    "signal": strategies["volume_price"]["signal"],
                    "confidence": f"{round(strategies['volume_price']['confidence'] * 100)}%",
                    "metrics": self._normalize_metrics(strategies["volume_price"]["metrics"])
                }
            }
        }
        
        # 创建消息
        message = HumanMessage(
            content=json.dumps(analysis_report),
            name="technical_analyst_agent",
        )
        
        if show_reasoning:
            show_agent_reasoning(analysis_report, "技术分析代理")
            
        show_workflow_status("技术分析代理", "completed")
        return {
            "messages": [message],
            "data": data
        }
        
    def _convert_to_dataframe(self, prices):
        """将价格数据转换为DataFrame"""
        if isinstance(prices, pd.DataFrame):
            return prices
            
        df = pd.DataFrame(prices)
        
        # 处理时间列
        if 'timestamp' in df.columns:
            if isinstance(df['timestamp'].iloc[0], str):
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                
        # 确保必要的列存在
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in required_columns:
            if col not in df.columns:
                if col == 'close' and 'price' in df.columns:
                    df['close'] = df['price']
                elif col == 'volume' and 'volume_24h' in df.columns:
                    df['volume'] = df['volume_24h']
                else:
                    df[col] = 0  # 添加默认值
                    
        # 计算基本技术指标
        self._add_technical_indicators(df)
        
        return df
        
    def _add_technical_indicators(self, df):
        """计算基本技术指标"""
        # 移动平均线
        df['sma5'] = df['close'].rolling(window=5).mean()
        df['sma10'] = df['close'].rolling(window=10).mean()
        df['sma20'] = df['close'].rolling(window=20).mean()
        df['sma50'] = df['close'].rolling(window=50).mean()
        df['sma200'] = df['close'].rolling(window=200).mean()
        
        df['ema5'] = df['close'].ewm(span=5, adjust=False).mean()
        df['ema10'] = df['close'].ewm(span=10, adjust=False).mean()
        df['ema20'] = df['close'].ewm(span=20, adjust=False).mean()
        df['ema50'] = df['close'].ewm(span=50, adjust=False).mean()
        
        # MACD
        df['ema12'] = df['close'].ewm(span=12, adjust=False).mean()
        df['ema26'] = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = df['ema12'] - df['ema26']
        df['signal_line'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_histogram'] = df['macd'] - df['signal_line']
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi14'] = 100 - (100 / (1 + rs))
        
        # 布林带
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        bb_std = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        
        # 价格变化率
        df['price_change_1d'] = df['close'].pct_change(periods=1)
        df['price_change_3d'] = df['close'].pct_change(periods=3)
        df['price_change_7d'] = df['close'].pct_change(periods=7)
        
        # ATR (Average True Range)
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift())
        low_close = abs(df['low'] - df['close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr14'] = tr.rolling(window=14).mean()
        
        # 成交量变化
        df['volume_change_1d'] = df['volume'].pct_change(periods=1)
        df['volume_sma5'] = df['volume'].rolling(window=5).mean()
        df['volume_sma20'] = df['volume'].rolling(window=20).mean()
        
    def _analyze_trend(self):
        """趋势跟踪策略分析"""
        df = self.prices_df
        
        # 检查移动平均线
        try:
            # 短期趋势（快速移动平均线交叉）
            ema_cross_bullish = df['ema5'].iloc[-1] > df['ema20'].iloc[-1] and df['ema5'].iloc[-2] <= df['ema20'].iloc[-2]
            ema_cross_bearish = df['ema5'].iloc[-1] < df['ema20'].iloc[-1] and df['ema5'].iloc[-2] >= df['ema20'].iloc[-2]
            
            # 中期趋势（移动平均线排列）
            bullish_alignment = df['ema5'].iloc[-1] > df['ema10'].iloc[-1] > df['ema20'].iloc[-1]
            bearish_alignment = df['ema5'].iloc[-1] < df['ema10'].iloc[-1] < df['ema20'].iloc[-1]
            
            # 长期趋势（基于50天移动平均线）
            long_term_bullish = df['close'].iloc[-1] > df['sma50'].iloc[-1]
            long_term_bearish = df['close'].iloc[-1] < df['sma50'].iloc[-1]
            
            # 高低点分析（检查是否形成更高的高点和更高的低点）
            recent_highs = df['high'].iloc[-15:].rolling(window=5).max()
            recent_lows = df['low'].iloc[-15:].rolling(window=5).min()
            higher_highs = recent_highs.iloc[-1] > recent_highs.iloc[-6]
            higher_lows = recent_lows.iloc[-1] > recent_lows.iloc[-6]
            lower_highs = recent_highs.iloc[-1] < recent_highs.iloc[-6]
            lower_lows = recent_lows.iloc[-1] < recent_lows.iloc[-6]
            
            # 整合信号
            bullish_signals = sum([ema_cross_bullish, bullish_alignment, long_term_bullish, higher_highs and higher_lows])
            bearish_signals = sum([ema_cross_bearish, bearish_alignment, long_term_bearish, lower_highs and lower_lows])
            
            # 确定信号
            if bullish_signals > bearish_signals:
                signal = "bullish"
                confidence = min(0.5 + bullish_signals * 0.1, 0.9)
            elif bearish_signals > bullish_signals:
                signal = "bearish"
                confidence = min(0.5 + bearish_signals * 0.1, 0.9)
            else:
                signal = "neutral"
                confidence = 0.5
                
            # 准备指标数据
            metrics = {
                "ema_cross_bullish": ema_cross_bullish,
                "ema_cross_bearish": ema_cross_bearish,
                "bullish_alignment": bullish_alignment,
                "bearish_alignment": bearish_alignment,
                "long_term_bullish": long_term_bullish,
                "long_term_bearish": long_term_bearish,
                "higher_highs_higher_lows": higher_highs and higher_lows,
                "lower_highs_lower_lows": lower_highs and lower_lows,
                "price_vs_sma50": (df['close'].iloc[-1] / df['sma50'].iloc[-1] - 1) * 100
            }
            
            return {
                "signal": signal,
                "confidence": confidence,
                "metrics": metrics
            }
            
        except Exception as e:
            print(f"趋势分析出错: {e}")
            return {
                "signal": "neutral",
                "confidence": 0.5,
                "metrics": {}
            }
        
    def _analyze_mean_reversion(self):
        """均值回归策略分析"""
        df = self.prices_df
        
        try:
            # 布林带信号
            price = df['close'].iloc[-1]
            bb_upper = df['bb_upper'].iloc[-1]
            bb_lower = df['bb_lower'].iloc[-1]
            bb_middle = df['bb_middle'].iloc[-1]
            
            # 位置在布林带内的百分比 (0 = 下轨, 0.5 = 中轨, 1 = 上轨)
            bb_position = (price - bb_lower) / (bb_upper - bb_lower) if (bb_upper - bb_lower) > 0 else 0.5
            
            # 布林带宽度（波动率指标）
            bb_width = df['bb_width'].iloc[-1]
            bb_width_percentile = (bb_width - df['bb_width'].min()) / (df['bb_width'].max() - df['bb_width'].min() + 1e-10)
            
            # RSI过买过卖
            rsi = df['rsi14'].iloc[-1]
            
            # 偏离移动平均线距离
            deviation_from_sma20 = (price / bb_middle - 1) * 100
            
            # 整合信号
            oversold = (bb_position < 0.2 or rsi < 30 or deviation_from_sma20 < -10)
            overbought = (bb_position > 0.8 or rsi > 70 or deviation_from_sma20 > 10)
            
            # 确定信号
            if oversold:
                signal = "bullish"  # 超卖 = 买入信号 (均值回归)
                confidence = 0.6 + min(0.3, (1 - bb_position) * 0.3)
            elif overbought:
                signal = "bearish"  # 超买 = 卖出信号 (均值回归)
                confidence = 0.6 + min(0.3, bb_position * 0.3)
            else:
                signal = "neutral"
                confidence = 0.5
                
            # 准备指标数据
            metrics = {
                "rsi14": float(rsi),
                "bb_position": float(bb_position),
                "bb_width": float(bb_width),
                "bb_width_percentile": float(bb_width_percentile),
                "deviation_from_sma20_percent": float(deviation_from_sma20),
                "is_oversold": oversold,
                "is_overbought": overbought
            }
            
            return {
                "signal": signal,
                "confidence": confidence,
                "metrics": metrics
            }
            
        except Exception as e:
            print(f"均值回归分析出错: {e}")
            return {
                "signal": "neutral",
                "confidence": 0.5,
                "metrics": {}
            }
        
    def _analyze_momentum(self):
        """动量策略分析"""
        df = self.prices_df
        
        try:
            # MACD信号
            macd = df['macd'].iloc[-1]
            signal_line = df['signal_line'].iloc[-1]
            histogram = df['macd_histogram'].iloc[-1]
            
            macd_cross_bullish = macd > signal_line and df['macd'].iloc[-2] <= df['signal_line'].iloc[-2]
            macd_cross_bearish = macd < signal_line and df['macd'].iloc[-2] >= df['signal_line'].iloc[-2]
            
            # 价格动量（短、中、长）
            momentum_1d = df['price_change_1d'].iloc[-1] * 100
            momentum_3d = df['price_change_3d'].iloc[-1] * 100
            momentum_7d = df['price_change_7d'].iloc[-1] * 100
            
            # 价格加速度（动量的变化）
            acceleration = momentum_1d - df['price_change_1d'].iloc[-2] * 100
            
            # 计算动量强度
            momentum_strength = (momentum_1d * 0.5 + momentum_3d * 0.3 + momentum_7d * 0.2)
            
            # 整合信号
            if macd_cross_bullish or (macd > 0 and histogram > 0):
                macd_signal = "bullish"
            elif macd_cross_bearish or (macd < 0 and histogram < 0):
                macd_signal = "bearish"
            else:
                macd_signal = "neutral"
                
            if momentum_strength > 5:  # 强劲上涨动量
                price_signal = "bullish"
                price_confidence = min(0.6 + abs(momentum_strength) / 50, 0.9)
            elif momentum_strength < -5:  # 强劲下跌动量
                price_signal = "bearish"
                price_confidence = min(0.6 + abs(momentum_strength) / 50, 0.9)
            else:
                price_signal = "neutral"
                price_confidence = 0.5
                
            # 最终信号判断
            if macd_signal == price_signal:
                signal = macd_signal
                confidence = price_confidence
            elif macd_signal == "neutral":
                signal = price_signal
                confidence = price_confidence * 0.8  # 降低信心
            elif price_signal == "neutral":
                signal = macd_signal
                confidence = 0.6
            else:
                # 信号冲突
                signal = "neutral"
                confidence = 0.5
                
            # 准备指标数据
            metrics = {
                "macd": float(macd),
                "signal_line": float(signal_line),
                "histogram": float(histogram),
                "momentum_1d_percent": float(momentum_1d),
                "momentum_3d_percent": float(momentum_3d),
                "momentum_7d_percent": float(momentum_7d),
                "acceleration": float(acceleration),
                "momentum_strength": float(momentum_strength),
                "macd_cross_bullish": macd_cross_bullish,
                "macd_cross_bearish": macd_cross_bearish
            }
            
            return {
                "signal": signal,
                "confidence": confidence,
                "metrics": metrics
            }
            
        except Exception as e:
            print(f"动量分析出错: {e}")
            return {
                "signal": "neutral",
                "confidence": 0.5,
                "metrics": {}
            }
        
    def _analyze_volatility(self):
        """波动率策略分析"""
        df = self.prices_df
        
        try:
            # ATR波动率
            atr = df['atr14'].iloc[-1]
            avg_price = df['close'].iloc[-1]
            atr_percent = (atr / avg_price) * 100
            
            # 相对历史波动率
            recent_atr = df['atr14'].iloc[-5:].mean()
            historical_atr = df['atr14'].iloc[-20:-5].mean() if len(df) >= 20 else df['atr14'].mean()
            volatility_change = (recent_atr / historical_atr - 1) * 100 if historical_atr > 0 else 0
            
            # 布林带宽度变化
            recent_bb_width = df['bb_width'].iloc[-5:].mean()
            historical_bb_width = df['bb_width'].iloc[-20:-5].mean() if len(df) >= 20 else df['bb_width'].mean()
            bb_width_change = (recent_bb_width / historical_bb_width - 1) * 100 if historical_bb_width > 0 else 0
            
            # 波动率收缩后的突破（通常是大行情的开始）
            contracting_volatility = bb_width_change < -15
            
            # 波动率周期分析
            if volatility_change < -10 and contracting_volatility:
                volatility_regime = "contracting"  # 波动率收缩
                regime_signal = "watch_breakout"  # 关注突破
            elif volatility_change > 20:
                volatility_regime = "expanding"  # 波动率扩大
                regime_signal = "trending_market"  # 趋势市场
            else:
                volatility_regime = "normal"
                regime_signal = "neutral"
                
            # 最终信号判断
            if regime_signal == "watch_breakout":
                # 检查是否有突破
                if df['close'].iloc[-1] > df['bb_upper'].iloc[-2]:  # 向上突破
                    signal = "bullish"
                    confidence = 0.7
                elif df['close'].iloc[-1] < df['bb_lower'].iloc[-2]:  # 向下突破
                    signal = "bearish"
                    confidence = 0.7
                else:
                    signal = "neutral"  # 尚未突破
                    confidence = 0.5
            elif regime_signal == "trending_market":
                # 在波动率扩大期，跟随当前动量
                momentum_1d = df['price_change_1d'].iloc[-1]
                if momentum_1d > 0:
                    signal = "bullish"
                    confidence = min(0.6 + abs(momentum_1d) * 2, 0.85)
                elif momentum_1d < 0:
                    signal = "bearish"
                    confidence = min(0.6 + abs(momentum_1d) * 2, 0.85)
                else:
                    signal = "neutral"
                    confidence = 0.5
            else:
                signal = "neutral"
                confidence = 0.5
                
            # 准备指标数据
            metrics = {
                "atr14": float(atr),
                "atr_percent": float(atr_percent),
                "volatility_change_percent": float(volatility_change),
                "bb_width_change_percent": float(bb_width_change),
                "volatility_regime": volatility_regime,
                "is_contracting": contracting_volatility
            }
            
            return {
                "signal": signal,
                "confidence": confidence,
                "metrics": metrics
            }
            
        except Exception as e:
            print(f"波动率分析出错: {e}")
            return {
                "signal": "neutral",
                "confidence": 0.5,
                "metrics": {}
            }
        
    def _analyze_support_resistance(self):
        """支撑阻力位分析"""
        df = self.prices_df
        
        try:
            # 简化的支撑阻力位识别
            price_history = df['close'].iloc[-30:].values if len(df) >= 30 else df['close'].values
            current_price = df['close'].iloc[-1]
            
            # 找到局部高点和低点
            peaks = []
            troughs = []
            
            for i in range(2, len(price_history) - 2):
                if price_history[i] > price_history[i-1] and price_history[i] > price_history[i-2] and \
                   price_history[i] > price_history[i+1] and price_history[i] > price_history[i+2]:
                    peaks.append(price_history[i])
                    
                if price_history[i] < price_history[i-1] and price_history[i] < price_history[i-2] and \
                   price_history[i] < price_history[i+1] and price_history[i] < price_history[i+2]:
                    troughs.append(price_history[i])
                    
            # 对高点和低点进行聚类
            resistance_levels = []
            support_levels = []
            
            # 聚类阈值（价格的百分比）
            cluster_threshold = 0.02
            
            # 聚类阻力位
            for peak in sorted(peaks, reverse=True):
                if not resistance_levels or abs(peak / resistance_levels[-1] - 1) > cluster_threshold:
                    resistance_levels.append(peak)
                    
            # 聚类支撑位
            for trough in sorted(troughs):
                if not support_levels or abs(trough / support_levels[-1] - 1) > cluster_threshold:
                    support_levels.append(trough)
                    
            # 找到最近的支撑位和阻力位
            closest_resistance = None
            closest_support = None
            
            for level in resistance_levels:
                if level > current_price:
                    closest_resistance = level
                    break
                    
            for level in reversed(support_levels):
                if level < current_price:
                    closest_support = level
                    break
                    
            # 如果没有找到，使用近期高低点
            if closest_resistance is None and len(peaks) > 0:
                closest_resistance = max(peaks)
                
            if closest_support is None and len(troughs) > 0:
                closest_support = min(troughs)
                
            # 如果仍然没有找到，使用最近5天的高低点
            if closest_resistance is None:
                closest_resistance = df['high'].iloc[-5:].max()
                
            if closest_support is None:
                closest_support = df['low'].iloc[-5:].min()
                
            # 计算距离最近支撑/阻力位的百分比
            distance_to_resistance = ((closest_resistance / current_price) - 1) * 100 if closest_resistance else 0
            distance_to_support = ((current_price / closest_support) - 1) * 100 if closest_support else 0
            
            # 获取价格在支撑和阻力位之间的位置 (0 = 支撑位, 1 = 阻力位)
            if closest_support and closest_resistance:
                position = (current_price - closest_support) / (closest_resistance - closest_support)
            else:
                position = 0.5
                
            # 确定信号
            if distance_to_support < 3:  # 靠近支撑位
                signal = "bullish"
                confidence = 0.6 + min(0.3, (3 - distance_to_support) / 10)
            elif distance_to_resistance < 3:  # 靠近阻力位
                signal = "bearish"
                confidence = 0.6 + min(0.3, (3 - distance_to_resistance) / 10)
            else:
                # 根据价格在支撑阻力区间中的位置
                if position < 0.3:  # 更接近支撑位
                    signal = "bullish"
                    confidence = 0.6
                elif position > 0.7:  # 更接近阻力位
                    signal = "bearish"
                    confidence = 0.6
                else:  # 价格在中间位置
                    signal = "neutral"
                    confidence = 0.5
                    
            # 准备指标数据
            metrics = {
                "closest_support": float(closest_support) if closest_support else None,
                "closest_resistance": float(closest_resistance) if closest_resistance else None,
                "distance_to_support_percent": float(distance_to_support),
                "distance_to_resistance_percent": float(distance_to_resistance),
                "position_in_range": float(position)
            }
            
            return {
                "signal": signal,
                "confidence": confidence,
                "metrics": metrics
            }
            
        except Exception as e:
            print(f"支撑阻力位分析出错: {e}")
            return {
                "signal": "neutral",
                "confidence": 0.5,
                "metrics": {}
            }
        
    def _analyze_volume_price(self):
        """量价关系分析"""
        df = self.prices_df
        
        try:
            # 检查是否有成交量数据
            if 'volume' not in df.columns or df['volume'].sum() == 0:
                return {
                    "signal": "neutral",
                    "confidence": 0.5,
                    "metrics": {"volume_data_missing": True}
                }
                
            # 量价背离分析
            price_change = df['close'].iloc[-1] / df['close'].iloc[-2] - 1
            volume_change = df['volume'].iloc[-1] / df['volume'].iloc[-2] - 1 if df['volume'].iloc[-2] > 0 else 0
            
            # 成交量相对于平均成交量
            volume_vs_avg = df['volume'].iloc[-1] / df['volume_sma20'].iloc[-1] if df['volume_sma20'].iloc[-1] > 0 else 1
            
            # 连续成交量变化
            volume_trend = 0
            for i in range(-1, -min(6, len(df)), -1):
                if df['volume'].iloc[i] > df['volume'].iloc[i-1]:
                    volume_trend += 1
                elif df['volume'].iloc[i] < df['volume'].iloc[i-1]:
                    volume_trend -= 1
                    
            # 检测量价背离
            divergence = False
            divergence_type = "none"
            
            # 正向背离：价格下跌但成交量减少
            if price_change < -0.02 and volume_change < -0.1:
                divergence = True
                divergence_type = "positive"  # 看多信号
                
            # 负向背离：价格上涨但成交量减少
            elif price_change > 0.02 and volume_change < -0.1:
                divergence = True
                divergence_type = "negative"  # 看空信号
                
            # 成交量确认：价格上涨且成交量增加
            volume_confirmation = (price_change > 0 and volume_change > 0.1)
            
            # 成交量异常：成交量显著高于平均水平
            volume_spike = volume_vs_avg > 2
            
            # 确定信号
            if divergence_type == "positive":
                signal = "bullish"
                confidence = 0.7
            elif divergence_type == "negative":
                signal = "bearish"
                confidence = 0.7
            elif volume_confirmation:
                signal = "bullish" if price_change > 0 else "bearish"
                confidence = 0.65
            elif volume_spike:
                # 成交量异常通常预示着趋势变化
                signal = "bullish" if price_change > 0 else "bearish"
                confidence = 0.6
            elif volume_trend > 2:  # 成交量持续增加
                signal = "bullish" if price_change > 0 else "bearish"
                confidence = 0.6
            elif volume_trend < -2:  # 成交量持续减少
                signal = "neutral"
                confidence = 0.5
            else:
                signal = "neutral"
                confidence = 0.5
                
            # 准备指标数据
            metrics = {
                "price_change_percent": float(price_change * 100),
                "volume_change_percent": float(volume_change * 100),
                "volume_vs_avg": float(volume_vs_avg),
                "volume_trend": int(volume_trend),
                "divergence": divergence,
                "divergence_type": divergence_type,
                "volume_confirmation": volume_confirmation,
                "volume_spike": volume_spike
            }
            
            return {
                "signal": signal,
                "confidence": confidence,
                "metrics": metrics
            }
            
        except Exception as e:
            print(f"量价关系分析出错: {e}")
            return {
                "signal": "neutral",
                "confidence": 0.5,
                "metrics": {}
            }
        
    def _calculate_combined_signal(self, strategies):
        """计算综合技术分析信号"""
        # 设置各策略权重
        weights = {
            "trend_following": 0.20,
            "mean_reversion": 0.15,
            "momentum": 0.20,
            "volatility": 0.10,
            "support_resistance": 0.20,
            "volume_price": 0.15
        }
        
        # 信号数值映射
        signal_values = {
            "bullish": 1,
            "neutral": 0,
            "bearish": -1
        }
        
        # 计算加权得分
        weighted_score = 0
        total_confidence = 0
        
        for strategy, data in strategies.items():
            strategy_weight = weights.get(strategy, 0)
            signal_value = signal_values.get(data["signal"], 0)
            strategy_confidence = data["confidence"]
            
            weighted_score += signal_value * strategy_weight * strategy_confidence
            total_confidence += strategy_weight * strategy_confidence
            
        # 归一化得分
        if total_confidence > 0:
            normalized_score = weighted_score / total_confidence
        else:
            normalized_score = 0
            
        # 确定最终信号
        if normalized_score > 0.2:
            signal = "bullish"
            confidence = 0.5 + min(abs(normalized_score) / 2, 0.4)
        elif normalized_score < -0.2:
            signal = "bearish"
            confidence = 0.5 + min(abs(normalized_score) / 2, 0.4)
        else:
            signal = "neutral"
            confidence = 0.5
            
        return signal, confidence
        
    def _normalize_metrics(self, metrics):
        """将指标数据规范化为JSON可序列化的格式"""
        if not metrics:
            return {}
            
        normalized = {}
        for key, value in metrics.items():
            if isinstance(value, (int, float, str, bool)) or value is None:
                normalized[key] = value
            elif isinstance(value, (np.integer, np.floating)):
                normalized[key] = float(value)
            elif isinstance(value, np.ndarray):
                normalized[key] = value.tolist()
            elif isinstance(value, pd.Series):
                normalized[key] = value.tolist()
            else:
                normalized[key] = str(value)
                
        return normalized


def technical_analyst_agent(state: AgentState):
    """技术分析代理入口函数"""
    agent = TechnicalAnalysisAgent()
    return agent.analyze(state)