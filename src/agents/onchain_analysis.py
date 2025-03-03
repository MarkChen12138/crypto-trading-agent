from langchain_core.messages import HumanMessage
from src.agents.state import AgentState, show_agent_reasoning, show_workflow_status
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

class OnchainAnalysisAgent:
    """
    链上分析代理 - 负责分析区块链上的数据指标，评估加密货币的基础使用情况和网络健康状况
    
    分析的指标包括:
    1. 活跃地址 - 网络活动水平指标
    2. 交易量 - 网络使用情况
    3. 算力/质押率 - 网络安全性指标
    4. 持币分布 - 评估去中心化程度
    5. NVT比率 - 类似股票市盈率的估值指标
    6. MVRV比率 - 评估当前价格相对已实现价格的水平
    """
    
    def __init__(self):
        """初始化链上分析代理"""
        self.indicators = {}
        self.asset = None
    
    def analyze(self, state: AgentState):
        """执行链上分析"""
        show_workflow_status("链上分析代理")
        if "metadata" in state and "show_reasoning" in state.get("metadata", {}):
            show_reasoning = state["metadata"]["show_reasoning"]
        else:
            show_reasoning = False
        data = state["data"]

        # 获取基础数据
        self.asset = data["asset"]
        onchain_metrics = data.get("onchain_metrics", {})
        market_data = data.get("market_data", {})
        
        # 如果没有链上数据，返回中性信号
        if not onchain_metrics:
            message_content = {
                "signal": "neutral",
                "confidence": "0%",
                "reasoning": "缺少链上数据，无法进行分析"
            }
            
            message = HumanMessage(
                content=json.dumps(message_content),
                name="onchain_analysis_agent",
            )
            
            if show_reasoning:
                show_agent_reasoning(message_content, "链上分析代理")
                
            show_workflow_status("链上分析代理", "completed")
            return {
                "messages": [message],
                "data": data
            }
            
        # 分析各个指标
        signals = self._analyze_indicators(onchain_metrics, market_data)
        
        # 计算总体信号
        overall_signal, confidence, reasoning = self._calculate_overall_signal(signals)
        
        # 构建详细的分析结果
        detailed_analysis = {}
        for category, signal_data in signals.items():
            detailed_analysis[category] = {
                "signal": signal_data["signal"],
                "confidence": f"{signal_data['confidence']:.0%}",
                "metrics": signal_data["metrics"],
                "reasoning": signal_data["reasoning"]
            }
            
        message_content = {
            "signal": overall_signal,
            "confidence": f"{confidence:.0%}",
            "detailed_analysis": detailed_analysis,
            "reasoning": reasoning
        }
        
        message = HumanMessage(
            content=json.dumps(message_content),
            name="onchain_analysis_agent",
        )
        
        if show_reasoning:
            show_agent_reasoning(message_content, "链上分析代理")
            
        show_workflow_status("链上分析代理", "completed")
        return {
            "messages": [message],
            "data": {
                **data,
                "onchain_analysis": message_content
            }
        }
        
    def _analyze_indicators(self, onchain_metrics, market_data):
        """分析各类链上指标"""
        signals = {}
        
        # 1. 网络活动分析
        signals["network_activity"] = self._analyze_network_activity(onchain_metrics)
        
        # 2. 网络安全分析
        signals["network_security"] = self._analyze_network_security(onchain_metrics)
        
        # 3. 持币分布分析
        signals["holder_distribution"] = self._analyze_holder_distribution(onchain_metrics)
        
        # 4. 估值指标分析
        signals["valuation_metrics"] = self._analyze_valuation_metrics(onchain_metrics, market_data)
        
        return signals
        
    def _analyze_network_activity(self, metrics):
        """分析网络活动指标"""
        active_addresses = metrics.get("active_addresses_24h")
        transaction_count = metrics.get("transaction_count_24h")
        avg_transaction_value = metrics.get("average_transaction_value")
        
        # 创建评估指标
        activity_metrics = {
            "active_addresses_24h": active_addresses,
            "transaction_count_24h": transaction_count,
            "average_transaction_value": avg_transaction_value
        }
        
        # 评估网络活动水平
        # 以下阈值需要根据具体资产调整
        if self.asset == "BTC":
            # 比特币的阈值
            address_threshold_bullish = 900000
            address_threshold_bearish = 700000
            tx_threshold_bullish = 300000
            tx_threshold_bearish = 200000
            
        elif self.asset == "ETH":
            # 以太坊的阈值
            address_threshold_bullish = 600000
            address_threshold_bearish = 400000
            tx_threshold_bullish = 1200000
            tx_threshold_bearish = 800000
            
        else:
            # 其他资产的默认阈值
            address_threshold_bullish = 100000
            address_threshold_bearish = 50000
            tx_threshold_bullish = 50000
            tx_threshold_bearish = 20000
            
        # 计算信号
        signal_points = 0
        max_points = 0
        
        if active_addresses is not None:
            max_points += 1
            if active_addresses > address_threshold_bullish:
                signal_points += 1
            elif active_addresses < address_threshold_bearish:
                signal_points -= 1
                
        if transaction_count is not None:
            max_points += 1
            if transaction_count > tx_threshold_bullish:
                signal_points += 1
            elif transaction_count < tx_threshold_bearish:
                signal_points -= 1
        
        # 确定信号
        if max_points == 0:
            signal = "neutral"
            confidence = 0.5
            reasoning = "缺少网络活动数据，无法评估"
        else:
            normalized_score = signal_points / max_points
            if normalized_score > 0.3:
                signal = "bullish"
                confidence = 0.5 + min(normalized_score / 2, 0.4)  # 0.5-0.9范围
                reasoning = f"网络活动指标显示强劲的用户参与度，活跃地址数量为{active_addresses}，交易数量为{transaction_count}"
            elif normalized_score < -0.3:
                signal = "bearish"
                confidence = 0.5 + min(abs(normalized_score) / 2, 0.4)  # 0.5-0.9范围
                reasoning = f"网络活动指标显示用户参与度下降，活跃地址数量仅为{active_addresses}，交易数量为{transaction_count}"
            else:
                signal = "neutral"
                confidence = 0.5
                reasoning = f"网络活动指标处于正常水平，活跃地址数量为{active_addresses}，交易数量为{transaction_count}"
                
        return {
            "signal": signal,
            "confidence": confidence,
            "metrics": activity_metrics,
            "reasoning": reasoning
        }
        
    def _analyze_network_security(self, metrics):
        """分析网络安全性指标"""
        hashrate = metrics.get("hashrate")  # 比特币
        staking_rate = metrics.get("staking_rate")  # 以太坊
        
        # 创建评估指标
        security_metrics = {
            "hashrate": hashrate,
            "staking_rate": staking_rate
        }
        
        # 评估网络安全性
        if self.asset == "BTC" and hashrate is not None:
            # 比特币算力评估
            # 例如，300 EH/s是强劲的，200 EH/s以下可能表示矿工撤离
            if hashrate > 300000000000000000000:  # 300 EH/s
                signal = "bullish"
                confidence = 0.8
                reasoning = f"比特币网络算力处于历史高位，达到{hashrate/1000000000000000000:.2f} EH/s，表明矿工对网络有强烈信心"
            elif hashrate < 200000000000000000000:  # 200 EH/s
                signal = "bearish"
                confidence = 0.7
                reasoning = f"比特币网络算力下降至{hashrate/1000000000000000000:.2f} EH/s，可能表明矿工正在减少投入"
            else:
                signal = "neutral"
                confidence = 0.6
                reasoning = f"比特币网络算力保持稳定，为{hashrate/1000000000000000000:.2f} EH/s"
                
        elif self.asset == "ETH" and staking_rate is not None:
            # 以太坊质押率评估
            if staking_rate > 60:  # 60%以上是强劲的
                signal = "bullish"
                confidence = 0.8
                reasoning = f"以太坊质押率高达{staking_rate:.1f}%，表明持有者对网络有长期信心"
            elif staking_rate < 40:  # 40%以下可能表示信心不足
                signal = "bearish"
                confidence = 0.7
                reasoning = f"以太坊质押率较低，仅为{staking_rate:.1f}%，可能表明持有者信心不足"
            else:
                signal = "neutral"
                confidence = 0.6
                reasoning = f"以太坊质押率处于正常水平，为{staking_rate:.1f}%"
                
        else:
            signal = "neutral"
            confidence = 0.5
            reasoning = "缺少安全性数据，无法评估"
            
        return {
            "signal": signal,
            "confidence": confidence,
            "metrics": security_metrics,
            "reasoning": reasoning
        }
        
    def _analyze_holder_distribution(self, metrics):
        """分析持币分布情况"""
        supply_last_active = metrics.get("supply_last_active_1y_percent")  # 过去1年活跃的供应比例
        
        # 创建评估指标
        distribution_metrics = {
            "supply_last_active_1y_percent": supply_last_active
        }
        
        # 评估持币分布
        if supply_last_active is not None:
            # 分析持币者行为
            # 较低的活跃比例表示长期持有意愿强，通常是看涨信号
            if supply_last_active < 30:  # 不到30%的供应在流通
                signal = "bullish"
                confidence = 0.75
                reasoning = f"仅{supply_last_active:.1f}%的供应在过去一年内活跃，表明强劲的持有意愿和较少的卖压"
            elif supply_last_active > 50:  # 超过50%的供应在流通
                signal = "bearish"
                confidence = 0.7
                reasoning = f"高达{supply_last_active:.1f}%的供应在过去一年内活跃，可能表明较高的流通性和潜在卖压"
            else:
                signal = "neutral"
                confidence = 0.6
                reasoning = f"{supply_last_active:.1f}%的供应在过去一年内活跃，处于正常水平"
        else:
            signal = "neutral"
            confidence = 0.5
            reasoning = "缺少持币分布数据，无法评估"
            
        return {
            "signal": signal,
            "confidence": confidence,
            "metrics": distribution_metrics,
            "reasoning": reasoning
        }
        
    def _analyze_valuation_metrics(self, metrics, market_data):
        """分析估值指标"""
        nvt_ratio = metrics.get("nvt_ratio")  # 网络价值对交易量比率
        mvrv_ratio = metrics.get("mvrv_ratio")  # 市值对已实现价值比率
        realized_price = metrics.get("realized_price")  # 已实现价格
        puell_multiple = metrics.get("puell_multiple")  # Puell Multiple (仅BTC)
        
        current_price = market_data.get("price", 0)
        
        # 创建评估指标
        valuation_metrics = {
            "nvt_ratio": nvt_ratio,
            "mvrv_ratio": mvrv_ratio,
            "realized_price": realized_price,
            "price_to_realized_ratio": current_price / realized_price if realized_price else None,
            "puell_multiple": puell_multiple
        }
        
        # 评估估值指标
        signal_points = 0
        max_points = 0
        reasoning_parts = []
        
        # NVT比率分析
        if nvt_ratio is not None:
            max_points += 1
            if nvt_ratio < 25:  # 低NVT表示相对交易活动而言价格低
                signal_points += 1
                reasoning_parts.append(f"NVT比率较低({nvt_ratio:.1f})，表明相对于网络活动水平，当前价格仍具吸引力")
            elif nvt_ratio > 45:  # 高NVT表示相对交易活动而言价格高
                signal_points -= 1
                reasoning_parts.append(f"NVT比率较高({nvt_ratio:.1f})，表明相对于网络活动水平，当前价格可能偏高")
            else:
                reasoning_parts.append(f"NVT比率适中({nvt_ratio:.1f})")
                
        # MVRV比率分析
        if mvrv_ratio is not None:
            max_points += 1
            if mvrv_ratio < 1.2:  # MVRV接近1表示接近成本价，通常是买入机会
                signal_points += 1
                reasoning_parts.append(f"MVRV比率较低({mvrv_ratio:.2f})，表明价格接近历史成本，可能是良好的买入机会")
            elif mvrv_ratio > 2.5:  # 高MVRV表示价格远高于成本，可能存在回调风险
                signal_points -= 1
                reasoning_parts.append(f"MVRV比率较高({mvrv_ratio:.2f})，表明价格远高于历史成本，可能面临回调风险")
            else:
                reasoning_parts.append(f"MVRV比率适中({mvrv_ratio:.2f})")
                
        # Puell Multiple分析(仅BTC)
        if self.asset == "BTC" and puell_multiple is not None:
            max_points += 1
            if puell_multiple < 0.8:  # 低Puell Multiple表示矿工收入低，通常是底部区域
                signal_points += 1
                reasoning_parts.append(f"Puell Multiple较低({puell_multiple:.2f})，表明矿工收入处于历史低位，通常是周期性底部区域")
            elif puell_multiple > 1.5:  # 高Puell Multiple表示矿工收入高，可能是顶部区域
                signal_points -= 1
                reasoning_parts.append(f"Puell Multiple较高({puell_multiple:.2f})，表明矿工收入处于历史高位，可能接近周期性顶部")
            else:
                reasoning_parts.append(f"Puell Multiple适中({puell_multiple:.2f})")
                
        # 确定信号
        if max_points == 0:
            signal = "neutral"
            confidence = 0.5
            reasoning = "缺少估值指标数据，无法评估"
        else:
            normalized_score = signal_points / max_points
            reasoning = "，".join(reasoning_parts)
            
            if normalized_score > 0.3:
                signal = "bullish"
                confidence = 0.5 + min(normalized_score / 2, 0.4)  # 0.5-0.9范围
            elif normalized_score < -0.3:
                signal = "bearish"
                confidence = 0.5 + min(abs(normalized_score) / 2, 0.4)  # 0.5-0.9范围
            else:
                signal = "neutral"
                confidence = 0.5
                
        return {
            "signal": signal,
            "confidence": confidence,
            "metrics": valuation_metrics,
            "reasoning": reasoning
        }
        
    def _calculate_overall_signal(self, signals):
        """计算总体信号"""
        # 设置各个类别的权重
        weights = {
            "network_activity": 0.25,
            "network_security": 0.20,
            "holder_distribution": 0.20,
            "valuation_metrics": 0.35
        }
        
        # 将信号转换为数值
        signal_values = {
            "bullish": 1,
            "neutral": 0,
            "bearish": -1
        }
        
        # 计算加权得分
        weighted_score = 0
        total_confidence = 0
        
        for category, signal_data in signals.items():
            category_weight = weights.get(category, 0)
            signal_value = signal_values.get(signal_data["signal"], 0)
            signal_confidence = signal_data["confidence"]
            
            weighted_score += signal_value * category_weight * signal_confidence
            total_confidence += category_weight * signal_confidence
            
        # 归一化得分
        if total_confidence > 0:
            normalized_score = weighted_score / total_confidence
        else:
            normalized_score = 0
            
        # 确定总体信号
        if normalized_score > 0.2:
            signal = "bullish"
            confidence = 0.5 + min(abs(normalized_score) / 2, 0.4)
        elif normalized_score < -0.2:
            signal = "bearish"
            confidence = 0.5 + min(abs(normalized_score) / 2, 0.4)
        else:
            signal = "neutral"
            confidence = 0.5
            
        # 生成综合推理
        bullish_categories = [category for category, data in signals.items() if data["signal"] == "bullish"]
        bearish_categories = [category for category, data in signals.items() if data["signal"] == "bearish"]
        
        if signal == "bullish":
            reasoning = f"基于链上分析，主要看多信号来自{', '.join(bullish_categories)}。"
            if bearish_categories:
                reasoning += f" 同时也有来自{', '.join(bearish_categories)}的看空信号，但整体仍偏向积极。"
        elif signal == "bearish":
            reasoning = f"基于链上分析，主要看空信号来自{', '.join(bearish_categories)}。"
            if bullish_categories:
                reasoning += f" 同时也有来自{', '.join(bullish_categories)}的看多信号，但整体仍偏向谨慎。"
        else:
            reasoning = "基于链上分析，积极和消极信号基本平衡，建议保持中性立场。"
            
        return signal, confidence, reasoning


def onchain_analysis_agent(state: AgentState):
    """链上分析代理入口函数"""
    agent = OnchainAnalysisAgent()
    return agent.analyze(state)