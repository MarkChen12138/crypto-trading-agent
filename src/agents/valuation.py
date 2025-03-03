from langchain_core.messages import HumanMessage
from src.agents.state import AgentState, show_agent_reasoning, show_workflow_status
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

class ValuationAnalysisAgent:
    """
    估值分析代理 - 为加密货币资产提供估值分析
    
    功能:
    1. 计算各种加密货币特定的估值指标
    2. 分析链上数据与价格的关系
    3. 执行相对估值和绝对估值分析
    4. 提供估值区间和公允价值判断
    """
    
    def __init__(self):
        """初始化估值分析代理"""
        self.asset = None
        self.market_data = None
        self.onchain_metrics = None
        
    def analyze(self, state: AgentState):
        """执行估值分析"""
        show_workflow_status("估值分析代理")
        
        if "metadata" in state and "show_reasoning" in state.get("metadata", {}):
            show_reasoning = state["metadata"]["show_reasoning"]
        else:
            show_reasoning = False  # 默认不显示推理过程
        
        data = state.get("data", {})
        
        # 获取基础数据
        self.asset = data.get("asset", "BTC")
        self.market_data = data.get("market_data", {})
        self.onchain_metrics = data.get("onchain_metrics", {})
        prices_data = data.get("prices", [])
        
        # 如果数据不足，返回中性信号
        if not self.market_data or not prices_data:
            message_content = {
                "signal": "neutral",
                "confidence": "50%",
                "reasoning": {
                    "missing_data": "缺少必要的市场数据或价格历史，无法进行可靠的估值分析"
                }
            }
            
            message = HumanMessage(
                content=json.dumps(message_content),
                name="valuation_agent",
            )
            
            if show_reasoning:
                show_agent_reasoning(message_content, "估值分析代理")
                
            show_workflow_status("估值分析代理", "completed")
            return {
                "messages": [message],
                "data": data
            }
            
        # 获取当前价格
        current_price = self.market_data.get("price", 0)
        if current_price == 0 and prices_data:
            # 尝试从价格历史获取
            if isinstance(prices_data, list) and len(prices_data) > 0:
                if isinstance(prices_data[-1], dict) and "close" in prices_data[-1]:
                    current_price = prices_data[-1]["close"]
                else:
                    # 将价格数据转换为DataFrame便于处理
                    prices_df = pd.DataFrame(prices_data)
                    if "close" in prices_df.columns:
                        current_price = prices_df["close"].iloc[-1]
                        
        # 如果仍然无法获取价格，返回中性信号
        if current_price == 0:
            message_content = {
                "signal": "neutral",
                "confidence": "50%",
                "reasoning": {
                    "missing_price": "无法获取当前资产价格，无法进行估值分析"
                }
            }
            
            message = HumanMessage(
                content=json.dumps(message_content),
                name="valuation_agent",
            )
            
            if show_reasoning:
                show_agent_reasoning(message_content, "估值分析代理")
                
            show_workflow_status("估值分析代理", "completed")
            return {
                "messages": [message],
                "data": data
            }
            
        # 选择合适的估值方法
        valuation_results = {}
        
        if self.asset == "BTC":
            # 比特币特有的估值方法
            valuation_results["stock_to_flow"] = self._calculate_stock_to_flow()
            valuation_results["thermocap_multiple"] = self._calculate_thermocap_multiple()
            valuation_results["nvt_analysis"] = self._calculate_nvt_ratio()
            valuation_results["mvrv_analysis"] = self._calculate_mvrv_ratio()
            valuation_results["puell_multiple"] = self._calculate_puell_multiple()
            
        elif self.asset == "ETH":
            # 以太坊特有的估值方法
            valuation_results["fee_model"] = self._calculate_fee_based_valuation()
            valuation_results["staking_yield"] = self._calculate_staking_yield_valuation()
            valuation_results["nvt_analysis"] = self._calculate_nvt_ratio()
            valuation_results["mvrv_analysis"] = self._calculate_mvrv_ratio()
            
        else:
            # 其他代币的通用估值方法
            valuation_results["network_value_analysis"] = self._calculate_network_value()
            valuation_results["relative_valuation"] = self._calculate_relative_valuation()
            
        # 添加通用估值方法
        valuation_results["fair_value_range"] = self._calculate_fair_value_range(current_price)
        
        # 综合估值结果
        overall_valuation, confidence, reasoning = self._synthesize_valuation_results(
            valuation_results, current_price
        )
        
        # 准备消息内容
        message_content = {
            "signal": overall_valuation,
            "confidence": f"{confidence:.0%}",
            "reasoning": reasoning
        }
        
        # 创建消息
        message = HumanMessage(
            content=json.dumps(message_content),
            name="valuation_agent",
        )
        
        if show_reasoning:
            show_agent_reasoning(message_content, "估值分析代理")
            
        show_workflow_status("估值分析代理", "completed")
        return {
            "messages": [message],
            "data": {
                **data,
                "valuation_analysis": message_content
            }
        }
        
    def _calculate_stock_to_flow(self):
        """
        计算比特币的库存流量(S2F)模型估值
        
        S2F = 当前供应量 / 年供应增长量
        """
        # 获取必要数据
        realized_price = self.onchain_metrics.get("realized_price", 0)
        # 硬编码的当前BTC库存流量比率（实际中应从API或数据源获取）
        estimated_s2f_ratio = 56  # 2023年后的近似值
        
        if not realized_price:
            return {
                "signal": "neutral",
                "confidence": 0.5,
                "estimated_price": 0,
                "details": "缺少实现价格数据，无法计算S2F模型估值"
            }
            
        # S2F模型预测价格 = exp(a) * S2F^b
        # 其中a=14.6，b=3.3是历史拟合的参数
        try:
            # 简化的S2F计算
            a = 14.6
            b = 3.3
            estimated_price = np.exp(a) * (estimated_s2f_ratio ** b)
            
            # 获取当前价格
            current_price = self.market_data.get("price", 0)
            
            # 计算偏差
            if current_price > 0:
                deviation = (current_price / estimated_price) - 1
                
                # 确定信号
                if deviation < -0.3:  # 当前价格比模型预测低30%以上
                    signal = "bullish"
                    confidence = 0.7
                    details = f"当前价格显著低于S2F模型预测值({deviation:.1%})，表明可能被低估"
                elif deviation > 0.3:  # 当前价格比模型预测高30%以上
                    signal = "bearish"
                    confidence = 0.7
                    details = f"当前价格显著高于S2F模型预测值({deviation:.1%})，表明可能被高估"
                else:
                    signal = "neutral"
                    confidence = 0.5
                    details = f"当前价格接近S2F模型预测值，偏差为{deviation:.1%}"
                    
                return {
                    "signal": signal,
                    "confidence": confidence,
                    "estimated_price": estimated_price,
                    "deviation": deviation,
                    "details": details
                }
            else:
                return {
                    "signal": "neutral",
                    "confidence": 0.5,
                    "estimated_price": estimated_price,
                    "details": "无法获取当前价格进行比较"
                }
                
        except Exception as e:
            print(f"计算S2F模型时出错: {e}")
            return {
                "signal": "neutral",
                "confidence": 0.5,
                "estimated_price": 0,
                "details": f"计算S2F模型时出错: {e}"
            }
            
    def _calculate_thermocap_multiple(self):
        """
        计算比特币的Thermocap倍数
        
        Thermocap = 历史挖矿成本的总和
        Thermocap Multiple = 市值 / Thermocap
        """
        # 由于无法直接获取历史挖矿成本，使用估计值
        # 实际应用中应从专业数据提供商获取
        
        # 假设的近似值（实际中应从API获取）
        estimated_thermocap = 20e9  # 200亿美元
        market_cap = self.market_data.get("price", 0) * 19e6  # 假设当前流通量1900万
        
        if market_cap == 0:
            return {
                "signal": "neutral",
                "confidence": 0.5,
                "thermocap_multiple": 0,
                "details": "无法获取市值数据，无法计算Thermocap倍数"
            }
            
        thermocap_multiple = market_cap / estimated_thermocap
        
        # Thermocap倍数的历史区间通常在2-25之间
        # 低于3通常被视为底部区域，高于20通常被视为顶部区域
        if thermocap_multiple < 3:
            signal = "bullish"
            confidence = 0.8
            details = f"Thermocap倍数({thermocap_multiple:.2f})处于历史底部区域，表明价格接近挖矿成本"
        elif thermocap_multiple > 20:
            signal = "bearish"
            confidence = 0.8
            details = f"Thermocap倍数({thermocap_multiple:.2f})处于历史顶部区域，表明价格远高于挖矿成本"
        elif thermocap_multiple > 10:
            signal = "bearish"
            confidence = 0.6
            details = f"Thermocap倍数({thermocap_multiple:.2f})处于中高位区域，接近历史周期高点"
        elif thermocap_multiple < 5:
            signal = "bullish"
            confidence = 0.6
            details = f"Thermocap倍数({thermocap_multiple:.2f})处于中低位区域，接近历史周期低点"
        else:
            signal = "neutral"
            confidence = 0.5
            details = f"Thermocap倍数({thermocap_multiple:.2f})处于中间区域，既不高估也不低估"
            
        return {
            "signal": signal,
            "confidence": confidence,
            "thermocap_multiple": thermocap_multiple,
            "details": details
        }
        
    def _calculate_nvt_ratio(self):
        """
        计算网络价值对交易量比率(NVT)
        
        NVT = 市值 / 日交易量
        """
        # 获取必要数据
        nvt_ratio = self.onchain_metrics.get("nvt_ratio", 0)
        
        if nvt_ratio == 0:
            # 尝试自行计算
            market_cap = self.market_data.get("price", 0) * 19e6  # 假设BTC流通量1900万
            transaction_volume_24h = self.onchain_metrics.get("transaction_count_24h", 0) * self.onchain_metrics.get("average_transaction_value", 0)
            
            if market_cap > 0 and transaction_volume_24h > 0:
                nvt_ratio = market_cap / transaction_volume_24h
            else:
                return {
                    "signal": "neutral",
                    "confidence": 0.5,
                    "nvt_ratio": 0,
                    "details": "缺少市值或交易量数据，无法计算NVT比率"
                }
                
        # NVT比率的解释
        # 低NVT表示相对于网络活动而言，价格偏低
        # 高NVT表示相对于网络活动而言，价格偏高
        if nvt_ratio < 15:  # 极低的NVT
            signal = "bullish"
            confidence = 0.8
            details = f"NVT比率极低({nvt_ratio:.1f})，表明相对于链上活动量，当前价格被显著低估"
        elif nvt_ratio < 25:  # 较低的NVT
            signal = "bullish"
            confidence = 0.6
            details = f"NVT比率较低({nvt_ratio:.1f})，表明相对于链上活动量，当前价格可能被低估"
        elif nvt_ratio > 90:  # 极高的NVT
            signal = "bearish"
            confidence = 0.8
            details = f"NVT比率极高({nvt_ratio:.1f})，表明相对于链上活动量，当前价格被显著高估"
        elif nvt_ratio > 60:  # 较高的NVT
            signal = "bearish"
            confidence = 0.6
            details = f"NVT比率较高({nvt_ratio:.1f})，表明相对于链上活动量，当前价格可能被高估"
        else:  # 中间区域
            signal = "neutral"
            confidence = 0.5
            details = f"NVT比率处于中间区域({nvt_ratio:.1f})，价格与链上活动基本匹配"
            
        return {
            "signal": signal,
            "confidence": confidence,
            "nvt_ratio": nvt_ratio,
            "details": details
        }
        
    def _calculate_mvrv_ratio(self):
        """
        计算市值与实现值比率(MVRV)
        
        MVRV = 市值 / 实现值
        实现值 = 每个币的最后移动价格的总和
        """
        # 获取必要数据
        mvrv_ratio = self.onchain_metrics.get("mvrv_ratio", 0)
        
        if mvrv_ratio == 0:
            # 尝试自行计算
            current_price = self.market_data.get("price", 0)
            realized_price = self.onchain_metrics.get("realized_price", 0)
            
            if current_price > 0 and realized_price > 0:
                mvrv_ratio = current_price / realized_price
            else:
                return {
                    "signal": "neutral",
                    "confidence": 0.5,
                    "mvrv_ratio": 0,
                    "details": "缺少价格或实现价格数据，无法计算MVRV比率"
                }
                
        # MVRV比率的解释
        # MVRV < 1: 价格低于实现价格，通常是买入机会
        # MVRV > 3.5: 历史上通常表示市场顶部区域
        if mvrv_ratio < 0.8:  # 极低的MVRV
            signal = "bullish"
            confidence = 0.9
            details = f"MVRV比率极低({mvrv_ratio:.2f})，表明价格远低于持币者的平均成本，历史上是极佳的买入机会"
        elif mvrv_ratio < 1:  # MVRV低于1
            signal = "bullish"
            confidence = 0.7
            details = f"MVRV比率低于1({mvrv_ratio:.2f})，表明价格低于持币者的平均成本，通常是良好的买入时机"
        elif mvrv_ratio > 3.5:  # 极高的MVRV
            signal = "bearish"
            confidence = 0.9
            details = f"MVRV比率极高({mvrv_ratio:.2f})，表明市场估值过高，历史上通常是市场顶部区域"
        elif mvrv_ratio > 2.5:  # 较高的MVRV
            signal = "bearish"
            confidence = 0.7
            details = f"MVRV比率较高({mvrv_ratio:.2f})，表明市场估值偏高，接近历史周期高点"
        else:  # 中间区域
            signal = "neutral"
            confidence = 0.5
            details = f"MVRV比率处于中间区域({mvrv_ratio:.2f})，既不显著高估也不显著低估"
            
        return {
            "signal": signal,
            "confidence": confidence,
            "mvrv_ratio": mvrv_ratio,
            "details": details
        }
        
    def _calculate_puell_multiple(self):
        """
        计算Puell Multiple (矿工收入指标)
        
        Puell Multiple = 当前日挖矿收入 / 365日平均挖矿收入
        """
        # 获取必要数据
        puell_multiple = self.onchain_metrics.get("puell_multiple", 0)
        
        if puell_multiple == 0:
            return {
                "signal": "neutral",
                "confidence": 0.5,
                "puell_multiple": 0,
                "details": "缺少Puell Multiple数据，无法分析矿工收入指标"
            }
            
        # Puell Multiple的解释
        # 低Puell Multiple表示矿工收入低，通常是市场底部区域
        # 高Puell Multiple表示矿工收入高，通常是市场顶部区域
        if puell_multiple < 0.5:  # 极低的Puell Multiple
            signal = "bullish"
            confidence = 0.9
            details = f"Puell Multiple极低({puell_multiple:.2f})，表明矿工收入极低，历史上是极佳的买入机会"
        elif puell_multiple < 0.8:  # 较低的Puell Multiple
            signal = "bullish"
            confidence = 0.7
            details = f"Puell Multiple较低({puell_multiple:.2f})，表明矿工收入低于长期平均，通常是良好的买入区域"
        elif puell_multiple > 4:  # 极高的Puell Multiple
            signal = "bearish"
            confidence = 0.9
            details = f"Puell Multiple极高({puell_multiple:.2f})，表明矿工收入极高，历史上通常是市场顶部区域"
        elif puell_multiple > 2:  # 较高的Puell Multiple
            signal = "bearish"
            confidence = 0.7
            details = f"Puell Multiple较高({puell_multiple:.2f})，表明矿工收入高于长期平均，通常是减仓区域"
        else:  # 中间区域
            signal = "neutral"
            confidence = 0.5
            details = f"Puell Multiple处于中间区域({puell_multiple:.2f})，矿工收入接近长期平均"
            
        return {
            "signal": signal,
            "confidence": confidence,
            "puell_multiple": puell_multiple,
            "details": details
        }
        
    def _calculate_fee_based_valuation(self):
        """
        基于网络费用的估值模型 (主要用于ETH等智能合约平台)
        
        估值 = 年度网络费用 × P/E倍数
        """
        # 由于无法直接获取网络费用数据，使用估计值
        # 实际应用中应从专业数据提供商获取
        
        # 假设的近似值（实际中应从API获取）
        if self.asset != "ETH":
            return {
                "signal": "neutral",
                "confidence": 0.5,
                "fee_based_price": 0,
                "details": "费用估值模型仅适用于以太坊等智能合约平台"
            }
            
        daily_fee = self.onchain_metrics.get("average_transaction_value", 0) * 0.001  # 假设平均费率为0.1%
        transaction_count = self.onchain_metrics.get("transaction_count_24h", 5000000)  # 默认500万笔交易/天
        
        if daily_fee == 0 or transaction_count == 0:
            # 使用估计值
            daily_fee = 5000000  # 500万美元/天的网络费用
            
        # 计算年度费用
        annual_fee = daily_fee * 365
        
        # 应用P/E倍数 (以太坊的合理P/E范围约为20-50)
        low_pe = 20
        mid_pe = 35
        high_pe = 50
        
        # 计算基于不同P/E的估值
        circulating_supply = 120e6  # ETH流通量约1.2亿
        
        low_value = (annual_fee * low_pe) / circulating_supply
        mid_value = (annual_fee * mid_pe) / circulating_supply
        high_value = (annual_fee * high_pe) / circulating_supply
        
        # 获取当前价格
        current_price = self.market_data.get("price", 0)
        
        if current_price == 0:
            return {
                "signal": "neutral",
                "confidence": 0.5,
                "fee_based_price": mid_value,
                "details": "无法获取当前价格进行比较"
            }
            
        # 计算当前价格相对于估值的位置
        if current_price < low_value:
            signal = "bullish"
            confidence = 0.8
            details = f"当前价格({current_price:.0f})低于基于保守P/E({low_pe}x)的费用估值({low_value:.0f})，表明可能被低估"
        elif current_price > high_value:
            signal = "bearish"
            confidence = 0.8
            details = f"当前价格({current_price:.0f})高于基于积极P/E({high_pe}x)的费用估值({high_value:.0f})，表明可能被高估"
        elif current_price < mid_value:
            signal = "bullish"
            confidence = 0.6
            details = f"当前价格({current_price:.0f})低于基于中等P/E({mid_pe}x)的费用估值({mid_value:.0f})，表明略微低估"
        elif current_price > mid_value:
            signal = "bearish"
            confidence = 0.6
            details = f"当前价格({current_price:.0f})高于基于中等P/E({mid_pe}x)的费用估值({mid_value:.0f})，表明略微高估"
        else:
            signal = "neutral"
            confidence = 0.5
            details = f"当前价格({current_price:.0f})接近基于中等P/E的费用估值({mid_value:.0f})"
            
        return {
            "signal": signal,
            "confidence": confidence,
            "fee_based_price": mid_value,
            "low_value": low_value,
            "high_value": high_value,
            "details": details
        }
        
    def _calculate_staking_yield_valuation(self):
        """
        基于质押收益率的估值模型 (主要用于PoS代币如ETH)
        
        使用股息折现模型的变体估值
        """
        if self.asset != "ETH":
            return {
                "signal": "neutral",
                "confidence": 0.5,
                "yield_based_price": 0,
                "details": "质押收益率估值模型仅适用于PoS代币"
            }
            
        # 获取质押率和收益率
        staking_rate = self.onchain_metrics.get("staking_rate", 0)
        
        if staking_rate == 0:
            # 使用估计值
            staking_rate = 65  # 当前ETH质押率约65%
            
        # 估计年化质押收益率 (基于质押率的近似公式)
        # 收益率 = 基础收益 * (1 - 质押率/100)^0.5
        base_yield = 0.05  # 5%基础收益
        estimated_yield = base_yield * (1 - staking_rate/100)**0.5
        
        # 应用收益率贴现模型
        # 使用风险溢价范围
        low_risk_premium = 0.02  # 2%
        mid_risk_premium = 0.04  # 4%
        high_risk_premium = 0.06  # 6%
        
        # 计算基于不同风险溢价的估值
        # 价格 = 年收益 / (风险溢价 + 质押率)
        annual_eth_emission = 0.012  # 以太坊年通胀率约1.2%
        low_value = estimated_yield / (annual_eth_emission + low_risk_premium) * 1000  # 缩放因子
        mid_value = estimated_yield / (annual_eth_emission + mid_risk_premium) * 1000
        high_value = estimated_yield / (annual_eth_emission + high_risk_premium) * 1000
        
        # 获取当前价格
        current_price = self.market_data.get("price", 0)
        
        if current_price == 0:
            return {
                "signal": "neutral",
                "confidence": 0.5,
                "yield_based_price": mid_value,
                "details": "无法获取当前价格进行比较"
            }
            
        # 计算当前价格相对于估值的位置
        if current_price < low_value:
            signal = "bullish"
            confidence = 0.7
            details = f"当前价格({current_price:.0f})低于基于低风险溢价的收益率估值({low_value:.0f})，表明可能被低估"
        elif current_price > high_value:
            signal = "bearish"
            confidence = 0.7
            details = f"当前价格({current_price:.0f})高于基于高风险溢价的收益率估值({high_value:.0f})，表明可能被高估"
        elif current_price < mid_value:
            signal = "bullish"
            confidence = 0.6
            details = f"当前价格({current_price:.0f})低于基于中等风险溢价的收益率估值({mid_value:.0f})，表明略微低估"
        elif current_price > mid_value:
            signal = "bearish"
            confidence = 0.6
            details = f"当前价格({current_price:.0f})高于基于中等风险溢价的收益率估值({mid_value:.0f})，表明略微高估"
        else:
            signal = "neutral"
            confidence = 0.5
            details = f"当前价格({current_price:.0f})接近基于中等风险溢价的收益率估值({mid_value:.0f})"
            
        return {
            "signal": signal,
            "confidence": confidence,
            "yield_based_price": mid_value,
            "estimated_yield": estimated_yield,
            "low_value": low_value,
            "high_value": high_value,
            "details": details
        }
        
    def _calculate_network_value(self):
        """
        基于网络价值的估值模型 (适用于所有加密资产)
        使用Metcalfe定律的变体
        """
        # 获取必要数据
        active_addresses = self.onchain_metrics.get("active_addresses_24h", 0)
        
        if active_addresses == 0:
            return {
                "signal": "neutral",
                "confidence": 0.5,
                "network_value": 0,
                "details": "缺少活跃地址数据，无法计算网络价值"
            }
            
        # 使用Metcalfe定律的变体计算网络价值
        # 网络价值与活跃用户数的平方成正比
        # 实际公式中会有一个系数，这里根据资产类型选择不同系数
        if self.asset == "BTC":
            coefficient = 1000  # 比特币网络价值系数
        elif self.asset == "ETH":
            coefficient = 500   # 以太坊网络价值系数
        else:
            coefficient = 200   # 其他代币默认系数
            
        # 计算网络价值 (简化的Metcalfe模型)
        network_value = coefficient * (active_addresses ** 1.8)  # 使用1.8次方而不是2次方
        
        # 计算每个代币的价值
        if self.asset == "BTC":
            circulating_supply = 19e6  # 约1900万BTC
        elif self.asset == "ETH":
            circulating_supply = 120e6  # 约1.2亿ETH
        else:
            circulating_supply = 1e9  # 默认10亿
            
        estimated_price = network_value / circulating_supply
        
        # 获取当前价格
        current_price = self.market_data.get("price", 0)
        
        if current_price == 0:
            return {
                "signal": "neutral",
                "confidence": 0.5,
                "network_value": network_value,
                "estimated_price": estimated_price,
                "details": "无法获取当前价格进行比较"
            }
            
        # 计算偏差
        deviation = (current_price / estimated_price) - 1
        
        # 确定信号
        if deviation < -0.3:  # 当前价格比估值低30%以上
            signal = "bullish"
            confidence = 0.7
            details = f"当前价格显著低于网络价值估值({deviation:.1%})，表明可能被低估"
        elif deviation > 0.3:  # 当前价格比估值高30%以上
            signal = "bearish"
            confidence = 0.7
            details = f"当前价格显著高于网络价值估值({deviation:.1%})，表明可能被高估"
        elif deviation < -0.1:  # 当前价格略低于估值
            signal = "bullish"
            confidence = 0.6
            details = f"当前价格略低于网络价值估值({deviation:.1%})"
        elif deviation > 0.1:  # 当前价格略高于估值
            signal = "bearish"
            confidence = 0.6
            details = f"当前价格略高于网络价值估值({deviation:.1%})"
        else:
            signal = "neutral"
            confidence = 0.5
            details = f"当前价格接近网络价值估值，偏差为{deviation:.1%}"
            
        return {
            "signal": signal,
            "confidence": confidence,
            "network_value": network_value,
            "estimated_price": estimated_price,
            "deviation": deviation,
            "details": details
        }
        
    def _calculate_relative_valuation(self):
        """
        相对估值分析 (与同类资产比较)
        """
        # 这里需要有一个同类资产的数据库
        # 由于缺乏实时数据，使用简化的逻辑
        
        # 获取当前市值
        current_price = self.market_data.get("price", 0)
        
        if self.asset == "BTC":
            circulating_supply = 19e6  # 约1900万BTC
        elif self.asset == "ETH":
            circulating_supply = 120e6  # 约1.2亿ETH
        else:
            circulating_supply = 1e9  # 默认10亿
            
        market_cap = current_price * circulating_supply
        
        if market_cap == 0:
            return {
                "signal": "neutral",
                "confidence": 0.5,
                "details": "缺少市值数据，无法进行相对估值分析"
            }
            
        # 为不同资产类别应用不同的相对估值逻辑
        if self.asset == "BTC":
            # 比特币与黄金相比
            gold_market_cap = 12e12  # 12万亿美元 (全球黄金市值)
            btc_gold_ratio = market_cap / gold_market_cap
            
            if btc_gold_ratio < 0.03:  # 低于黄金市值的3%
                signal = "bullish"
                confidence = 0.7
                details = f"比特币市值仅为黄金市值的{btc_gold_ratio:.1%}，相对估值较低"
            elif btc_gold_ratio > 0.10:  # 高于黄金市值的10%
                signal = "bearish"
                confidence = 0.7
                details = f"比特币市值已达黄金市值的{btc_gold_ratio:.1%}，相对估值较高"
            else:
                signal = "neutral"
                confidence = 0.5
                details = f"比特币市值为黄金市值的{btc_gold_ratio:.1%}，相对估值适中"
                
        elif self.asset == "ETH":
            # 以太坊与比特币相比
            btc_market_cap = 1e12  # 假设BTC市值约1万亿美元
            eth_btc_ratio = market_cap / btc_market_cap
            
            # 历史上ETH/BTC比率在0.03到0.08之间波动
            if eth_btc_ratio < 0.03:
                signal = "bullish"
                confidence = 0.7
                details = f"以太坊市值仅为比特币市值的{eth_btc_ratio:.1%}，处于历史低位，相对估值较低"
            elif eth_btc_ratio > 0.08:
                signal = "bearish"
                confidence = 0.7
                details = f"以太坊市值已达比特币市值的{eth_btc_ratio:.1%}，处于历史高位，相对估值较高"
            else:
                signal = "neutral"
                confidence = 0.5
                details = f"以太坊市值为比特币市值的{eth_btc_ratio:.1%}，相对估值适中"
                
        else:
            # 其他代币的通用相对估值逻辑
            # 此处可根据代币类别(DeFi, NFT, Layer-1等)添加更具体的估值逻辑
            # 由于缺乏具体资产信息，返回中性信号
            signal = "neutral"
            confidence = 0.5
            details = f"缺少{self.asset}的同类资产数据，无法进行精确的相对估值分析"
            
        return {
            "signal": signal,
            "confidence": confidence,
            "market_cap": market_cap,
            "details": details
        }
        
    def _calculate_fair_value_range(self, current_price):
        """
        计算公允价值区间
        
        结合多种估值方法，给出一个价值区间
        """
        # 为不同资产设置不同的价值区间计算方法
        if self.asset == "BTC":
            # 基于MVRV和NVT的历史值区间
            mvrv_ratio = self.onchain_metrics.get("mvrv_ratio", 1)
            nvt_ratio = self.onchain_metrics.get("nvt_ratio", 25)
            realized_price = self.onchain_metrics.get("realized_price", current_price * 0.8)
            
            # 根据MVRV计算区间
            if mvrv_ratio != 0:
                # 历史MVRV范围通常在0.8到3.5之间
                low_range = realized_price * 0.8  # 底部区域
                high_range = realized_price * 3.5  # 顶部区域
            else:
                # 如果没有MVRV数据，使用当前价格的±50%作为区间
                low_range = current_price * 0.5
                high_range = current_price * 1.5
                
            # 根据当前价格在区间中的位置确定信号
            position_in_range = (current_price - low_range) / (high_range - low_range) if high_range > low_range else 0.5
            
            if position_in_range < 0.3:  # 接近区间底部
                signal = "bullish"
                confidence = 0.8
                details = f"当前价格({current_price:.0f})接近估值区间底部({low_range:.0f})，表明被低估"
            elif position_in_range > 0.7:  # 接近区间顶部
                signal = "bearish"
                confidence = 0.8
                details = f"当前价格({current_price:.0f})接近估值区间顶部({high_range:.0f})，表明被高估"
            else:  # 区间中间
                signal = "neutral"
                confidence = 0.5
                details = f"当前价格({current_price:.0f})在估值区间中间，公允区间为{low_range:.0f}-{high_range:.0f}"
                
        elif self.asset == "ETH":
            # 计算以太坊的公允价值区间
            # 结合费用估值和质押收益率估值
            realized_price = self.onchain_metrics.get("realized_price", current_price * 0.8)
            
            # 使用实现价格的0.8到3倍作为区间
            low_range = realized_price * 0.8
            high_range = realized_price * 3.0
            
            # 根据当前价格在区间中的位置确定信号
            position_in_range = (current_price - low_range) / (high_range - low_range) if high_range > low_range else 0.5
            
            if position_in_range < 0.3:  # 接近区间底部
                signal = "bullish"
                confidence = 0.8
                details = f"当前价格({current_price:.0f})接近估值区间底部({low_range:.0f})，表明被低估"
            elif position_in_range > 0.7:  # 接近区间顶部
                signal = "bearish"
                confidence = 0.8
                details = f"当前价格({current_price:.0f})接近估值区间顶部({high_range:.0f})，表明被高估"
            else:  # 区间中间
                signal = "neutral"
                confidence = 0.5
                details = f"当前价格({current_price:.0f})在估值区间中间，公允区间为{low_range:.0f}-{high_range:.0f}"
                
        else:
            # 其他代币的通用区间估值
            # 使用当前价格的±50%作为区间
            low_range = current_price * 0.5
            high_range = current_price * 1.5
            
            # 由于缺乏具体估值方法，返回中性信号
            signal = "neutral"
            confidence = 0.5
            details = f"缺少{self.asset}的特定估值模型，使用默认价值区间{low_range:.0f}-{high_range:.0f}"
            
        return {
            "signal": signal,
            "confidence": confidence,
            "low_range": low_range,
            "high_range": high_range,
            "position_in_range": position_in_range if 'position_in_range' in locals() else 0.5,
            "details": details
        }
        
    def _synthesize_valuation_results(self, valuation_results, current_price):
        """
        综合各种估值结果，得出最终估值信号
        
        Args:
            valuation_results: 各估值模型的结果
            current_price: 当前价格
            
        Returns:
            tuple: (综合信号, 信心水平, 详细推理)
        """
        # 初始化权重和得分
        method_weights = {
            "stock_to_flow": 0.15,
            "thermocap_multiple": 0.10,
            "nvt_analysis": 0.15,
            "mvrv_analysis": 0.20,
            "puell_multiple": 0.10,
            "fee_model": 0.15,
            "staking_yield": 0.15,
            "network_value_analysis": 0.15,
            "relative_valuation": 0.10,
            "fair_value_range": 0.25  # 公允价值区间权重较高
        }
        
        bullish_score = 0
        bearish_score = 0
        total_weight = 0
        reasoning = {}
        
        # 计算加权得分
        for method, result in valuation_results.items():
            if not result:
                continue
                
            weight = method_weights.get(method, 0.1)
            signal = result.get("signal", "neutral")
            confidence = result.get("confidence", 0.5)
            
            if signal == "bullish":
                bullish_score += weight * confidence
            elif signal == "bearish":
                bearish_score += weight * confidence
                
            total_weight += weight
            
            # 记录详细推理
            reasoning[method] = {
                "signal": signal,
                "confidence": confidence,
                "details": result.get("details", "")
            }
            
        # 如果没有足够的估值结果，返回中性信号
        if total_weight < 0.3:
            return "neutral", 0.5, {"insufficient_data": "估值方法有限，无法提供可靠的综合估值"}
            
        # 归一化得分
        if total_weight > 0:
            bullish_score /= total_weight
            bearish_score /= total_weight
            
        # 确定最终信号
        if bullish_score > bearish_score and bullish_score > 0.6:
            signal = "bullish"
            confidence = bullish_score
        elif bearish_score > bullish_score and bearish_score > 0.6:
            signal = "bearish"
            confidence = bearish_score
        else:
            signal = "neutral"
            confidence = 0.5
            
        return signal, confidence, reasoning


def valuation_agent(state: AgentState):
    """估值分析代理入口函数"""
    agent = ValuationAnalysisAgent()
    return agent.analyze(state)