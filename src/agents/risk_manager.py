import math
import numpy as np
import pandas as pd
from langchain_core.messages import HumanMessage
from src.agents.state import AgentState, show_agent_reasoning, show_workflow_status
import json
import ast

class RiskManagementAgent:
    """
    风险管理代理 - 负责评估市场风险，设置仓位大小限制，执行资金管理策略
    
    主要功能:
    1. 计算市场波动性指标
    2. 评估下行风险
    3. 设置最大仓位大小
    4. 根据市场条件调整风险敞口
    5. 压力测试不同市场情景
    """
    
    def __init__(self):
        """初始化风险管理代理"""
        self.risk_metrics = {}
        self.portfolio = None
        
    def analyze(self, state: AgentState):
        """执行风险分析和管理"""
        show_workflow_status("风险管理代理")
        if "metadata" in state and "show_reasoning" in state.get("metadata", {}):
            show_reasoning = state["metadata"]["show_reasoning"]
        else:
            show_reasoning = False
        data = state["data"]
        self.portfolio = data["portfolio"]
        
        # 获取辩论室的结论
        debate_message = next(
            msg for msg in state["messages"] if msg.name == "debate_room_agent")
            
        try:
            debate_results = json.loads(debate_message.content)
        except Exception as e:
            debate_results = ast.literal_eval(debate_message.content)
            
        # 获取价格数据
        prices_df = self._convert_prices_to_df(data["prices"])
        
        # 计算风险指标
        risk_metrics = self._calculate_risk_metrics(prices_df)
        
        # 评估市场风险得分
        market_risk_score = self._assess_market_risk(risk_metrics)
        
        # 设置最大仓位限制
        max_position_size, position_reasoning = self._set_position_limits(
            market_risk_score, 
            debate_results, 
            self.portfolio
        )
        
        # 进行压力测试
        stress_test_results = self._stress_test_portfolio(prices_df, self.portfolio)
        
        # 确定交易行动
        trading_action, action_reasoning = self._determine_trading_action(
            market_risk_score, 
            debate_results,
            risk_metrics
        )
        
        # 准备最终的风险评估和建议
        risk_assessment = {
            "risk_score": market_risk_score,
            "max_position_size": max_position_size,
            "trading_action": trading_action,
            "risk_metrics": {
                "volatility": float(risk_metrics["volatility"]),
                "value_at_risk_95": float(risk_metrics["var_95"]),
                "max_drawdown": float(risk_metrics["max_drawdown"]),
                "volatility_percentile": float(risk_metrics["volatility_percentile"]),
                "sharpe_ratio": float(risk_metrics["sharpe_ratio"]),
                "market_risk_score": market_risk_score,
                "stress_test_results": stress_test_results
            },
            "debate_analysis": {
                "bull_confidence": debate_results["bull_confidence"],
                "bear_confidence": debate_results["bear_confidence"],
                "debate_confidence": debate_results["confidence"],
                "debate_signal": debate_results["signal"]
            },
            "reasoning": (
                f"风险评分 {market_risk_score}/10: "
                f"波动率={risk_metrics['volatility']:.2%}, "
                f"VaR(95%)={risk_metrics['var_95']:.2%}, "
                f"最大回撤={risk_metrics['max_drawdown']:.2%}, "
                f"夏普比率={risk_metrics['sharpe_ratio']:.2f}, "
                f"辩论信号={debate_results['signal']}\n"
                f"仓位限制: {position_reasoning}\n"
                f"交易建议: {action_reasoning}"
            )
        }
        
        # 创建消息
        message = HumanMessage(
            content=json.dumps(risk_assessment),
            name="risk_management_agent",
        )
        
        if show_reasoning:
            show_agent_reasoning(risk_assessment, "风险管理代理")
            
        show_workflow_status("风险管理代理", "completed")
        return {
            "messages": state["messages"] + [message],
            "data": {
                **data,
                "risk_analysis": risk_assessment
            }
        }
        
    def _convert_prices_to_df(self, prices):
        """将价格数据转换为DataFrame"""
        if isinstance(prices, pd.DataFrame):
            return prices
            
        df = pd.DataFrame(prices)
        
        # 确保关键列存在
        if 'timestamp' in df.columns and isinstance(df['timestamp'][0], str):
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
        # 处理可能的列名不一致
        if 'close' not in df.columns and 'price' in df.columns:
            df['close'] = df['price']
            
        return df
        
    def _calculate_risk_metrics(self, prices_df):
        """计算各种风险指标"""
        # 确保数据足够
        if len(prices_df) < 7:
            # 返回默认风险指标
            return {
                "volatility": 0.03,  # 默认3%每日波动率
                "var_95": -0.05,     # 默认5%的VaR
                "max_drawdown": -0.10, # 默认10%最大回撤
                "volatility_percentile": 0.5, # 默认中等波动率
                "sharpe_ratio": 1.0  # 默认正常夏普比率
            }
            
        # 计算收益率
        if 'close' in prices_df.columns:
            returns = prices_df['close'].pct_change().dropna()
        else:
            # 尝试使用其他价格列
            for price_col in ['price', 'last', 'lastPrice']:
                if price_col in prices_df.columns:
                    returns = prices_df[price_col].pct_change().dropna()
                    break
            else:
                # 如果没有合适的价格列，返回默认值
                return {
                    "volatility": 0.03,
                    "var_95": -0.05,
                    "max_drawdown": -0.10,
                    "volatility_percentile": 0.5,
                    "sharpe_ratio": 1.0
                }
        
        # 1. 波动率 (日度波动率，年化)
        volatility = returns.std() * np.sqrt(365)
        
        # 2. 波动率百分位 (相对于过去120天)
        # 如果没有足够数据，使用波动率的75%作为波动率平均值的估计
        vol_mean = volatility * 0.75 if len(returns) < 30 else returns.rolling(window=min(30, len(returns))).std().mean() * np.sqrt(365)
        vol_std = volatility * 0.25 if len(returns) < 30 else returns.rolling(window=min(30, len(returns))).std().std() * np.sqrt(365)
        
        if vol_std == 0:
            volatility_percentile = 0.5  # 默认中等水平
        else:
            volatility_percentile = (volatility - vol_mean) / vol_std
            
        # 3. 风险价值 (VaR)
        # 如果数据不足，使用正态分布假设计算VaR
        if len(returns) < 30:
            # 假设正态分布，95% VaR约为1.65*标准差
            var_95 = -1.65 * returns.std()
        else:
            var_95 = returns.quantile(0.05)
            
        # 4. 最大回撤
        # 创建价格的滚动最大值序列
        if 'close' in prices_df.columns:
            prices = prices_df['close']
        else:
            for price_col in ['price', 'last', 'lastPrice']:
                if price_col in prices_df.columns:
                    prices = prices_df[price_col]
                    break
            else:
                prices = None
                
        if prices is not None:
            rolling_max = prices.rolling(window=min(len(prices), 30), min_periods=1).max()
            drawdowns = prices / rolling_max - 1
            max_drawdown = drawdowns.min()
        else:
            # 如果没有价格数据，估算最大回撤为波动率的2.5倍
            max_drawdown = -2.5 * returns.std()
            
        # 5. 夏普比率
        # 假设无风险利率为2%
        risk_free_rate = 0.02 / 365  # 日度无风险利率
        if volatility == 0:
            sharpe_ratio = 1.0  # 默认值
        else:
            sharpe_ratio = (returns.mean() - risk_free_rate) / returns.std() * np.sqrt(365)
            
        return {
            "volatility": volatility,
            "var_95": var_95,
            "max_drawdown": max_drawdown,
            "volatility_percentile": volatility_percentile,
            "sharpe_ratio": sharpe_ratio
        }
        
    def _assess_market_risk(self, risk_metrics):
        """评估市场风险得分 (1-10，10为最高风险)"""
        # 初始化风险得分
        risk_score = 0
        
        # 1. 基于波动率的风险评分
        volatility = risk_metrics["volatility"]
        if volatility > 0.12:  # 极高波动率 (>120% 年化)
            risk_score += 3
        elif volatility > 0.08:  # 高波动率 (>80% 年化)
            risk_score += 2
        elif volatility > 0.05:  # 中高波动率 (>50% 年化)
            risk_score += 1
            
        # 2. 基于波动率百分位的风险评分
        vol_percentile = risk_metrics["volatility_percentile"]
        if vol_percentile > 2.0:  # 极高于历史水平
            risk_score += 2
        elif vol_percentile > 1.0:  # 高于历史水平
            risk_score += 1
            
        # 3. 基于VaR的风险评分
        var_95 = risk_metrics["var_95"]
        if var_95 < -0.08:  # 极端下行风险
            risk_score += 2
        elif var_95 < -0.05:  # 严重下行风险
            risk_score += 1
            
        # 4. 基于最大回撤的风险评分
        max_drawdown = risk_metrics["max_drawdown"]
        if max_drawdown < -0.30:  # 极端回撤
            risk_score += 2
        elif max_drawdown < -0.15:  # 严重回撤
            risk_score += 1
            
        # 5. 基于夏普比率的风险评分
        sharpe = risk_metrics["sharpe_ratio"]
        if sharpe < 0:  # 负夏普比率
            risk_score += 2
        elif sharpe < 0.5:  # 低夏普比率
            risk_score += 1
            
        # 确保最终得分在1-10范围内
        risk_score = max(1, min(10, risk_score))
        
        return risk_score
        
    def _set_position_limits(self, risk_score, debate_results, portfolio):
        """设置最大仓位限制"""
        # 计算投资组合总价值
        total_portfolio_value = portfolio['cash']
        
        # 根据风险评分调整基础仓位比例
        if risk_score >= 8:  # 极高风险
            base_position_percent = 0.10  # 最多10%的资金可用于单一仓位
            reasoning = "极高风险环境，采用严格保守的仓位管理，限制单一头寸不超过10%"
        elif risk_score >= 6:  # 高风险
            base_position_percent = 0.20  # 最多20%的资金可用于单一仓位
            reasoning = "高风险环境，采用保守的仓位管理，限制单一头寸不超过20%"
        elif risk_score >= 4:  # 中等风险
            base_position_percent = 0.33  # 最多33%的资金可用于单一仓位
            reasoning = "中等风险环境，采用均衡的仓位管理，限制单一头寸不超过33%"
        else:  # 低风险
            base_position_percent = 0.50  # 最多50%的资金可用于单一仓位
            reasoning = "低风险环境，允许较高的资金利用率，单一头寸最高可达50%"
            
        # 根据辩论结果调整仓位
        confidence = debate_results["confidence"]
        signal = debate_results["signal"]
        
        # 如果辩论结果不确定性高，进一步降低仓位
        if confidence < 0.4:
            base_position_percent *= 0.7
            reasoning += "；辩论结果确定性较低，进一步降低仓位限制至原来的70%"
        
        # 如果辩论结果非常明确，可以略微提高仓位
        elif confidence > 0.8 and risk_score < 7:
            base_position_percent *= 1.2
            base_position_percent = min(base_position_percent, 0.5)  # 不超过50%
            reasoning += "；辩论结果确定性很高，适当提高仓位限制至原来的120%"
            
        # 计算最终的最大仓位大小
        max_position_size = total_portfolio_value * base_position_percent
        
        return max_position_size, reasoning
        
    def _stress_test_portfolio(self, prices_df, portfolio):
        """对投资组合进行压力测试"""
        # 设置压力测试情景
        scenarios = {
            "moderate_decline": -0.10,  # 10%的下跌
            "severe_decline": -0.25,    # 25%的下跌
            "crypto_crash": -0.50,      # 50%的崩溃
            "moderate_rise": 0.10,      # 10%的上涨
            "strong_rise": 0.25         # 25%的上涨
        }
        
        # 获取当前价格（如果可用）
        current_price = None
        if 'close' in prices_df.columns and len(prices_df) > 0:
            current_price = prices_df['close'].iloc[-1]
        elif 'price' in prices_df.columns and len(prices_df) > 0:
            current_price = prices_df['price'].iloc[-1]
            
        if current_price is None:
            current_price = 1.0  # 默认值
            
        # 测试不同场景下的投资组合表现
        stress_results = {}
        current_value = portfolio['cash']
        
        for scenario, price_change in scenarios.items():
            new_price = current_price * (1 + price_change)
            
            # 计算新的投资组合价值
            new_value = portfolio['cash']
            
            # 计算价值变化
            value_change = new_value - current_value
            value_change_percent = value_change / current_value if current_value > 0 else 0
            
            stress_results[scenario] = {
                "price_change": price_change,
                "portfolio_value_change": value_change,
                "portfolio_value_change_percent": value_change_percent
            }
            
        return stress_results
        
    def _determine_trading_action(self, risk_score, debate_results, risk_metrics):
        """根据风险评估和辩论结果确定交易行动"""
        signal = debate_results["signal"]
        confidence = debate_results["confidence"]
        
        # 根据风险评分设置阈值
        if risk_score >= 9:  # 极端风险
            action = "reduce_exposure"  # 无论辩论结果如何，都建议减少风险敞口
            reasoning = "市场风险极高，建议减少敞口，保护资本安全"
        elif risk_score >= 7:  # 高风险
            if signal == "bearish" and confidence > 0.5:
                action = "sell"
                reasoning = "市场风险较高，且辩论结果偏向看空，建议减仓"
            elif signal == "bullish" and confidence > 0.7:
                # 只有在非常确定的看多信号下才会在高风险环境中买入
                action = "small_buy"
                reasoning = "尽管市场风险较高，但辩论结果强烈看多，可小仓位买入"
            else:
                action = "hold"
                reasoning = "市场风险较高，建议持币观望，等待更明确信号"
        elif risk_score >= 4:  # 中等风险
            if signal == "bullish" and confidence > 0.6:
                action = "buy"
                reasoning = "市场风险适中，辩论结果看多，适合建仓"
            elif signal == "bearish" and confidence > 0.6:
                action = "sell"
                reasoning = "市场风险适中，辩论结果看空，建议减仓"
            else:
                action = "hold"
                reasoning = "市场风险适中，信号不明确，建议持币观望"
        else:  # 低风险
            if signal == "bullish":
                action = "buy"
                reasoning = "市场风险较低，辩论结果看多，适合积极建仓"
            elif signal == "bearish" and confidence > 0.7:
                action = "sell"
                reasoning = "尽管市场风险较低，但辩论结果强烈看空，建议谨慎减仓"
            else:
                action = "hold"
                reasoning = "市场风险较低，但信号不够明确，建议持仓观望"
                
        # 特殊情况：极端波动率或VaR
        if risk_metrics["volatility"] > 0.15 or risk_metrics["var_95"] < -0.12:
            if action == "buy":
                action = "small_buy"
                reasoning += "；但由于极端波动率或风险值，建议降低买入规模"
            elif action == "hold":
                action = "reduce_exposure"
                reasoning += "；考虑到极端波动率或风险值，建议适当减少风险敞口"
                
        # 映射到标准交易动作
        if action == "buy" or action == "small_buy":
            final_action = "buy"
        elif action == "sell" or action == "reduce_exposure":
            final_action = "sell"
        else:
            final_action = "hold"
            
        return final_action, reasoning
        

def risk_management_agent(state: AgentState):
    """风险管理代理入口函数"""
    agent = RiskManagementAgent()
    return agent.analyze(state)