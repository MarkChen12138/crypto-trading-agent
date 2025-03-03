from langchain_core.messages import HumanMessage
import json
import ast
from src.agents.state import AgentState, show_agent_reasoning, show_workflow_status

class PortfolioManagerAgent:
    """
    投资组合管理代理 - 负责根据各代理分析结果做出最终交易决策
    
    功能包括:
    1. 综合各代理信号，做出交易决策
    2. 确定具体的交易数量
    3. 遵守风险管理约束
    4. 生成详细交易报告
    """
    
    def __init__(self):
        """初始化投资组合管理代理"""
        self.portfolio = None
        self.symbol = None
        self.current_price = 0
        
    def analyze(self, state: AgentState):
        """执行投资组合分析和决策"""
        show_workflow_status("投资组合管理代理")
        if "metadata" in state and "show_reasoning" in state.get("metadata", {}):
            show_reasoning = state["metadata"]["show_reasoning"]
        else:
            show_reasoning = False
        
        # 获取必要数据
        self.portfolio = state["data"]["portfolio"]
        self.symbol = state["data"]["symbol"]
        
        # 获取当前价格
        prices = state["data"]["prices"]
        if prices and len(prices) > 0:
            self.current_price = prices[-1].get("close", 0)
        else:
            self.current_price = state["data"].get("market_data", {}).get("price", 0)
            
        if self.current_price <= 0:
            self.current_price = 1.0  # 防止除零错误的默认值
            
        # 获取各个代理的消息
        agent_messages = {}
        
        for message in state["messages"]:
            if message.name in ["technical_analyst_agent", "onchain_analysis_agent", 
                               "sentiment_agent", "valuation_agent", 
                               "risk_management_agent", "debate_room_agent"]:
                agent_messages[message.name] = message
                
        # 检查是否有必要的代理消息
        required_agents = ["technical_analyst_agent", "risk_management_agent"]
        for agent in required_agents:
            if agent not in agent_messages:
                # 缺少关键代理分析，返回保守决策
                message_content = {
                    "action": "hold",
                    "quantity": 0,
                    "confidence": 0.5,
                    "reasoning": f"缺少{agent}的分析，无法做出可靠决策，保持现有仓位"
                }
                
                message = HumanMessage(
                    content=json.dumps(message_content),
                    name="portfolio_management",
                )
                
                if show_reasoning:
                    show_agent_reasoning(message_content, "投资组合管理代理")
                    
                show_workflow_status("投资组合管理代理", "completed")
                return {
                    "messages": state["messages"] + [message],
                    "data": state["data"],
                }
                
        # 解析各代理的信号
        agent_signals = []
        
        try:
            # 技术分析信号
            technical_content = json.loads(agent_messages["technical_analyst_agent"].content)
            technical_signal = {
                "agent_name": "technical_analysis",
                "signal": technical_content["signal"],
                "confidence": float(technical_content["confidence"].replace("%", "")) / 100
            }
            agent_signals.append(technical_signal)
            
            # 链上分析信号（如果有）
            if "onchain_analysis_agent" in agent_messages:
                onchain_content = json.loads(agent_messages["onchain_analysis_agent"].content)
                onchain_signal = {
                    "agent_name": "onchain_analysis",
                    "signal": onchain_content["signal"],
                    "confidence": float(onchain_content["confidence"].replace("%", "")) / 100
                }
                agent_signals.append(onchain_signal)
                
            # 情绪分析信号（如果有）
            if "sentiment_agent" in agent_messages:
                sentiment_content = json.loads(agent_messages["sentiment_agent"].content)
                sentiment_signal = {
                    "agent_name": "sentiment_analysis",
                    "signal": sentiment_content["signal"],
                    "confidence": float(sentiment_content["confidence"].replace("%", "")) / 100
                }
                agent_signals.append(sentiment_signal)
                
            # 估值分析信号（如果有）
            if "valuation_agent" in agent_messages:
                valuation_content = json.loads(agent_messages["valuation_agent"].content)
                valuation_signal = {
                    "agent_name": "valuation_analysis",
                    "signal": valuation_content["signal"],
                    "confidence": float(valuation_content["confidence"].replace("%", "")) / 100
                }
                agent_signals.append(valuation_signal)
                
            # 辩论室信号（如果有）
            if "debate_room_agent" in agent_messages:
                debate_content = json.loads(agent_messages["debate_room_agent"].content)
                debate_signal = {
                    "agent_name": "debate_room",
                    "signal": debate_content["signal"],
                    "confidence": float(debate_content["confidence"])
                }
                agent_signals.append(debate_signal)
                
            # 风险管理信号
            risk_content = json.loads(agent_messages["risk_management_agent"].content)
            risk_signal = {
                "agent_name": "risk_management",
                "signal": risk_content["trading_action"],
                "confidence": 1.0  # 风险管理的约束是强制的
            }
            agent_signals.append(risk_signal)
            
        except Exception as e:
            print(f"解析代理信号时出错: {e}")
            # 出错时返回保守决策
            message_content = {
                "action": "hold",
                "quantity": 0,
                "confidence": 0.5,
                "reasoning": f"解析代理信号时出错: {e}，采取保守策略，保持现有仓位"
            }
            
            message = HumanMessage(
                content=json.dumps(message_content),
                name="portfolio_management",
            )
            
            if show_reasoning:
                show_agent_reasoning(message_content, "投资组合管理代理")
                
            show_workflow_status("投资组合管理代理", "completed")
            return {
                "messages": state["messages"] + [message],
                "data": state["data"],
            }
            
        # 获取风险管理约束
        try:
            max_position_size = risk_content["max_position_size"]
            trading_action = risk_content["trading_action"]
        except:
            max_position_size = self.portfolio["cash"] * 0.2  # 默认最大仓位为20%资金
            trading_action = "hold"  # 默认持仓策略
            
        # 设置代理权重
        agent_weights = {
            "technical_analysis": 0.25,
            "onchain_analysis": 0.15,
            "sentiment_analysis": 0.10,
            "valuation_analysis": 0.15,
            "debate_room": 0.35,
            "risk_management": 1.0  # 风险管理具有否决权
        }
        
        # 计算加权信号
        bullish_score = 0
        bearish_score = 0
        total_weight = 0
        
        for signal in agent_signals:
            if signal["agent_name"] == "risk_management":
                continue  # 风险管理单独处理
                
            agent_weight = agent_weights.get(signal["agent_name"], 0.0)
            weighted_confidence = signal["confidence"] * agent_weight
            
            if signal["signal"] == "bullish":
                bullish_score += weighted_confidence
            elif signal["signal"] == "bearish":
                bearish_score += weighted_confidence
                
            total_weight += agent_weight
            
        # 计算综合信心得分
        if total_weight > 0:
            bullish_score /= total_weight
            bearish_score /= total_weight
            
        # 确定初步行动
        if bullish_score > bearish_score and bullish_score > 0.6:
            preliminary_action = "buy"
            action_confidence = bullish_score
        elif bearish_score > bullish_score and bearish_score > 0.6:
            preliminary_action = "sell"
            action_confidence = bearish_score
        else:
            preliminary_action = "hold"
            action_confidence = max(0.5, 1 - abs(bullish_score - bearish_score))
            
        # 应用风险管理约束
        if trading_action == "buy" and preliminary_action == "buy":
            final_action = "buy"
        elif trading_action == "sell" and preliminary_action == "sell":
            final_action = "sell"
        elif trading_action == "hold" or preliminary_action == "hold":
            final_action = "hold"
        else:
            # 风险管理和初步行动不一致，优先风险管理
            final_action = trading_action
            action_confidence = 0.7  # 降低信心
            
        # 计算交易数量
        quantity = self._calculate_position_size(
            action=final_action,
            confidence=action_confidence,
            max_position_size=max_position_size
        )
        
        # 验证交易是否可行
        quantity = self._validate_trade(final_action, quantity)
        
        # 生成决策理由
        reasoning = self._generate_reasoning(
            action=final_action,
            original_action=preliminary_action,
            risk_action=trading_action,
            bullish_score=bullish_score,
            bearish_score=bearish_score,
            agent_signals=agent_signals
        )
        
        # 准备最终决策
        message_content = {
            "action": final_action,
            "quantity": quantity,
            "confidence": action_confidence,
            "agent_signals": agent_signals,
            "reasoning": reasoning
        }
        
        # 添加详细分析报告
        detailed_report = self._generate_detailed_report(
            action=final_action,
            quantity=quantity,
            confidence=action_confidence,
            agent_signals=agent_signals,
            max_position_size=max_position_size
        )
        
        message_content["detailed_report"] = detailed_report
        
        # 创建消息
        message = HumanMessage(
            content=json.dumps(message_content),
            name="portfolio_management",
        )
        
        if show_reasoning:
            show_agent_reasoning(message_content, "投资组合管理代理")
            
        show_workflow_status("投资组合管理代理", "completed")
        return {
            "messages": state["messages"] + [message],
            "data": state["data"],
        }
        
    def _calculate_position_size(self, action, confidence, max_position_size):
        """计算交易数量"""
        if action == "hold":
            return 0
            
        # 根据信心水平调整仓位大小
        position_percentage = min(confidence * 1.2, 1.0)  # 最高信心使用100%允许仓位
        target_position_value = max_position_size * position_percentage
        
        if action == "buy":
            # 计算可买入的数量
            max_quantity = target_position_value / self.current_price
            
            # 根据可用现金调整
            available_cash = self.portfolio["cash"]
            affordable_quantity = available_cash / self.current_price
            
            quantity = min(max_quantity, affordable_quantity)
            
        elif action == "sell":
            # 计算卖出数量
            current_position = self.portfolio.get("stock", 0)
            
            if current_position <= 0:
                return 0  # 没有持仓，无法卖出
                
            # 根据信心水平决定卖出比例
            sell_percentage = position_percentage
            quantity = current_position * sell_percentage
            
        else:
            quantity = 0
            
        return round(quantity, 8)  # 四舍五入到8位小数
        
    def _validate_trade(self, action, quantity):
        """验证交易的可行性，并调整数量"""
        if action == "buy":
            # 检查可用资金
            cost = quantity * self.current_price
            
            if cost > self.portfolio["cash"]:
                # 调整为可负担的最大数量
                quantity = self.portfolio["cash"] / self.current_price
                
        elif action == "sell":
            # 检查可用持仓
            current_position = self.portfolio.get("stock", 0)
            
            if quantity > current_position:
                quantity = current_position
                
        # 确保数量为正数
        quantity = max(0, quantity)
        
        # 如果数量太小，不执行交易
        min_trade_value = 10  # 最小交易价值（美元）
        if quantity * self.current_price < min_trade_value and quantity > 0:
            return 0
            
        return round(quantity, 8)
        
    def _generate_reasoning(self, action, original_action, risk_action, bullish_score, bearish_score, agent_signals):
        """生成决策理由"""
        # 准备各代理的信号摘要
        signal_summary = {}
        for signal in agent_signals:
            signal_summary[signal["agent_name"]] = {
                "signal": signal["signal"],
                "confidence": signal["confidence"]
            }
            
        # 生成摘要文本
        reasoning_parts = []
        
        # 总体信号强度
        if bullish_score > 0.7:
            reasoning_parts.append(f"整体市场信号强烈看多 (得分: {bullish_score:.2f})")
        elif bullish_score > 0.5:
            reasoning_parts.append(f"整体市场信号偏向看多 (得分: {bullish_score:.2f})")
        elif bearish_score > 0.7:
            reasoning_parts.append(f"整体市场信号强烈看空 (得分: {bearish_score:.2f})")
        elif bearish_score > 0.5:
            reasoning_parts.append(f"整体市场信号偏向看空 (得分: {bearish_score:.2f})")
        else:
            reasoning_parts.append(f"整体市场信号中性 (看多: {bullish_score:.2f}, 看空: {bearish_score:.2f})")
            
        # 风险管理约束
        if risk_action != original_action:
            reasoning_parts.append(f"风险管理建议{risk_action}，覆盖了原始信号{original_action}")
        else:
            reasoning_parts.append(f"风险管理与市场信号一致，均建议{action}")
            
        # 具体交易理由
        if action == "buy":
            reasoning_parts.append(f"技术分析{signal_summary.get('technical_analysis', {}).get('signal', 'neutral')}，"
                                 f"信心{signal_summary.get('technical_analysis', {}).get('confidence', 0):.2f}")
                                 
            if "onchain_analysis" in signal_summary:
                reasoning_parts.append(f"链上分析{signal_summary['onchain_analysis']['signal']}，"
                                     f"信心{signal_summary['onchain_analysis']['confidence']:.2f}")
                                     
            if "debate_room" in signal_summary:
                reasoning_parts.append(f"研究辩论结论{signal_summary['debate_room']['signal']}，"
                                     f"信心{signal_summary['debate_room']['confidence']:.2f}")
                                     
        elif action == "sell":
            reasoning_parts.append(f"技术分析{signal_summary.get('technical_analysis', {}).get('signal', 'neutral')}，"
                                 f"信心{signal_summary.get('technical_analysis', {}).get('confidence', 0):.2f}")
                                 
            if "onchain_analysis" in signal_summary:
                reasoning_parts.append(f"链上分析{signal_summary['onchain_analysis']['signal']}，"
                                     f"信心{signal_summary['onchain_analysis']['confidence']:.2f}")
                                     
            if "debate_room" in signal_summary:
                reasoning_parts.append(f"研究辩论结论{signal_summary['debate_room']['signal']}，"
                                     f"信心{signal_summary['debate_room']['confidence']:.2f}")
                                     
        else:  # hold
            reasoning_parts.append("市场信号不明确或相互矛盾，保持观望")
            
        return " ".join(reasoning_parts)
        
    def _generate_detailed_report(self, action, quantity, confidence, agent_signals, max_position_size):
        """生成详细的分析报告"""
        # 计算交易价值
        trade_value = quantity * self.current_price
        
        # 计算交易后的投资组合状态
        current_position = self.portfolio.get("stock", 0)
        current_cash = self.portfolio["cash"]
        
        if action == "buy":
            new_position = current_position + quantity
            new_cash = current_cash - trade_value
        elif action == "sell":
            new_position = current_position - quantity
            new_cash = current_cash + trade_value
        else:
            new_position = current_position
            new_cash = current_cash
            
        # 计算投资组合状态变化
        position_change = new_position - current_position
        position_change_percent = (position_change / current_position) * 100 if current_position > 0 else 0
        
        # 构建详细报告
        report = {
            "summary": {
                "symbol": self.symbol,
                "action": action,
                "quantity": quantity,
                "price": self.current_price,
                "trade_value": trade_value,
                "confidence": confidence
            },
            "portfolio": {
                "before": {
                    "cash": current_cash,
                    "position": current_position,
                    "position_value": current_position * self.current_price,
                    "total_value": current_cash + (current_position * self.current_price)
                },
                "after": {
                    "cash": new_cash,
                    "position": new_position,
                    "position_value": new_position * self.current_price,
                    "total_value": new_cash + (new_position * self.current_price)
                },
                "change": {
                    "position": position_change,
                    "position_percent": position_change_percent
                }
            },
            "risk_management": {
                "max_position_size": max_position_size,
                "position_utilization": (new_position * self.current_price) / max_position_size if max_position_size > 0 else 0
            },
            "agent_signals": {}
        }
        
        # 添加各代理的详细信号
        for signal in agent_signals:
            report["agent_signals"][signal["agent_name"]] = {
                "signal": signal["signal"],
                "confidence": signal["confidence"]
            }
            
        return report
        

def portfolio_management_agent(state: AgentState):
    """投资组合管理代理入口函数"""
    agent = PortfolioManagerAgent()
    return agent.analyze(state)