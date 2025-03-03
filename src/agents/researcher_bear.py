from langchain_core.messages import HumanMessage
from src.agents.state import AgentState, show_agent_reasoning, show_workflow_status
import json
import ast

class BearishResearcherAgent:
    """
    看空研究代理 - 整合各类分析，谨慎寻找看空信号和风险因素
    
    功能:
    1. 分析各代理的信号，筛选潜在的风险因素
    2. 构建看空论点和风险警示
    3. 确定看空信号的可信度
    """
    
    def __init__(self):
        """初始化看空研究代理"""
        pass
        
    def analyze(self, state: AgentState):
        """执行看空研究分析"""
        show_workflow_status("看空研究代理")
        if "metadata" in state and "show_reasoning" in state.get("metadata", {}):
            show_reasoning = state["metadata"]["show_reasoning"]
        else:
            show_reasoning = False
        
        # 获取各分析代理的消息
        agent_messages = {}
        
        for message in state["messages"]:
            if message.name in ["technical_analyst_agent", "onchain_analysis_agent", 
                               "sentiment_agent", "valuation_agent"]:
                agent_messages[message.name] = message
                
        # 检查是否有足够的分析信息
        if len(agent_messages) < 2:
            # 信息不足，返回低信心的看空信号
            message_content = {
                "perspective": "bearish",
                "confidence": 0.4,
                "thesis_points": ["信息有限，保持谨慎态度"],
                "reasoning": "分析信息有限，无法构建完整的看空论点，但风险管理要求我们保持谨慎"
            }
            
            message = HumanMessage(
                content=json.dumps(message_content),
                name="researcher_bear_agent",
            )
            
            if show_reasoning:
                show_agent_reasoning(message_content, "看空研究代理")
                
            show_workflow_status("看空研究代理", "completed")
            return {
                "messages": state["messages"] + [message],
                "data": state["data"],
            }
            
        # 解析各代理信号
        signals = {}
        
        try:
            # 技术分析信号
            if "technical_analyst_agent" in agent_messages:
                tech_content = json.loads(agent_messages["technical_analyst_agent"].content)
                signals["technical"] = {
                    "signal": tech_content["signal"],
                    "confidence": self._parse_confidence(tech_content["confidence"]),
                    "details": tech_content.get("strategy_signals", {})
                }
                
            # 链上分析信号
            if "onchain_analysis_agent" in agent_messages:
                onchain_content = json.loads(agent_messages["onchain_analysis_agent"].content)
                signals["onchain"] = {
                    "signal": onchain_content["signal"],
                    "confidence": self._parse_confidence(onchain_content["confidence"]),
                    "details": onchain_content.get("detailed_analysis", {})
                }
                
            # 情绪分析信号
            if "sentiment_agent" in agent_messages:
                sentiment_content = json.loads(agent_messages["sentiment_agent"].content)
                signals["sentiment"] = {
                    "signal": sentiment_content["signal"],
                    "confidence": self._parse_confidence(sentiment_content["confidence"]),
                    "details": sentiment_content.get("detailed_analysis", {})
                }
                
            # 估值分析信号
            if "valuation_agent" in agent_messages:
                valuation_content = json.loads(agent_messages["valuation_agent"].content)
                signals["valuation"] = {
                    "signal": valuation_content["signal"],
                    "confidence": self._parse_confidence(valuation_content["confidence"]),
                    "details": valuation_content.get("reasoning", {})
                }
                
        except Exception as e:
            print(f"解析代理信号时出错: {e}")
            signals = {}
            
        # 构建看空论点
        bearish_points = []
        confidence_scores = []
        
        # 1. 技术分析论点
        if "technical" in signals:
            tech_signal = signals["technical"]["signal"]
            tech_confidence = signals["technical"]["confidence"]
            
            if tech_signal == "bearish":
                # 直接使用看空信号
                bearish_points.append(
                    f"技术分析显示明显的下跌趋势，信心指数: {tech_confidence:.0%}"
                )
                confidence_scores.append(tech_confidence)
                
                # 添加详细技术指标
                if signals["technical"].get("details"):
                    tech_details = signals["technical"]["details"]
                    
                    # 趋势跟踪
                    if "trend_following" in tech_details:
                        trend = tech_details["trend_following"]
                        if trend["signal"] == "bearish":
                            bearish_points.append(
                                f"价格跌破重要支撑位，移动平均线呈空头排列"
                            )
                            
                    # 动量
                    if "momentum" in tech_details:
                        momentum = tech_details["momentum"]
                        if momentum["signal"] == "bearish":
                            bearish_points.append(
                                f"价格动量减弱，MACD呈现死叉信号"
                            )
                            
                    # 波动率
                    if "volatility" in tech_details:
                        volatility = tech_details["volatility"]
                        if volatility["signal"] == "bearish":
                            bearish_points.append(
                                f"波动率上升，可能暗示市场不稳定性增加"
                            )
                            
            elif tech_signal == "bullish":
                # 即使是看多信号，也寻找风险因素
                bearish_points.append(
                    f"虽然短期技术指标呈{tech_signal}，但过度买入可能导致回调风险"
                )
                confidence_scores.append(0.4)  # 较低信心
                
            else:
                # 中性信号，寻找消极因素
                bearish_points.append(
                    f"技术面缺乏明确方向，市场可能陷入盘整，动能不足以支撑持续上涨"
                )
                confidence_scores.append(0.5)  # 中等信心
                
        # 2. 链上分析论点
        if "onchain" in signals:
            onchain_signal = signals["onchain"]["signal"]
            onchain_confidence = signals["onchain"]["confidence"]
            
            if onchain_signal == "bearish":
                bearish_points.append(
                    f"链上数据显示网络活动下降，可能暗示用户兴趣减弱，信心指数: {onchain_confidence:.0%}"
                )
                confidence_scores.append(onchain_confidence)
                
                # 添加详细链上指标
                if signals["onchain"].get("details"):
                    onchain_details = signals["onchain"]["details"]
                    
                    # 网络活动
                    if "network_activity" in onchain_details:
                        activity = onchain_details["network_activity"]
                        if activity["signal"] == "bearish":
                            bearish_points.append(
                                f"链上交易量和活跃地址数量下降，表明实际使用率减少"
                            )
                            
                    # 持币分布
                    if "holder_distribution" in onchain_details:
                        holders = onchain_details["holder_distribution"]
                        if holders["signal"] == "bearish":
                            bearish_points.append(
                                f"大型持有者（鲸鱼）减少持仓，可能预示即将抛售"
                            )
                            
                    # 估值指标
                    if "valuation_metrics" in onchain_details:
                        valuation = onchain_details["valuation_metrics"]
                        if valuation["signal"] == "bearish":
                            bearish_points.append(
                                f"链上估值指标如MVRV和NVT处于历史高位，表明价格可能被高估"
                            )
                            
            else:
                # 寻找风险因素
                bearish_points.append(
                    f"尽管链上活动整体{onchain_signal}，但短期持有者比例上升，可能增加卖压"
                )
                confidence_scores.append(0.4)  # 较低信心
                
        # 3. 情绪分析论点
        if "sentiment" in signals:
            sentiment_signal = signals["sentiment"]["signal"]
            sentiment_confidence = signals["sentiment"]["confidence"]
            
            if sentiment_signal == "bearish":
                bearish_points.append(
                    f"市场情绪悲观，社交媒体和新闻报道呈负面态度，信心指数: {sentiment_confidence:.0%}"
                )
                confidence_scores.append(sentiment_confidence)
                
                # 添加详细情绪指标
                if signals["sentiment"].get("details"):
                    sentiment_details = signals["sentiment"]["details"]
                    
                    # 恐慌贪婪指数
                    if "fear_greed" in sentiment_details:
                        fear_greed = sentiment_details["fear_greed"]
                        if fear_greed["signal"] == "bearish":
                            bearish_points.append(
                                f"恐慌贪婪指数处于'极度贪婪'区域，历史上这通常预示着市场顶部"
                            )
                            
                    # 社交媒体情绪
                    if "social_sentiment" in sentiment_details:
                        social = sentiment_details["social_sentiment"]
                        if social["signal"] == "bearish":
                            bearish_points.append(
                                f"社交媒体情绪迅速转为负面，可能引发抛售浪潮"
                            )
                            
            elif sentiment_signal == "bullish":
                # 过度乐观可能是风险信号
                bearish_points.append(
                    f"市场情绪过度乐观，根据逆向投资理论，这可能预示着市场顶部"
                )
                confidence_scores.append(0.5)  # 中等信心
                
            else:
                bearish_points.append(
                    f"市场情绪中性，但社交媒体关注度下降，可能表明交易兴趣减弱"
                )
                confidence_scores.append(0.4)  # 较低信心
                
        # 4. 估值分析论点
        if "valuation" in signals:
            valuation_signal = signals["valuation"]["signal"]
            valuation_confidence = signals["valuation"]["confidence"]
            
            if valuation_signal == "bearish":
                bearish_points.append(
                    f"根据估值分析，当前价格被高估，存在回调风险，信心指数: {valuation_confidence:.0%}"
                )
                confidence_scores.append(valuation_confidence)
                
                # 添加估值详情
                if isinstance(signals["valuation"]["details"], dict):
                    valuation_details = signals["valuation"]["details"]
                    
                    # DCF分析
                    if "dcf_analysis" in valuation_details:
                        dcf = valuation_details["dcf_analysis"]
                        if isinstance(dcf, dict) and dcf.get("signal") == "bearish":
                            bearish_points.append(
                                f"DCF估值显示当前价格高于内在价值，缺乏安全边际"
                            )
                            
                    # 其他估值指标
                    if "owner_earnings_analysis" in valuation_details:
                        owner = valuation_details["owner_earnings_analysis"]
                        if isinstance(owner, dict) and owner.get("signal") == "bearish":
                            bearish_points.append(
                                f"所有者收益分析表明资产价格不具吸引力的风险/回报比"
                            )
                            
            else:
                # 寻找风险因素
                bearish_points.append(
                    f"尽管当前估值{valuation_signal}，但市场预期过高，实际增长可能难以支撑"
                )
                confidence_scores.append(0.4)  # 较低信心
                
        # 添加通用看空论点（如果特定论点不足）
        if len(bearish_points) < 3:
            general_points = [
                "加密市场高度波动，历史上经历过多次80%以上的回调",
                "监管风险持续存在，可能突然影响市场流动性和信心",
                "市场杠杆率和投机程度高，容易引发连锁清算事件",
                "宏观经济环境不确定，可能触发风险资产抛售",
                "加密货币生态系统仍然存在中心化和安全风险"
            ]
            
            # 添加需要数量的通用论点
            needed = max(0, 3 - len(bearish_points))
            bearish_points.extend(general_points[:needed])
            
            # 为通用论点添加适中的信心分数
            confidence_scores.extend([0.5] * needed)
            
        # 计算总体信心分数
        if confidence_scores:
            avg_confidence = sum(confidence_scores) / len(confidence_scores)
        else:
            avg_confidence = 0.4  # 默认中低等信心
            
        # 准备消息内容
        message_content = {
            "perspective": "bearish",
            "confidence": avg_confidence,
            "thesis_points": bearish_points,
            "reasoning": "基于技术、链上数据、市场情绪和估值分析，综合识别风险因素"
        }
        
        # 创建消息
        message = HumanMessage(
            content=json.dumps(message_content),
            name="researcher_bear_agent",
        )
        
        if show_reasoning:
            show_agent_reasoning(message_content, "看空研究代理")
            
        show_workflow_status("看空研究代理", "completed")
        return {
            "messages": state["messages"] + [message],
            "data": state["data"],
        }
        
    def _parse_confidence(self, confidence_value):
        """解析信心值，处理可能的字符串表示"""
        if isinstance(confidence_value, (int, float)):
            return confidence_value
            
        if isinstance(confidence_value, str):
            # 处理百分比字符串
            confidence_value = confidence_value.strip()
            if confidence_value.endswith('%'):
                try:
                    return float(confidence_value.rstrip('%')) / 100
                except ValueError:
                    pass
                    
            # 尝试直接解析为浮点数
            try:
                return float(confidence_value)
            except ValueError:
                pass
                
        # 默认值
        return 0.5


def researcher_bear_agent(state: AgentState):
    """看空研究代理入口函数"""
    agent = BearishResearcherAgent()
    return agent.analyze(state)