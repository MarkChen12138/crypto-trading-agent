from langchain_core.messages import HumanMessage
from src.agents.state import AgentState, show_agent_reasoning, show_workflow_status
import json
import ast

class BullishResearcherAgent:
    """
    看多研究代理 - 整合各类分析，积极寻找看多信号和投资机会
    
    功能:
    1. 分析各代理的信号，筛选有利的看多因素
    2. 构建看多论点和投资论文
    3. 确定看多信号的可信度
    """
    
    def __init__(self):
        """初始化看多研究代理"""
        pass
        
    def analyze(self, state: AgentState):
        """执行看多研究分析"""
        show_workflow_status("看多研究代理")
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
            # 信息不足，返回低信心的看多信号
            message_content = {
                "perspective": "bullish",
                "confidence": 0.4,
                "thesis_points": ["信息有限，暂时持乐观态度"],
                "reasoning": "分析信息有限，无法构建完整的看多论点"
            }
            
            message = HumanMessage(
                content=json.dumps(message_content),
                name="researcher_bull_agent",
            )
            
            if show_reasoning:
                show_agent_reasoning(message_content, "看多研究代理")
                
            show_workflow_status("看多研究代理", "completed")
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
            
        # 构建看多论点
        bullish_points = []
        confidence_scores = []
        
        # 1. 技术分析论点
        if "technical" in signals:
            tech_signal = signals["technical"]["signal"]
            tech_confidence = signals["technical"]["confidence"]
            
            if tech_signal == "bullish":
                # 直接使用看多信号
                bullish_points.append(
                    f"技术分析显示强烈的看多动能，信心指数: {tech_confidence:.0%}"
                )
                confidence_scores.append(tech_confidence)
                
                # 添加详细技术指标
                if signals["technical"].get("details"):
                    tech_details = signals["technical"]["details"]
                    
                    # 趋势跟踪
                    if "trend_following" in tech_details:
                        trend = tech_details["trend_following"]
                        if trend["signal"] == "bullish":
                            bullish_points.append(
                                f"价格形成上升趋势，移动平均线呈多头排列"
                            )
                            
                    # 动量
                    if "momentum" in tech_details:
                        momentum = tech_details["momentum"]
                        if momentum["signal"] == "bullish":
                            bullish_points.append(
                                f"价格动量强劲，MACD呈现金叉信号"
                            )
                            
                    # 支撑阻力
                    if "support_resistance" in tech_details:
                        sr = tech_details["support_resistance"]
                        if sr["signal"] == "bullish":
                            bullish_points.append(
                                f"价格突破关键阻力位，确认上行趋势"
                            )
                            
            else:
                # 即使是看空或中性信号，也寻找积极因素
                bullish_points.append(
                    f"虽然整体技术面呈{tech_signal}，但市场超卖区域可能提供买入机会"
                )
                confidence_scores.append(0.4)  # 较低信心
                
        # 2. 链上分析论点
        if "onchain" in signals:
            onchain_signal = signals["onchain"]["signal"]
            onchain_confidence = signals["onchain"]["confidence"]
            
            if onchain_signal == "bullish":
                bullish_points.append(
                    f"链上数据显示积极的网络活动和用户增长，信心指数: {onchain_confidence:.0%}"
                )
                confidence_scores.append(onchain_confidence)
                
                # 添加详细链上指标
                if signals["onchain"].get("details"):
                    onchain_details = signals["onchain"]["details"]
                    
                    # 网络活动
                    if "network_activity" in onchain_details:
                        activity = onchain_details["network_activity"]
                        if activity["signal"] == "bullish":
                            bullish_points.append(
                                f"活跃地址数量增加，表明用户参与度提高"
                            )
                            
                    # 持币分布
                    if "holder_distribution" in onchain_details:
                        holders = onchain_details["holder_distribution"]
                        if holders["signal"] == "bullish":
                            bullish_points.append(
                                f"长期持有者比例增加，减少了卖压风险"
                            )
                            
                    # 估值指标
                    if "valuation_metrics" in onchain_details:
                        valuation = onchain_details["valuation_metrics"]
                        if valuation["signal"] == "bullish":
                            bullish_points.append(
                                f"链上估值指标如MVRV和NVT处于合理区域，表明价格可能被低估"
                            )
                            
            else:
                # 寻找积极因素
                bullish_points.append(
                    f"尽管链上活动整体{onchain_signal}，但鲸鱼地址持仓稳定，表明机构信心"
                )
                confidence_scores.append(0.4)  # 较低信心
                
        # 3. 情绪分析论点
        if "sentiment" in signals:
            sentiment_signal = signals["sentiment"]["signal"]
            sentiment_confidence = signals["sentiment"]["confidence"]
            
            if sentiment_signal == "bullish":
                bullish_points.append(
                    f"市场情绪积极乐观，社交媒体和新闻报道呈正面态度，信心指数: {sentiment_confidence:.0%}"
                )
                confidence_scores.append(sentiment_confidence)
                
                # 添加详细情绪指标
                if signals["sentiment"].get("details"):
                    sentiment_details = signals["sentiment"]["details"]
                    
                    # 恐慌贪婪指数
                    if "fear_greed" in sentiment_details:
                        fear_greed = sentiment_details["fear_greed"]
                        if fear_greed["signal"] == "bullish":
                            bullish_points.append(
                                f"恐慌贪婪指数显示市场从恐慌向中性区域转变，这通常是很好的买入时机"
                            )
                            
                    # KOL情绪
                    if "kol_sentiment" in sentiment_details:
                        kol = sentiment_details["kol_sentiment"]
                        if kol["signal"] == "bullish":
                            bullish_points.append(
                                f"主要行业KOL持乐观态度，可能影响更多投资者买入"
                            )
                            
            elif sentiment_signal == "bearish":
                # 逆向思维，过度悲观可能是买入信号
                bullish_points.append(
                    f"市场情绪过度悲观，根据逆向投资理论，这可能是买入的良机"
                )
                confidence_scores.append(0.5)  # 中等信心
                
            else:
                bullish_points.append(
                    f"市场情绪中性，但社交媒体讨论热度上升，可能预示着即将到来的上涨"
                )
                confidence_scores.append(0.4)  # 较低信心
                
        # 4. 估值分析论点
        if "valuation" in signals:
            valuation_signal = signals["valuation"]["signal"]
            valuation_confidence = signals["valuation"]["confidence"]
            
            if valuation_signal == "bullish":
                bullish_points.append(
                    f"根据估值分析，当前价格被低估，存在上涨空间，信心指数: {valuation_confidence:.0%}"
                )
                confidence_scores.append(valuation_confidence)
                
                # 添加估值详情
                if isinstance(signals["valuation"]["details"], dict):
                    valuation_details = signals["valuation"]["details"]
                    
                    # DCF分析
                    if "dcf_analysis" in valuation_details:
                        dcf = valuation_details["dcf_analysis"]
                        if isinstance(dcf, dict) and dcf.get("signal") == "bullish":
                            bullish_points.append(
                                f"DCF估值显示当前价格低于内在价值，提供安全边际"
                            )
                            
                    # 其他估值指标
                    if "owner_earnings_analysis" in valuation_details:
                        owner = valuation_details["owner_earnings_analysis"]
                        if isinstance(owner, dict) and owner.get("signal") == "bullish":
                            bullish_points.append(
                                f"所有者收益分析表明资产价格具有吸引力的风险/回报比"
                            )
                            
            else:
                # 寻找积极因素
                bullish_points.append(
                    f"尽管当前估值{valuation_signal}，但长期前景依然看好，基本面指标稳健"
                )
                confidence_scores.append(0.3)  # 低信心
                
        # 添加通用看多论点（如果特定论点不足）
        if len(bullish_points) < 3:
            general_points = [
                "加密市场正处于周期性底部区域，历史表明这是最佳买入时机",
                "机构投资者持续进入市场，增加了长期需求和价格支撑",
                "技术创新和采用率加速，增强了网络长期价值",
                "监管环境逐渐明朗，减少了不确定性，有利于长期投资",
                "宏观经济不确定性可能导致投资者寻求加密资产作为对冲工具"
            ]
            
            # 添加需要数量的通用论点
            needed = max(0, 3 - len(bullish_points))
            bullish_points.extend(general_points[:needed])
            
            # 为通用论点添加适中的信心分数
            confidence_scores.extend([0.5] * needed)
            
        # 计算总体信心分数
        if confidence_scores:
            avg_confidence = sum(confidence_scores) / len(confidence_scores)
        else:
            avg_confidence = 0.4  # 默认中低等信心
            
        # 准备消息内容
        message_content = {
            "perspective": "bullish",
            "confidence": avg_confidence,
            "thesis_points": bullish_points,
            "reasoning": "基于技术、链上数据、市场情绪和估值分析，综合得出看多结论"
        }
        
        # 创建消息
        message = HumanMessage(
            content=json.dumps(message_content),
            name="researcher_bull_agent",
        )
        
        if show_reasoning:
            show_agent_reasoning(message_content, "看多研究代理")
            
        show_workflow_status("看多研究代理", "completed")
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


def researcher_bull_agent(state: AgentState):
    """看多研究代理入口函数"""
    agent = BullishResearcherAgent()
    return agent.analyze(state)