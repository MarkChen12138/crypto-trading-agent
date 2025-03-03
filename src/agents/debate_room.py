from langchain_core.messages import HumanMessage
from src.agents.state import AgentState, show_agent_reasoning, show_workflow_status
import json
import ast

class DebateRoomAgent:
    """
    辩论室代理 - 综合评估看多和看空观点，得出平衡结论
    
    功能:
    1. 分析看多和看空研究者的论点
    2. 评估双方论点的质量和说服力
    3. 权衡不同因素，形成综合判断
    4. 给出最终信号和信心水平
    """
    
    def __init__(self):
        """初始化辩论室代理"""
        pass
        
    def analyze(self, state: AgentState):
        """执行辩论分析"""
        show_workflow_status("辩论室")
        if "metadata" in state and "show_reasoning" in state.get("metadata", {}):
            show_reasoning = state["metadata"]["show_reasoning"]
        else:
            show_reasoning = False
        
        # 获取研究者代理的消息
        researcher_messages = {}
        
        for message in state["messages"]:
            if message.name in ["researcher_bull_agent", "researcher_bear_agent"]:
                researcher_messages[message.name] = message
                
        # 检查是否有必要的研究消息
        if len(researcher_messages) < 2:
            # 缺少研究信息，返回中性信号
            message_content = {
                "signal": "neutral",
                "confidence": 0.5,
                "bull_confidence": 0.5,
                "bear_confidence": 0.5,
                "debate_summary": ["缺少完整的研究分析，无法进行有效辩论"],
                "reasoning": "缺少看多或看空研究者的分析，暂时保持中立立场"
            }
            
            message = HumanMessage(
                content=json.dumps(message_content),
                name="debate_room_agent",
            )
            
            if show_reasoning:
                show_agent_reasoning(message_content, "辩论室")
                
            show_workflow_status("辩论室", "completed")
            return {
                "messages": state["messages"] + [message],
                "data": state["data"],
            }
            
        # 解析研究者论点
        try:
            bull_content = json.loads(researcher_messages["researcher_bull_agent"].content)
            bear_content = json.loads(researcher_messages["researcher_bear_agent"].content)
        except Exception as e:
            try:
                bull_content = ast.literal_eval(researcher_messages["researcher_bull_agent"].content)
                bear_content = ast.literal_eval(researcher_messages["researcher_bear_agent"].content)
            except Exception as e:
                print(f"解析研究者内容时出错: {e}")
                # 返回中性信号
                message_content = {
                    "signal": "neutral",
                    "confidence": 0.5,
                    "bull_confidence": 0.5,
                    "bear_confidence": 0.5,
                    "debate_summary": ["解析研究分析时出错，无法进行有效辩论"],
                    "reasoning": f"解析错误: {e}"
                }
                
                message = HumanMessage(
                    content=json.dumps(message_content),
                    name="debate_room_agent",
                )
                
                if show_reasoning:
                    show_agent_reasoning(message_content, "辩论室")
                    
                show_workflow_status("辩论室", "completed")
                return {
                    "messages": state["messages"] + [message],
                    "data": state["data"],
                }
                
        # 提取研究者信心水平和论点
        try:
            bull_confidence = bull_content.get("confidence", 0.5)
            bear_confidence = bear_content.get("confidence", 0.5)
            bull_points = bull_content.get("thesis_points", [])
            bear_points = bear_content.get("thesis_points", [])
        except Exception as e:
            print(f"提取研究者数据时出错: {e}")
            bull_confidence = 0.5
            bear_confidence = 0.5
            bull_points = []
            bear_points = []
            
        # 分析辩论双方的论点
        debate_summary = []
        
        # 添加看多论点
        debate_summary.append("看多论点:")
        for i, point in enumerate(bull_points, 1):
            debate_summary.append(f"+ {i}. {point}")
            
        # 添加看空论点
        debate_summary.append("\n看空论点:")
        for i, point in enumerate(bear_points, 1):
            debate_summary.append(f"- {i}. {point}")
            
        # 评估论点质量和数量
        bull_quality = self._evaluate_thesis_quality(bull_points)
        bear_quality = self._evaluate_thesis_quality(bear_points)
        
        # 调整信心水平
        adjusted_bull_confidence = (bull_confidence * 0.7) + (bull_quality * 0.3)
        adjusted_bear_confidence = (bear_confidence * 0.7) + (bear_quality * 0.3)
        
        # 确定最终信号
        confidence_difference = adjusted_bull_confidence - adjusted_bear_confidence
        
        if abs(confidence_difference) < 0.1:  # 信心差距很小，几乎平衡
            signal = "neutral"
            reasoning = "看多和看空论点几乎同样有说服力，市场可能处于横盘整理阶段"
            confidence = 0.5  # 中等信心
        elif confidence_difference > 0:  # 看多更有说服力
            # 根据信心差距确定信号强度
            if confidence_difference > 0.3:
                signal = "bullish"
                reasoning = "看多论点明显更有说服力，市场可能即将上涨"
                confidence = adjusted_bull_confidence
            else:
                signal = "bullish"
                reasoning = "看多论点略占优势，但仍存在重要风险因素"
                confidence = adjusted_bull_confidence * 0.8  # 稍微降低信心
        else:  # 看空更有说服力
            # 根据信心差距确定信号强度
            if confidence_difference < -0.3:
                signal = "bearish"
                reasoning = "看空论点明显更有说服力，市场可能面临回调"
                confidence = adjusted_bear_confidence
            else:
                signal = "bearish"
                reasoning = "看空论点略占优势，建议谨慎行事"
                confidence = adjusted_bear_confidence * 0.8  # 稍微降低信心
                
        # 准备最终辩论结果
        message_content = {
            "signal": signal,
            "confidence": confidence,
            "bull_confidence": adjusted_bull_confidence,
            "bear_confidence": adjusted_bear_confidence,
            "debate_summary": debate_summary,
            "reasoning": reasoning
        }
        
        # 创建消息
        message = HumanMessage(
            content=json.dumps(message_content),
            name="debate_room_agent",
        )
        
        if show_reasoning:
            show_agent_reasoning(message_content, "辩论室")
            
        show_workflow_status("辩论室", "completed")
        return {
            "messages": state["messages"] + [message],
            "data": {
                **state["data"],
                "debate_analysis": message_content
            }
        }
        
    def _evaluate_thesis_quality(self, thesis_points):
        """评估论点质量，返回0-1的分数"""
        if not thesis_points:
            return 0.0
            
        # 评估论点数量（最多5分）
        points_count = min(len(thesis_points), 5) / 5
        
        # 评估论点质量（基于长度和关键词）
        quality_scores = []
        
        key_technical_terms = ["趋势", "支撑", "阻力", "均线", "MACD", "RSI", "波动率", "成交量"]
        key_fundamental_terms = ["链上", "活跃地址", "交易量", "持有者", "鲸鱼", "矿工", "质押", "网络"]
        key_sentiment_terms = ["情绪", "社交媒体", "新闻", "KOL", "恐慌", "贪婪", "信心"]
        key_valuation_terms = ["估值", "内在价值", "低估", "高估", "DCF", "风险回报"]
        
        all_key_terms = key_technical_terms + key_fundamental_terms + key_sentiment_terms + key_valuation_terms
        
        for point in thesis_points:
            # 计算长度得分（较长的论点通常更详细）
            length = len(point)
            length_score = min(length / 100, 1.0)  # 100字符或以上得满分
            
            # 计算关键词得分
            keyword_count = sum(1 for term in all_key_terms if term in point)
            keyword_score = min(keyword_count / 3, 1.0)  # 包含3个或以上关键词得满分
            
            # 综合分数
            point_score = (length_score * 0.4) + (keyword_score * 0.6)
            quality_scores.append(point_score)
            
        # 计算平均质量得分
        avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0
        
        # 综合分数（数量和质量各占50%）
        return (points_count * 0.5) + (avg_quality * 0.5)


def debate_room_agent(state: AgentState):
    """辩论室代理入口函数"""
    agent = DebateRoomAgent()
    return agent.analyze(state)