from langchain_core.messages import HumanMessage
from src.agents.state import AgentState, show_agent_reasoning, show_workflow_status
import logging

# 设置日志记录
logger = logging.getLogger('sentiment_agent')

class SentimentAnalysisAgent:
    """
    情感分析代理 - 专注于基于RapidAPI新闻的Gemini情感分析
    
    主要功能:
    1. 接收来自market_data代理的新闻数据
    2. 分析新闻情感，生成投资信号
    3. 计算情感分布和比率
    """
    
    def __init__(self):
        """初始化情感分析代理"""
        pass
        
    def analyze(self, state: AgentState):
        """执行情感分析"""
        show_workflow_status("情感分析代理")
        data = state.get("data", {})
        
        # 获取基础数据
        asset = data.get("asset", "BTC")
        symbol = data.get("symbol", "BTC/USDT")
        
        # 从market_data获取新闻数据
        news_data = data.get("news_data", {})
        
        # 分析新闻情感
        sentiment_results = self._analyze_news_sentiment(news_data)
        
        # 计算情感信号
        signal, confidence = self._calculate_sentiment_signal(sentiment_results)
        
        # 更新数据
        updated_data = {
            **data,
            "sentiment_distribution": sentiment_results,
            "sentiment_signal": {
                "signal": signal,
                "confidence": confidence
            }
        }
        
        # 创建消息
        positive_pct = sentiment_results.get("positive_percent", 0)
        negative_pct = sentiment_results.get("negative_percent", 0)
        neutral_pct = sentiment_results.get("neutral_percent", 0)
        overall_sentiment = sentiment_results.get("overall", "中性")
        
        message_content = f"已完成{asset}的情感分析。\n\n"
        message_content += f"情感分析结果: {overall_sentiment}\n"
        
        details = sentiment_results.get("details", {})
        if details:
            ratio = details.get("sentiment_ratio", 1.0)
            total = details.get("total_news", 0)
            message_content += f"积极新闻: {positive_pct:.1f}% ({details.get('positive_count', 0)})\n"
            message_content += f"消极新闻: {negative_pct:.1f}% ({details.get('negative_count', 0)})\n"
            message_content += f"中性新闻: {neutral_pct:.1f}% ({details.get('neutral_count', 0)})\n"
            message_content += f"总新闻数: {total}\n"
            message_content += f"积极/消极比例: {ratio:.2f}\n"
            message_content += f"投资信号: {signal} (置信度: {confidence:.1%})"
        
        show_agent_reasoning(message_content, "情感分析代理")
        message = HumanMessage(content=message_content)
        
        messages = state["messages"] + [message]
        show_workflow_status("情感分析代理", "completed")
        
        return {
            "messages": messages,
            "data": updated_data
        }
    
    def _analyze_news_sentiment(self, news_data):
        """分析新闻情感"""
        try:
            # 检查是否有新闻数据 - 增强检查逻辑，兼容不同的数据结构
            if not news_data:
                logger.warning("没有获取到新闻数据，无法进行情感分析")
                return {}
                
            # 检查news_data结构并获取新闻列表
            news_list = []
            if 'news' in news_data:
                news_list = news_data.get('news', [])
            elif isinstance(news_data, list):
                news_list = news_data
            else:
                logger.warning(f"新闻数据格式不符合预期: {type(news_data)}")
                logger.info(f"新闻数据内容: {news_data}")
                return {}
            
            if not news_list:
                logger.warning("新闻列表为空，无法进行情感分析")
                return {}
            
            # 计算情感分布
            sentiment_counts = {'positive': 0, 'negative': 0, 'neutral': 0}
            for article in news_list:
                if 'sentiment' in article:
                    sentiment = article.get('sentiment', '').lower()
                    if sentiment in sentiment_counts:
                        sentiment_counts[sentiment] += 1
            
            total = sum(sentiment_counts.values())
            if total == 0:
                logger.warning("没有找到有效的情感分析结果")
                return {}
                
            # 计算情感分布
            sentiment_distribution = {
                'positive_percent': (sentiment_counts['positive'] / total) * 100 if total > 0 else 0,
                'negative_percent': (sentiment_counts['negative'] / total) * 100 if total > 0 else 0,
                'neutral_percent': (sentiment_counts['neutral'] / total) * 100 if total > 0 else 0,
                'overall': self._determine_overall_sentiment(sentiment_counts)
            }
            
            # 添加详细的情感分析结果
            sentiment_distribution['details'] = {
                'positive_count': sentiment_counts['positive'],
                'negative_count': sentiment_counts['negative'],
                'neutral_count': sentiment_counts['neutral'],
                'total_news': total,
                'sentiment_ratio': sentiment_counts['positive'] / max(sentiment_counts['negative'], 1)
            }
            
            logger.info(f"完成新闻情感分析，积极:{sentiment_counts['positive']}，"
                      f"消极:{sentiment_counts['negative']}，中性:{sentiment_counts['neutral']}")
            
            return sentiment_distribution
        except Exception as e:
            logger.error(f"分析新闻情感时出错: {e}")
            return {}
    
    def _calculate_sentiment_signal(self, sentiment_results):
        """计算情感分析投资信号"""
        if not sentiment_results or "details" not in sentiment_results:
            return "neutral", 0.5
            
        ratio = sentiment_results['details']['sentiment_ratio']
        
        # 根据积极/消极比例确定信号
        if ratio > 2.5:  # 极度看涨
            signal = "bullish"
            confidence = 0.9
        elif ratio > 1.5:  # 看涨
            signal = "bullish"
            confidence = 0.7
        elif ratio < 0.4:  # 极度看跌
            signal = "bearish"
            confidence = 0.9
        elif ratio < 0.67:  # 看跌
            signal = "bearish"
            confidence = 0.7
        else:  # 中性
            signal = "neutral"
            confidence = 0.6
        
        logger.info(f"根据情感分析生成信号：{signal}，置信度：{confidence:.1%}")
        return signal, confidence
            
    def _determine_overall_sentiment(self, counts):
        """根据情感计数确定整体情感"""
        if counts['positive'] > counts['negative'] * 1.5:
            return "极度积极"
        elif counts['positive'] > counts['negative']:
            return "积极"
        elif counts['negative'] > counts['positive'] * 1.5:
            return "极度消极"
        elif counts['negative'] > counts['positive']:
            return "消极"
        else:
            return "中性"


def sentiment_agent(state: AgentState):
    """情感分析代理入口函数"""
    agent = SentimentAnalysisAgent()
    return agent.analyze(state)