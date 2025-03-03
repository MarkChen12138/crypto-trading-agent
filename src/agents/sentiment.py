from langchain_core.messages import HumanMessage
from src.agents.state import AgentState, show_agent_reasoning, show_workflow_status
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import tweepy
import re
import requests
from src.tools.openrouter_config import get_chat_completion

class SentimentAnalysisAgent:
    """
    情绪分析代理 - 负责分析社交媒体、新闻和市场情绪指标，评估市场参与者的整体情绪
    
    分析的来源包括:
    1. 知名加密KOL的推文
    2. Reddit/社区讨论
    3. 新闻情绪
    4. 恐慌/贪婪指数
    """
    
    def __init__(self):
        """初始化情绪分析代理"""
        self.indicators = {}
        self.asset = None
        self.twitter_api = None
        self.initialize_twitter_api()
        
    def initialize_twitter_api(self):
        """初始化Twitter API连接"""
        try:
            import os
            from dotenv import load_dotenv
            
            # 加载环境变量
            load_dotenv()
            
            # 获取Twitter API密钥
            bearer_token = os.getenv("TWITTER_BEARER_TOKEN")
            
            if bearer_token:
                self.twitter_api = tweepy.Client(bearer_token=bearer_token)
                print("Twitter API初始化成功")
            else:
                print("未找到Twitter API密钥，将使用模拟数据")
                
        except Exception as e:
            print(f"初始化Twitter API时出错: {e}")
            self.twitter_api = None
        
    def analyze(self, state: AgentState):
        """执行情绪分析"""
        show_workflow_status("情绪分析代理")
        
        if "metadata" in state and "show_reasoning" in state.get("metadata", {}):
            show_reasoning = state["metadata"]["show_reasoning"]
        else:
            show_reasoning = False  # 默认不显示推理过程
        
        data = state.get("data", {})
        
        # 获取基础数据
        self.asset = data["asset"]
        sentiment_data = data.get("market_sentiment", {})
        
        # 如果没有情绪数据，获取模拟数据
        if not sentiment_data:
            sentiment_data = self.get_mock_sentiment_data()
            
        # 获取KOL推文
        kol_tweets = self.get_kol_tweets()
        
        # 分析各个指标
        signals = {}
        signals["fear_greed"] = self._analyze_fear_greed(sentiment_data)
        signals["social_sentiment"] = self._analyze_social_sentiment(sentiment_data)
        signals["kol_sentiment"] = self._analyze_kol_tweets(kol_tweets)
        signals["news_sentiment"] = self._analyze_news_sentiment(sentiment_data)
        
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
            name="sentiment_agent",
        )
        
        if show_reasoning:
            show_agent_reasoning(message_content, "情绪分析代理")
            
        show_workflow_status("情绪分析代理", "completed")
        return {
            "messages": [message],
            "data": {
                **data,
                "sentiment_analysis": message_content
            }
        }
        
    def get_mock_sentiment_data(self):
        """生成模拟情绪数据"""
        return {
            "fear_greed_index": 65,  # 0-100，0为极度恐惧，100为极度贪婪
            "fear_greed_classification": "Greed",
            "social_sentiment": 0.62,  # -1到1，-1为极度负面，1为极度正面
            "social_volume": 125000,
            "google_trends_value": 78,
            "twitter_sentiment": 0.58,
            "reddit_sentiment": 0.65,
            "news_sentiment": 0.45,
            "timestamp": datetime.now().timestamp()
        }
        
    def get_kol_tweets(self):
        """获取加密货币KOL的推文"""
        # 主要加密KOL的Twitter用户名
        kol_usernames = [
            "elonmusk",          # 埃隆·马斯克
            "cz_binance",        # 赵长鹏 (Binance创始人)
            "saylor",            # 迈克尔·塞勒 (MicroStrategy创始人)
            "VitalikButerin",    # 维塔利克 (以太坊创始人)
            "SBF_FTX",           # 萨姆·班克曼-弗里德 (FTX创始人)
            "brian_armstrong",   # 布莱恩·阿姆斯特朗 (Coinbase创始人)
            "aantonop",          # 安德烈亚斯·安东诺普洛斯
            "winklevoss",        # 卡梅隆·文克莱沃斯 (Gemini创始人)
            "novogratz",         # 迈克·诺沃格拉茨 (Galaxy Digital创始人)
            "Excellion"          # Samson Mow
        ]
        
        # 尝试使用Twitter API获取数据
        if self.twitter_api:
            try:
                all_tweets = []
                for username in kol_usernames[:4]:  # 限制查询数量，避免API限制
                    user = self.twitter_api.get_user(username=username)
                    if user and user.data:
                        user_id = user.data.id
                        tweets = self.twitter_api.get_users_tweets(
                            id=user_id, 
                            max_results=10,
                            tweet_fields=['created_at', 'public_metrics']
                        )
                        if tweets and tweets.data:
                            for tweet in tweets.data:
                                # 检查推文是否包含与加密货币相关的关键词
                                if self._is_crypto_related(tweet.text):
                                    all_tweets.append({
                                        'username': username,
                                        'text': tweet.text,
                                        'created_at': tweet.created_at,
                                        'likes': tweet.public_metrics['like_count'],
                                        'retweets': tweet.public_metrics['retweet_count']
                                    })
                return all_tweets
            except Exception as e:
                print(f"获取KOL推文时出错: {e}")
                
        # 如果API失败或不可用，返回模拟数据
        return self._get_mock_tweets()
        
    def _get_mock_tweets(self):
        """生成模拟KOL推文数据"""
        mock_tweets = [
            {
                'username': 'elonmusk',
                'text': 'Crypto is the future of finance. #Bitcoin #Dogecoin',
                'created_at': datetime.now() - timedelta(hours=5),
                'likes': 48500,
                'retweets': 9200
            },
            {
                'username': 'saylor',
                'text': 'Bitcoin is digital energy. The strategy is long. #BTC',
                'created_at': datetime.now() - timedelta(hours=12),
                'likes': 12300,
                'retweets': 2800
            },
            {
                'username': 'VitalikButerin',
                'text': 'Ethereum scaling solutions are progressing well. Optimism and ZK rollups are key for the future.',
                'created_at': datetime.now() - timedelta(hours=8),
                'likes': 15600,
                'retweets': 3100
            },
            {
                'username': 'cz_binance',
                'text': 'Markets are cyclical. Stay #SAFU and think long term. #BNB #Bitcoin',
                'created_at': datetime.now() - timedelta(hours=3),
                'likes': 8700,
                'retweets': 1500
            },
            {
                'username': 'brian_armstrong',
                'text': 'Regulatory clarity is important for crypto adoption. We continue to work with policymakers.',
                'created_at': datetime.now() - timedelta(hours=18),
                'likes': 5200,
                'retweets': 980
            }
        ]
        return mock_tweets
        
    def _is_crypto_related(self, text):
        """检查文本是否与加密货币相关"""
        crypto_keywords = [
            'bitcoin', 'btc', 'ethereum', 'eth', 'crypto', 'blockchain', 
            'defi', 'nft', 'altcoin', 'token', 'binance', 'coinbase',
            'wallet', 'mining', 'staking', 'hodl', 'bull', 'bear'
        ]
        
        text_lower = text.lower()
        for keyword in crypto_keywords:
            if keyword in text_lower:
                return True
        return False
        
    def _analyze_fear_greed(self, sentiment_data):
        """分析恐慌与贪婪指数"""
        fear_greed_index = sentiment_data.get("fear_greed_index")
        fear_greed_class = sentiment_data.get("fear_greed_classification")
        
        metrics = {
            "fear_greed_index": fear_greed_index,
            "classification": fear_greed_class
        }
        
        if fear_greed_index is None:
            return {
                "signal": "neutral",
                "confidence": 0.5,
                "metrics": metrics,
                "reasoning": "缺少恐慌贪婪指数数据"
            }
            
        # 反向指标：指数高表示市场过度乐观，潜在下跌风险
        if fear_greed_index >= 80:  # 极度贪婪
            signal = "bearish"  # 反向指标
            confidence = 0.8
            reasoning = f"恐慌贪婪指数处于极度贪婪区域({fear_greed_index})，表明市场可能过度乐观，存在回调风险"
        elif fear_greed_index >= 65:  # 贪婪
            signal = "bearish"  # 反向指标
            confidence = 0.7
            reasoning = f"恐慌贪婪指数处于贪婪区域({fear_greed_index})，表明市场情绪偏向乐观，可能接近短期顶部"
        elif fear_greed_index <= 20:  # 极度恐惧
            signal = "bullish"  # 反向指标
            confidence = 0.8
            reasoning = f"恐慌贪婪指数处于极度恐惧区域({fear_greed_index})，表明市场过度悲观，可能是买入机会"
        elif fear_greed_index <= 35:  # 恐惧
            signal = "bullish"  # 反向指标
            confidence = 0.7
            reasoning = f"恐慌贪婪指数处于恐惧区域({fear_greed_index})，表明市场情绪偏向悲观，可能接近短期底部"
        else:  # 中性区域
            signal = "neutral"
            confidence = 0.6
            reasoning = f"恐慌贪婪指数处于中性区域({fear_greed_index})，市场情绪相对平衡"
            
        return {
            "signal": signal,
            "confidence": confidence,
            "metrics": metrics,
            "reasoning": reasoning
        }
        
    def _analyze_social_sentiment(self, sentiment_data):
        """分析社交媒体情绪"""
        social_sentiment = sentiment_data.get("social_sentiment")
        twitter_sentiment = sentiment_data.get("twitter_sentiment")
        reddit_sentiment = sentiment_data.get("reddit_sentiment")
        social_volume = sentiment_data.get("social_volume")
        
        metrics = {
            "social_sentiment": social_sentiment,
            "twitter_sentiment": twitter_sentiment,
            "reddit_sentiment": reddit_sentiment,
            "social_volume": social_volume
        }
        
        if social_sentiment is None:
            return {
                "signal": "neutral",
                "confidence": 0.5,
                "metrics": metrics,
                "reasoning": "缺少社交媒体情绪数据"
            }
            
        # 计算情绪与信号
        reasoning_parts = []
        
        # 综合社交情绪分析
        if social_sentiment > 0.6:
            signal = "bullish"
            confidence = min(0.5 + social_sentiment / 2, 0.9)
            reasoning_parts.append(f"整体社交情绪非常积极({social_sentiment:.2f})")
        elif social_sentiment > 0.2:
            signal = "bullish"
            confidence = 0.6
            reasoning_parts.append(f"整体社交情绪偏积极({social_sentiment:.2f})")
        elif social_sentiment < -0.6:
            signal = "bearish"
            confidence = min(0.5 + abs(social_sentiment) / 2, 0.9)
            reasoning_parts.append(f"整体社交情绪非常消极({social_sentiment:.2f})")
        elif social_sentiment < -0.2:
            signal = "bearish"
            confidence = 0.6
            reasoning_parts.append(f"整体社交情绪偏消极({social_sentiment:.2f})")
        else:
            signal = "neutral"
            confidence = 0.5
            reasoning_parts.append(f"整体社交情绪中性({social_sentiment:.2f})")
            
        # 添加特定平台的情绪分析
        if twitter_sentiment is not None:
            reasoning_parts.append(f"Twitter情绪: {twitter_sentiment:.2f}")
        if reddit_sentiment is not None:
            reasoning_parts.append(f"Reddit情绪: {reddit_sentiment:.2f}")
            
        # 考虑社交讨论量
        if social_volume is not None:
            if social_volume > 100000:
                reasoning_parts.append(f"社交讨论量高({social_volume})，表明广泛关注")
                confidence = min(confidence + 0.1, 0.9)  # 增加信心
            elif social_volume < 50000:
                reasoning_parts.append(f"社交讨论量低({social_volume})，表明关注度不足")
                confidence = max(confidence - 0.1, 0.5)  # 降低信心
                
        reasoning = "，".join(reasoning_parts)
        
        return {
            "signal": signal,
            "confidence": confidence,
            "metrics": metrics,
            "reasoning": reasoning
        }
        
    def _analyze_kol_tweets(self, tweets):
        """分析KOL推文情绪"""
        if not tweets:
            return {
                "signal": "neutral",
                "confidence": 0.5,
                "metrics": {},
                "reasoning": "缺少KOL推文数据"
            }
            
        # 使用LLM分析推文情绪
        kol_sentiment = self._analyze_tweets_with_llm(tweets)
        
        # 计算加权影响力（根据点赞和转发）
        total_influence = 0
        weighted_sentiment = 0
        
        for tweet in tweets:
            influence = tweet.get('likes', 0) / 1000 + tweet.get('retweets', 0) / 500
            total_influence += influence
            
        # 生成结论
        if kol_sentiment > 0.6:
            signal = "bullish"
            confidence = min(0.5 + kol_sentiment / 2, 0.9)
            reasoning = f"KOL推文情绪非常积极(总分: {kol_sentiment:.2f})，主要意见领袖对市场持乐观态度"
        elif kol_sentiment > 0.2:
            signal = "bullish"
            confidence = 0.6
            reasoning = f"KOL推文情绪偏积极(总分: {kol_sentiment:.2f})，部分意见领袖表达了乐观看法"
        elif kol_sentiment < -0.6:
            signal = "bearish"
            confidence = min(0.5 + abs(kol_sentiment) / 2, 0.9)
            reasoning = f"KOL推文情绪非常消极(总分: {kol_sentiment:.2f})，主要意见领袖对市场持悲观态度"
        elif kol_sentiment < -0.2:
            signal = "bearish"
            confidence = 0.6
            reasoning = f"KOL推文情绪偏消极(总分: {kol_sentiment:.2f})，部分意见领袖表达了担忧"
        else:
            signal = "neutral"
            confidence = 0.5
            reasoning = f"KOL推文情绪中性(总分: {kol_sentiment:.2f})，意见领袖观点不一或保持中立"
            
        # 添加分析的推文数量
        reasoning += f"，分析了{len(tweets)}条推文"
        
        # 生成具体指标
        metrics = {
            "kol_sentiment_score": kol_sentiment,
            "analyzed_tweets_count": len(tweets),
            "top_influencers": [tweet['username'] for tweet in tweets[:3]]
        }
        
        return {
            "signal": signal,
            "confidence": confidence,
            "metrics": metrics,
            "reasoning": reasoning
        }
        
    def _analyze_tweets_with_llm(self, tweets):
        """使用LLM分析推文情绪"""
        # 如果没有推文，返回中性情绪
        if not tweets:
            return 0.0
            
        # 准备要分析的推文文本
        tweets_text = "\n\n".join([
            f"用户: {tweet['username']}\n"
            f"内容: {tweet['text']}\n"
            f"点赞: {tweet.get('likes', 0)}, 转发: {tweet.get('retweets', 0)}"
            for tweet in tweets[:5]  # 限制只分析前5条推文
        ])
        
        # 准备系统指令
        system_message = {
            "role": "system",
            "content": """你是加密货币市场情绪分析专家。你需要分析给定的加密货币意见领袖(KOL)推文，评估整体情绪，并给出一个介于-1到1之间的分数：
            - 1表示极其积极和看多
            - 0.5到0.9表示积极和乐观
            - 0.1到0.4表示轻微积极
            - 0表示中性
            - -0.1到-0.4表示轻微消极
            - -0.5到-0.9表示消极和担忧
            - -1表示极其消极和看空
            
            在分析时考虑：
            1. 推文内容的情绪和语气
            2. 对市场趋势的预测
            3. 对特定加密货币项目的评价
            4. 对行业事件和新闻的反应
            5. KOL影响力（根据点赞和转发数量）
            
            请确保分析：
            1. 侧重于加密货币相关内容
            2. 考虑推文发布时间的新近性
            3. 给予影响力更大的KOL更多权重"""
        }
        
        # 准备用户消息
        user_message = {
            "role": "user",
            "content": f"请分析以下加密货币KOL的推文，评估整体市场情绪：\n\n{tweets_text}\n\n请直接返回一个介于-1到1之间的数字，代表整体情绪得分。无需解释。"
        }
        
        try:
            # 获取LLM分析结果
            result = get_chat_completion([system_message, user_message])
            if result is None:
                print("Error: LLM API error occurred")
                return 0.0  # 返回中性情绪
                
            # 提取数字结果
            try:
                sentiment_score = float(result.strip())
                # 确保分数在-1到1之间
                sentiment_score = max(-1.0, min(1.0, sentiment_score))
                return sentiment_score
            except ValueError:
                print(f"Error parsing sentiment score: {result}")
                return 0.0  # 返回中性情绪
                
        except Exception as e:
            print(f"Error analyzing tweets with LLM: {e}")
            return 0.0  # 出错时返回中性情绪
        
    def _analyze_news_sentiment(self, sentiment_data):
        """分析新闻情绪"""
        news_sentiment = sentiment_data.get("news_sentiment")
        
        metrics = {
            "news_sentiment": news_sentiment
        }
        
        if news_sentiment is None:
            return {
                "signal": "neutral",
                "confidence": 0.5,
                "metrics": metrics,
                "reasoning": "缺少新闻情绪数据"
            }
            
        # 根据新闻情绪计算信号
        if news_sentiment > 0.6:
            signal = "bullish"
            confidence = min(0.5 + news_sentiment / 2, 0.9)
            reasoning = f"新闻情绪非常积极({news_sentiment:.2f})，主流媒体报道普遍乐观"
        elif news_sentiment > 0.2:
            signal = "bullish"
            confidence = 0.6
            reasoning = f"新闻情绪偏积极({news_sentiment:.2f})，媒体报道整体向好"
        elif news_sentiment < -0.6:
            signal = "bearish"
            confidence = min(0.5 + abs(news_sentiment) / 2, 0.9)
            reasoning = f"新闻情绪非常消极({news_sentiment:.2f})，主流媒体报道普遍负面"
        elif news_sentiment < -0.2:
            signal = "bearish"
            confidence = 0.6
            reasoning = f"新闻情绪偏消极({news_sentiment:.2f})，媒体报道整体负面"
        else:
            signal = "neutral"
            confidence = 0.5
            reasoning = f"新闻情绪中性({news_sentiment:.2f})，媒体报道褒贬不一或保持中立"
            
        return {
            "signal": signal,
            "confidence": confidence,
            "metrics": metrics,
            "reasoning": reasoning
        }
        
    def _calculate_overall_signal(self, signals):
        """计算总体情绪信号"""
        # 设置各个类别的权重
        weights = {
            "fear_greed": 0.30,  # 恐慌贪婪指数权重高
            "social_sentiment": 0.25,
            "kol_sentiment": 0.30,  # KOL情绪权重高
            "news_sentiment": 0.15
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
            reasoning = f"市场情绪整体积极，主要看多信号来自{', '.join(bullish_categories)}。"
            if bearish_categories:
                reasoning += f" 同时也有来自{', '.join(bearish_categories)}的看空信号，但整体仍偏向乐观。"
        elif signal == "bearish":
            reasoning = f"市场情绪整体消极，主要看空信号来自{', '.join(bearish_categories)}。"
            if bullish_categories:
                reasoning += f" 同时也有来自{', '.join(bullish_categories)}的看多信号，但整体仍偏向谨慎。"
        else:
            reasoning = "市场情绪相对平衡，积极和消极信号基本持平，建议保持中性立场。"
            
        return signal, confidence, reasoning


def sentiment_agent(state: AgentState):
    """情绪分析代理入口函数"""
    agent = SentimentAnalysisAgent()
    return agent.analyze(state)