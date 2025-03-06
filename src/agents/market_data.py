from langchain_core.messages import HumanMessage
from src.agents.state import AgentState, show_agent_reasoning, show_workflow_status
from src.tools.crypto_api import initialize_exchange, get_market_data, get_price_history, get_crypto_news
from datetime import datetime, timedelta
import pandas as pd
import logging

# 设置日志记录
logger = logging.getLogger('market_data_agent')

class MarketDataAgent:
    """
    市场数据代理 - 负责收集和初步处理加密货币市场数据
    
    收集的数据包括:
    1. Binance交易所实时价格和交易量数据
    2. 价格历史数据
    3. 加密货币新闻数据
    """
    
    def __init__(self):
        """初始化市场数据代理"""
        self.exchange = None
        
    def collect_data(self, state: AgentState):
        """收集市场数据"""
        show_workflow_status("市场数据代理")
        data = state["data"]
        
        # 获取设置
        symbol = data.get("symbol", "BTC/USDT")  # 默认交易对
        exchange_id = data.get("exchange_id", "binance")  # 默认交易所
        asset = symbol.split('/')[0]  # 资产代号，如BTC
        
        # 记录基本信息
        logger.info(f"正在收集{symbol}的市场数据，交易所：{exchange_id}")
        
        # 设置默认日期
        current_date = datetime.now()
        yesterday = current_date - timedelta(days=1)
        end_date = data.get("end_date") or yesterday.strftime('%Y-%m-%d')
        
        # 确保end_date不在未来
        end_date_obj = datetime.strptime(end_date, '%Y-%m-%d')
        if end_date_obj > yesterday:
            end_date = yesterday.strftime('%Y-%m-%d')
            end_date_obj = yesterday
            
        if not data.get("start_date"):
            # 默认获取90天的数据用于技术指标计算
            start_date = (end_date_obj - timedelta(days=90)).strftime('%Y-%m-%d')
        else:
            start_date = data["start_date"]
            
        # 初始化交易所连接
        self.exchange = initialize_exchange(exchange_id, test_mode=True)
        if not self.exchange:
            logger.warning(f"无法连接到交易所{exchange_id}，将使用模拟数据")
            
        # 1. 获取Binance实时市场数据
        try:
            # 获取交易所数据
            market_data = get_market_data(self.exchange, symbol)
            logger.info(f"成功获取{symbol}的实时市场数据")
        except Exception as e:
            logger.error(f"获取市场数据时出错：{e}")
            market_data = {}
            
        # 2. 获取价格历史数据
        try:
            # 获取价格历史数据
            prices_df = get_price_history(self.exchange, symbol, timeframe='1d', 
                                         start_date=start_date, end_date=end_date)
            prices_data = prices_df.to_dict('records') if not prices_df.empty else []
            logger.info(f"成功获取{symbol}的历史价格数据，共{len(prices_data)}条记录")
        except Exception as e:
            logger.error(f"获取价格历史数据时出错：{e}")
            prices_df = pd.DataFrame()
            prices_data = []
            
        # 3. 获取新闻数据
        try:
            # 获取加密货币新闻
            news_data = get_crypto_news(asset, limit=10)
            
            # 检查新闻数据格式
            if isinstance(news_data, dict) and 'news' in news_data:
                logger.info(f"成功获取{asset}的新闻数据，共{len(news_data['news'])}条")
            else:
                logger.warning(f"新闻数据格式异常: {type(news_data)}")
                # 尝试修复结构
                if isinstance(news_data, list):
                    news_data = {"news": news_data}
                    logger.info(f"修复新闻数据结构，共{len(news_data['news'])}条")
                else:
                    news_data = {"news": []}
                    logger.warning("无法修复新闻数据结构，使用空列表")
        except Exception as e:
            logger.error(f"获取新闻数据时出错：{e}")
            news_data = {"news": []}
            
        # 将所有数据整合到state中
        updated_data = {
            **data,
            "symbol": symbol,
            "asset": asset,
            "exchange_id": exchange_id,
            "market_data": market_data,
            "prices_df": prices_df,  # 保存DataFrame供技术分析使用
            "prices": prices_data,
            "start_date": start_date,
            "end_date": end_date,
            "news_data": news_data
        }
        
        # 创建消息
        show_agent_reasoning(f"已完成{asset}的市场数据收集", "市场数据代理")
        
        # 添加更多日志信息帮助调试数据传递
        logger.info(f"市场数据收集完成，包含：")
        logger.info(f"  - 价格数据: {len(prices_data)}条记录")
        logger.info(f"  - 新闻数据: {len(news_data.get('news', []))}条记录")
        
        message = HumanMessage(content=f"已完成{symbol}的市场数据收集，准备进行技术分析和情感分析。")
        
        messages = state["messages"] + [message]
        show_workflow_status("市场数据代理", "completed")
        
        return {
            "messages": messages,
            "data": updated_data
        }


def market_data_agent(state: AgentState):
    """市场数据代理入口函数"""
    agent = MarketDataAgent()
    return agent.collect_data(state)