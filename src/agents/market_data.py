from langchain_core.messages import HumanMessage
from src.agents.state import AgentState, show_agent_reasoning, show_workflow_status
from src.tools.crypto_api import initialize_exchange, get_market_data, get_price_history, get_onchain_metrics, get_market_sentiment
from datetime import datetime, timedelta
import pandas as pd
import logging

# 设置日志记录
logger = logging.getLogger('market_data_agent')

class MarketDataAgent:
    """
    市场数据代理 - 负责收集和预处理各种市场数据
    
    收集的数据包括:
    1. 交易所K线数据
    2. 链上数据
    3. 市场情绪数据
    4. 社交媒体数据
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
            # 默认获取90天的数据
            start_date = (end_date_obj - timedelta(days=90)).strftime('%Y-%m-%d')
        else:
            start_date = data["start_date"]
            
        # 初始化交易所连接
        self.exchange = initialize_exchange(exchange_id, test_mode=True)
        if not self.exchange:
            logger.warning(f"无法连接到交易所{exchange_id}，将使用模拟数据")
            # 这里可以添加生成模拟数据的逻辑
            
        # 1. 获取市场数据
        try:
            market_data = get_market_data(self.exchange, symbol)
            logger.info(f"成功获取{symbol}的市场数据")
        except Exception as e:
            logger.error(f"获取市场数据时出错：{e}")
            market_data = {}
            
        # 2. 获取价格历史数据
        try:
            prices_df = get_price_history(self.exchange, symbol, start_date=start_date, end_date=end_date)
            prices_data = prices_df.to_dict('records') if not prices_df.empty else []
            logger.info(f"成功获取{symbol}的价格历史数据，共{len(prices_data)}条记录")
        except Exception as e:
            logger.error(f"获取价格历史数据时出错：{e}")
            prices_data = []
            
        # 3. 获取链上指标数据
        try:
            onchain_metrics = get_onchain_metrics(asset)
            logger.info(f"成功获取{asset}的链上指标数据")
        except Exception as e:
            logger.error(f"获取链上指标数据时出错：{e}")
            onchain_metrics = {}
            
        # 4. 获取市场情绪数据
        try:
            market_sentiment = get_market_sentiment(asset)
            logger.info(f"成功获取{asset}的市场情绪数据")
        except Exception as e:
            logger.error(f"获取市场情绪数据时出错：{e}")
            market_sentiment = {}
            
        # 将所有数据整合到state中
        updated_data = {
            **data,
            "symbol": symbol,
            "asset": asset,
            "exchange_id": exchange_id,
            "market_data": market_data,
            "prices": prices_data,
            "start_date": start_date,
            "end_date": end_date,
            "onchain_metrics": onchain_metrics,
            "market_sentiment": market_sentiment
        }
        
        # 创建消息（通常市场数据代理不需要创建消息，只收集数据）
        show_workflow_status("市场数据代理", "completed")
        return {
            "messages": state["messages"],
            "data": updated_data
        }


def market_data_agent(state: AgentState):
    """市场数据代理入口函数"""
    agent = MarketDataAgent()
    return agent.collect_data(state)