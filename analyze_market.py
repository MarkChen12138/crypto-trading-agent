import os
import sys
import json
import logging
from datetime import datetime, timedelta
import pandas as pd
from src.agents.market_data import MarketDataAgent
from src.agents.state import AgentState

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('market_analyzer')

def analyze_market(symbol="BTC/USDT", days=90):
    """运行市场数据分析
    
    Args:
        symbol: 交易对，例如 "BTC/USDT"
        days: 历史数据天数
    """
    logger.info(f"开始{symbol}的市场分析...")
    
    # 初始化状态
    state = AgentState(
        messages=[],
        data={
            "symbol": symbol,
            "exchange_id": "binance",
            "start_date": (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d'),
            "end_date": datetime.now().strftime('%Y-%m-%d')
        }
    )
    
    # 运行市场数据代理
    agent = MarketDataAgent()
    result = agent.collect_data(state)
    
    # 提取结果
    market_data = result["data"]["market_data"]
    technical_indicators = result["data"]["technical_indicators"]
    technical_signals = result["data"]["technical_signals"]
    sentiment_distribution = result["data"]["sentiment_distribution"]
    news_data = result["data"]["news_data"]
    
    # 打印技术分析结果
    print("\n===== 技术分析结果 =====")
    print(f"当前价格: {market_data.get('price', 'N/A')} USD")
    print(f"24小时涨跌幅: {market_data.get('change_24h_percent', 'N/A')}%")
    print(f"24小时交易量: {market_data.get('volume_24h', 'N/A')}")
    
    if technical_indicators:
        print("\n主要技术指标:")
        print(f"MA5/MA20: {technical_indicators.get('ma5', 'N/A')}/{technical_indicators.get('ma20', 'N/A')}")
        print(f"RSI(14): {technical_indicators.get('rsi', 'N/A')}")
        print(f"MACD: {technical_indicators.get('macd', 'N/A')}")
        print(f"波动率: {technical_indicators.get('volatility', 'N/A')*100:.2f}%" if technical_indicators.get('volatility') else "波动率: N/A")
        print(f"布林带(上/中/下): {technical_indicators.get('bb_upper', 'N/A')}/{technical_indicators.get('bb_middle', 'N/A')}/{technical_indicators.get('bb_lower', 'N/A')}")
    
    if technical_signals:
        print("\n交易信号:")
        print(f"MA5穿越MA20: {'上穿' if technical_signals.get('ma5_cross_ma20') else '下穿'}")
        print(f"RSI状态: {'超卖' if technical_signals.get('rsi_oversold') else '超买' if technical_signals.get('rsi_overbought') else '正常'}")
        print(f"MACD信号: {'金叉' if technical_signals.get('macd_cross') else '死叉'}")
        print(f"价格相对MA20: {'上方' if technical_signals.get('price_above_ma20') else '下方'}")
        print(f"布林带位置: {'中轨' if technical_signals.get('in_bollinger') else '偏离'}")
    
    # 打印情感分析结果
    print("\n===== 基于RapidAPI新闻的Gemini情感分析 =====")
    if sentiment_distribution:
        print(f"积极新闻占比: {sentiment_distribution.get('positive_percent', 'N/A'):.1f}%")
        print(f"消极新闻占比: {sentiment_distribution.get('negative_percent', 'N/A'):.1f}%")
        print(f"中性新闻占比: {sentiment_distribution.get('neutral_percent', 'N/A'):.1f}%")
        print(f"整体市场情绪: {sentiment_distribution.get('overall', 'N/A')}")
        
        # 显示关于正负比例的额外信息
        details = sentiment_distribution.get('details', {})
        if details:
            print(f"\n新闻情感详情:")
            print(f"积极/消极比例: {details.get('sentiment_ratio', 'N/A'):.2f}")
            print(f"总新闻数量: {details.get('total_news', 'N/A')}")
    
    # 打印部分新闻标题和情感
    if news_data and 'news' in news_data and news_data['news']:
        print("\n最新新闻情感分析示例:")
        for i, article in enumerate(news_data['news'][:5]):  # 只显示前5条
            print(f"\n{i+1}. 标题: {article.get('title', 'N/A')}")
            print(f"   情感: {article.get('sentiment', 'N/A')}")
    
    # 返回结果数据
    return result["data"]

if __name__ == "__main__":
    # 从命令行参数获取交易对
    symbol = sys.argv[1] if len(sys.argv) > 1 else "BTC/USDT"
    
    # 运行分析
    analyze_market(symbol) 