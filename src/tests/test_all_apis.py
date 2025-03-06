import unittest
import pandas as pd
from src.tools.crypto_api import (
    initialize_exchange,
    get_market_data,
    get_crypto_news,
    get_price_history
)

class TestTradingAPIs(unittest.TestCase):
    def setUp(self):
        """测试前的准备工作"""
        self.exchange = initialize_exchange('binance', test_mode=True)
        self.test_symbol = 'BTC/USDT'
        self.test_asset = 'BTC'

    def test_binance_market_data(self):
        """测试Binance市场数据获取"""
        # 测试交易所连接
        self.assertIsNotNone(self.exchange)
        
        # 测试市场数据获取
        market_data = get_market_data(self.exchange, self.test_symbol)
        self.assertIsInstance(market_data, dict)
        
        # 验证市场数据字段
        required_fields = [
            'price', 'volume_24h', 'change_24h_percent', 
            'bid', 'ask', 'high', 'low', 'timestamp'
        ]
        for field in required_fields:
            self.assertIn(field, market_data)
            
        # 打印市场数据
        print("\nBinance市场数据:")
        print(f"当前价格: {market_data['price']}")
        print(f"24小时交易量: {market_data['volume_24h']}")
        print(f"24小时涨跌幅: {market_data['change_24h_percent']}%")
        print(f"最高/最低: {market_data['high']}/{market_data['low']}")
        print(f"买入/卖出: {market_data['bid']}/{market_data['ask']}")

    def test_news_sentiment(self):
        """测试新闻数据和情感分析"""
        news_data = get_crypto_news(self.test_asset)
        
        # 验证返回数据结构
        self.assertIsInstance(news_data, dict)
        self.assertIn('news', news_data)
        self.assertIn('timestamp', news_data)
        
        # 验证新闻数据和情感分析
        if news_data['news']:
            print("\n新闻情感分析:")
            sentiments = {'positive': 0, 'negative': 0, 'neutral': 0}
            
            for article in news_data['news']:
                self.assertIn('title', article)
                self.assertIn('description', article)
                self.assertIn('sentiment', article)
                self.assertIn(article['sentiment'], ['positive', 'negative', 'neutral'])
                
                sentiments[article['sentiment']] += 1
                print(f"\n标题: {article['title']}")
                print(f"情感: {article['sentiment']}")
            
            # 打印情感统计
            total = len(news_data['news'])
            print("\n情感分析统计:")
            print(f"积极: {sentiments['positive']/total*100:.1f}%")
            print(f"消极: {sentiments['negative']/total*100:.1f}%")
            print(f"中性: {sentiments['neutral']/total*100:.1f}%")

if __name__ == '__main__':
    unittest.main() 