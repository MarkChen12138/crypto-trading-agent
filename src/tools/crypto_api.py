from typing import Dict, Any, List, Optional
import pandas as pd
import ccxt
import numpy as np
from datetime import datetime, timedelta
import json
import time
import os
import logging
from dotenv import load_dotenv

# 设置日志记录
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger('crypto_api')

# 加载环境变量
load_dotenv()

# 初始化交易所API
def initialize_exchange(exchange_id='binance', test_mode=True):
    """初始化加密货币交易所连接
    
    Args:
        exchange_id: 交易所ID（默认binance）
        test_mode: 是否使用测试模式
        
    Returns:
        ccxt交易所实例
    """
    try:
        # 获取API密钥
        api_key = os.getenv(f'{exchange_id.upper()}_API_KEY')
        api_secret = os.getenv(f'{exchange_id.upper()}_API_SECRET')
        
        if not api_key or not api_secret:
            logger.warning(f"未找到{exchange_id}的API密钥，使用公共API")
            exchange = getattr(ccxt, exchange_id)()
        else:
            exchange_class = getattr(ccxt, exchange_id)
            exchange = exchange_class({
                'apiKey': api_key,
                'secret': api_secret,
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'spot'  # 默认现货市场
                }
            })
            
        # 设置测试模式（如果支持）
        if test_mode and hasattr(exchange, 'set_sandbox_mode'):
            exchange.set_sandbox_mode(True)
            logger.info(f"已启用{exchange_id}的测试模式")
            
        logger.info(f"成功连接到{exchange_id}交易所")
        return exchange
        
    except Exception as e:
        logger.error(f"连接到{exchange_id}交易所时出错: {e}")
        return None


def get_market_data(exchange, symbol: str) -> Dict[str, Any]:
    """获取市场数据
    
    Args:
        exchange: ccxt交易所实例
        symbol: 交易对，如'BTC/USDT'
        
    Returns:
        包含市场数据的字典
    """
    try:
        # 获取Ticker数据
        ticker = exchange.fetch_ticker(symbol)
        
        # 获取订单簿数据
        order_book = exchange.fetch_order_book(symbol)
        
        # 计算市场深度指标
        bid_ask_spread = ticker['ask'] - ticker['bid']
        spread_percentage = (bid_ask_spread / ticker['ask']) * 100
        
        # 计算买卖双方力量对比
        buy_volume = sum([order[1] for order in order_book['bids'][:10]])
        sell_volume = sum([order[1] for order in order_book['asks'][:10]])
        buy_sell_ratio = buy_volume / sell_volume if sell_volume > 0 else float('inf')
        
        # 构建返回数据
        market_data = {
            "price": ticker['last'],
            "volume_24h": ticker['quoteVolume'],
            "change_24h_percent": ticker['percentage'],
            "bid": ticker['bid'],
            "ask": ticker['ask'],
            "spread": bid_ask_spread,
            "spread_percent": spread_percentage,
            "buy_volume": buy_volume,
            "sell_volume": sell_volume,
            "buy_sell_ratio": buy_sell_ratio,
            "timestamp": ticker['timestamp']
        }
        
        return market_data
        
    except Exception as e:
        logger.error(f"获取{symbol}市场数据时出错: {e}")
        return {}
        

def get_price_history(exchange, symbol: str, timeframe: str = '1d', 
                      start_date: str = None, end_date: str = None,
                      limit: int = None) -> pd.DataFrame:
    """获取历史价格数据
    
    Args:
        exchange: ccxt交易所实例
        symbol: 交易对，如'BTC/USDT'
        timeframe: K线时间周期，如'1m', '1h', '1d'等
        start_date: 开始日期，格式'YYYY-MM-DD'
        end_date: 结束日期，格式'YYYY-MM-DD'
        limit: 获取的K线数量限制
        
    Returns:
        包含价格数据的DataFrame
    """
    try:
        # 处理日期参数
        if end_date:
            end_time = datetime.strptime(end_date, "%Y-%m-%d")
            # 确保end_date不会超过当前时间
            if end_time > datetime.now():
                end_time = datetime.now()
        else:
            end_time = datetime.now()
        
        end_timestamp = int(end_time.timestamp() * 1000)
        
        if start_date:
            start_time = datetime.strptime(start_date, "%Y-%m-%d")
            start_timestamp = int(start_time.timestamp() * 1000)
        else:
            # 默认获取过去90天的数据
            start_time = end_time - timedelta(days=90)
            start_timestamp = int(start_time.timestamp() * 1000)
        
        # 获取OHLCV数据
        if exchange.has['fetchOHLCV']:
            ohlcv = []
            
            if limit:
                # 如果指定了限制数量，直接获取
                ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            else:
                # 如果没有限制数量，使用时间范围分批获取
                now = end_timestamp
                all_ohlcv = []
                
                # 大多数交易所有每次请求的数量限制，通常是1000条
                max_request_limit = 1000
                
                while now >= start_timestamp:
                    partial_ohlcv = exchange.fetch_ohlcv(
                        symbol, timeframe, since=start_timestamp, 
                        limit=max_request_limit, params={"endTime": now}
                    )
                    
                    if len(partial_ohlcv) == 0:
                        break
                    
                    all_ohlcv = partial_ohlcv + all_ohlcv
                    now = partial_ohlcv[0][0] - 1  # 更新截止时间为最早K线的前一毫秒
                    
                    # 如果已经获取足够早的数据，退出循环
                    if partial_ohlcv[0][0] <= start_timestamp:
                        break
                    
                    # 防止请求过于频繁
                    time.sleep(exchange.rateLimit / 1000)  # 转换为秒
                
                ohlcv = all_ohlcv
            
            # 转换为DataFrame
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # 计算额外的技术指标
            # 1. 移动平均线
            df['ma5'] = df['close'].rolling(window=5).mean()
            df['ma10'] = df['close'].rolling(window=10).mean()
            df['ma20'] = df['close'].rolling(window=20).mean()
            df['ma50'] = df['close'].rolling(window=50).mean()
            df['ma200'] = df['close'].rolling(window=200).mean()
            
            # 2. 波动率
            df['daily_return'] = df['close'].pct_change()
            df['volatility_14d'] = df['daily_return'].rolling(window=14).std() * np.sqrt(365)
            
            # 3. RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['rsi_14'] = 100 - (100 / (1 + rs))
            
            # 4. MACD
            exp1 = df['close'].ewm(span=12, adjust=False).mean()
            exp2 = df['close'].ewm(span=26, adjust=False).mean()
            df['macd'] = exp1 - exp2
            df['signal_line'] = df['macd'].ewm(span=9, adjust=False).mean()
            df['macd_hist'] = df['macd'] - df['signal_line']
            
            # 5. 布林带
            df['bb_middle'] = df['close'].rolling(window=20).mean()
            bb_std = df['close'].rolling(window=20).std()
            df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
            df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
            
            # 6. 成交量指标
            df['volume_ma7'] = df['volume'].rolling(window=7).mean()
            df['volume_ma21'] = df['volume'].rolling(window=21).mean()
            
            # 7. 动量指标
            df['momentum_14'] = df['close'] / df['close'].shift(14) - 1
            
            # 重置索引
            df = df.reset_index()
            
            logger.info(f"成功获取{symbol} {timeframe}周期的历史价格数据，共{len(df)}条")
            return df
        else:
            logger.error(f"{exchange.id}不支持获取OHLCV数据")
            return pd.DataFrame()
            
    except Exception as e:
        logger.error(f"获取{symbol}历史价格数据时出错: {e}")
        return pd.DataFrame()


def get_onchain_metrics(asset: str) -> Dict[str, Any]:
    """获取链上指标数据（通过外部API）
    
    Args:
        asset: 资产代号，如'BTC'或'ETH'
        
    Returns:
        包含链上指标的字典
    """
    try:
        # 这里应该集成实际的链上数据API，如Glassnode、CryptoQuant等
        # 以下为模拟数据
        onchain_metrics = {
            "active_addresses_24h": 950000,
            "transaction_count_24h": 350000,
            "average_transaction_value": 25000,
            "hashrate": 305000000000000000000 if asset == "BTC" else None,
            "staking_rate": 65.3 if asset == "ETH" else None,
            "nvt_ratio": 27.5,
            "realized_price": 28500 if asset == "BTC" else 1850 if asset == "ETH" else 0,
            "mvrv_ratio": 1.23,
            "reserve_risk": 0.0073,
            "supply_last_active_1y_percent": 38.4,
            "puell_multiple": 1.05 if asset == "BTC" else None,
            "timestamp": datetime.now().timestamp()
        }
        
        logger.info(f"成功获取{asset}的链上指标数据")
        return onchain_metrics
        
    except Exception as e:
        logger.error(f"获取{asset}链上指标数据时出错: {e}")
        return {}


def execute_trade(exchange, symbol: str, order_type: str, side: str, 
                  amount: float = None, price: float = None) -> Dict[str, Any]:
    """执行交易
    
    Args:
        exchange: ccxt交易所实例
        symbol: 交易对，如'BTC/USDT'
        order_type: 订单类型，如'market', 'limit'
        side: 交易方向，'buy'或'sell'
        amount: 交易数量
        price: 限价单价格
        
    Returns:
        订单信息
    """
    try:
        # 验证交易所对象
        if not exchange or not hasattr(exchange, 'create_order'):
            raise ValueError("无效的交易所对象")
        
        # 验证参数
        if side not in ['buy', 'sell']:
            raise ValueError("交易方向必须是'buy'或'sell'")
            
        if order_type not in ['market', 'limit']:
            raise ValueError("订单类型必须是'market'或'limit'")
            
        if not amount or amount <= 0:
            raise ValueError("交易数量必须大于0")
            
        if order_type == 'limit' and (not price or price <= 0):
            raise ValueError("限价单必须指定有效价格")
            
        # 创建订单
        order = exchange.create_order(
            symbol=symbol,
            type=order_type,
            side=side,
            amount=amount,
            price=price if order_type == 'limit' else None
        )
        
        logger.info(f"成功创建{side} {order_type}订单，交易对: {symbol}，数量: {amount}")
        return order
        
    except Exception as e:
        logger.error(f"执行交易时出错: {e}")
        return {"error": str(e)}


def get_account_balance(exchange, quote_currency='USDT') -> Dict[str, Any]:
    """获取账户余额
    
    Args:
        exchange: ccxt交易所实例
        quote_currency: 计价货币（用于计算总资产价值）
        
    Returns:
        包含余额信息的字典
    """
    try:
        # 获取账户余额
        balance = exchange.fetch_balance()
        
        # 过滤有余额的资产
        non_zero_balances = {}
        for currency, data in balance['total'].items():
            if data > 0:
                non_zero_balances[currency] = {
                    'free': balance['free'].get(currency, 0),
                    'used': balance['used'].get(currency, 0),
                    'total': data
                }
        
        # 计算总资产价值（转换为USDT）
        total_value = 0
        for currency, data in non_zero_balances.items():
            if currency == quote_currency:
                total_value += data['total']
            else:
                try:
                    # 尝试获取该资产对USDT的价格
                    ticker = exchange.fetch_ticker(f"{currency}/{quote_currency}")
                    price = ticker['last']
                    value = data['total'] * price
                    total_value += value
                    non_zero_balances[currency]['value_in_usdt'] = value
                except:
                    # 如果无法获取价格，跳过
                    pass
        
        result = {
            'balances': non_zero_balances,
            'total_value_in_usdt': total_value,
            'timestamp': datetime.now().timestamp()
        }
        
        logger.info(f"成功获取账户余额，总价值: {total_value} {quote_currency}")
        return result
        
    except Exception as e:
        logger.error(f"获取账户余额时出错: {e}")
        return {}


def get_market_sentiment(asset: str) -> Dict[str, Any]:
    """获取市场情绪指标
    
    Args:
        asset: 资产代号，如'BTC'或'ETH'
        
    Returns:
        包含市场情绪指标的字典
    """
    try:
        # 这里应该集成实际的市场情绪API，如Fear & Greed Index
        # 以下为模拟数据
        sentiment_data = {
            "fear_greed_index": 65,  # 0-100，0为极度恐惧，100为极度贪婪
            "fear_greed_classification": "Greed",
            "social_sentiment": 0.62,  # -1到1，-1为极度负面，1为极度正面
            "social_volume": 125000,
            "google_trends_value": 78,
            "twitter_sentiment": 0.58,
            "reddit_sentiment": 0.65,
            "timestamp": datetime.now().timestamp()
        }
        
        logger.info(f"成功获取{asset}的市场情绪数据")
        return sentiment_data
        
    except Exception as e:
        logger.error(f"获取{asset}市场情绪数据时出错: {e}")
        return {}