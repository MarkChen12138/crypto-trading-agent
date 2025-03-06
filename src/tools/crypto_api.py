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
import requests
import google.generativeai as genai

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


def get_market_sentiment(asset: str) -> Dict[str, Any]:
    """获取市场情绪数据
    
    Args:
        asset: 资产代号，如'BTC'或'ETH'
        
    Returns:
        包含市场情绪数据的字典
    """
    try:
        # 获取新闻数据
        news_data = get_crypto_news(asset)
        
        # 获取CoinMarketCap数据
        cmc_data = get_coinmarketcap_data(asset)
        
        # 计算情绪得分
        sentiment_score = 0
        if news_data.get("sentiment_stats"):
            stats = news_data["sentiment_stats"]
            total = stats["positive"] + stats["negative"] + stats["neutral"]
            if total > 0:
                sentiment_score = (stats["positive"] - stats["negative"]) / total
        
        return {
            "news_data": news_data,
            "market_data": cmc_data,
            "sentiment_score": sentiment_score,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"获取{asset}市场情绪数据时出错: {e}")
        return {}

def get_onchain_metrics(asset: str) -> Dict[str, Any]:
    """获取链上指标数据
    
    Args:
        asset: 资产代号，如'BTC'或'ETH'
        
    Returns:
        包含链上指标的字典
    """
    try:
        # 1. 获取DEX交易对列表
        url = "https://pro-api.coinmarketcap.com/v4/dex/spot-pairs/latest"
        params = {
            "base_asset_symbol": asset.lower(),
            "limit": 1,  # 只获取流动性最高的交易对
            "sort": "liquidity",
            "sort_dir": "desc",
            "aux": "num_transactions_24h,holders,24h_no_of_buys,24h_no_of_sells,24h_buy_volume,24h_sell_volume"
        }
        headers = {
            "X-CMC_PRO_API_KEY": os.getenv('COINMARKETCAP_API_KEY'),
            "Accept": "application/json"
        }
        
        response = requests.get(url, params=params, headers=headers)
        if response.status_code != 200:
            logger.error(f"获取{asset} DEX交易对列表失败: HTTP {response.status_code}")
            logger.error(f"错误响应: {response.text}")
            return {}
            
        pairs_data = response.json()
        if not pairs_data.get("data"):
            logger.error(f"未找到{asset}的DEX交易对")
            return {}
            
        # 获取第一个交易对的合约地址
        pair = pairs_data["data"][0]
        contract_address = pair.get("contract_address")
        network_slug = pair.get("network_slug")
        
        if not contract_address or not network_slug:
            logger.error(f"交易对缺少必要信息: {pair}")
            return {}
            
        # 2. 获取历史OHLCV数据
        ohlcv_url = "https://pro-api.coinmarketcap.com/v4/dex/pairs/ohlcv/historical"
        ohlcv_params = {
            "contract_address": contract_address,
            "network_slug": network_slug,
            "time_period": "daily",
            "count": 30,  # 获取30天的数据
            "aux": "num_transactions_24h,holders,24h_no_of_buys,24h_no_of_sells,24h_buy_volume,24h_sell_volume"
        }
        
        ohlcv_response = requests.get(ohlcv_url, params=ohlcv_params, headers=headers)
        if ohlcv_response.status_code != 200:
            logger.error(f"获取{asset}历史OHLCV数据失败: HTTP {ohlcv_response.status_code}")
            logger.error(f"错误响应: {ohlcv_response.text}")
            return {}
            
        ohlcv_data = ohlcv_response.json()
        
        # 3. 获取最新交易数据
        trades_url = "https://pro-api.coinmarketcap.com/v4/dex/pairs/trade/latest"
        trades_params = {
            "contract_address": contract_address,
            "network_slug": network_slug,
            "aux": "transaction_hash,blockchain_explorer_link"
        }
        
        trades_response = requests.get(trades_url, params=trades_params, headers=headers)
        if trades_response.status_code != 200:
            logger.error(f"获取{asset}最新交易数据失败: HTTP {trades_response.status_code}")
            logger.error(f"错误响应: {trades_response.text}")
            return {}
            
        trades_data = trades_response.json()
        
        # 整合数据
        metrics = {
            "pair_info": pair,
            "historical_data": ohlcv_data.get("data", []),
            "recent_trades": trades_data.get("data", []),
            "timestamp": datetime.now().isoformat()
        }
        
        return metrics
        
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


def get_crypto_news(asset: str, limit: int = 10) -> Dict:
    """
    获取加密货币相关新闻
    
    Args:
        asset: 加密货币符号
        limit: 返回的新闻数量
        
    Returns:
        Dict: 包含新闻和情感分析结果的字典
    """
    try:
        url = "https://crypto-news51.p.rapidapi.com/api/v1/crypto/articles/search"
        params = {
            "title_keywords": asset,
            "page": 1,
            "limit": limit,
            "time_frame": "24h",
            "format": "json"
        }
        
        # 从环境变量获取RapidAPI密钥
        rapidapi_key = os.getenv('RAPIDAPI_KEY')
        if not rapidapi_key:
            logger.error("未找到RAPIDAPI_KEY环境变量，无法获取加密货币新闻")
            return {"news": [], "error": "未找到RAPIDAPI_KEY环境变量", "timestamp": datetime.now().isoformat()}
            
        headers = {
            "x-rapidapi-host": "crypto-news51.p.rapidapi.com",
            "x-rapidapi-key": rapidapi_key,
            "Accept": "application/json"
        }
        
        logger.info(f"正在获取{asset}相关新闻，参数：{params}")
        response = requests.get(url, params=params, headers=headers)
        
        # 记录API请求结果
        logger.info(f"API状态码: {response.status_code}")
        if response.status_code != 200:
            logger.error(f"API错误: {response.text}")
            # 返回结构化空结果
            return {"news": [], "error": response.text, "timestamp": datetime.now().isoformat()}
            
        response.raise_for_status()
        data = response.json()
        
        # 记录返回数据结构以便调试
        if isinstance(data, list):
            logger.info(f"API返回列表数据，包含{len(data)}条新闻")
        elif isinstance(data, dict):
            logger.info(f"API返回字典数据: {list(data.keys())}")
        else:
            logger.warning(f"API返回未知格式数据: {type(data)}")
        
        # 使用Gemini进行情感分析
        # 从环境变量获取Gemini API密钥
        gemini_key = os.getenv('GEMINI_API_KEY')
        if not gemini_key:
            logger.error("未找到GEMINI_API_KEY环境变量，无法进行情感分析")
            return {"news": [], "error": "未找到GEMINI_API_KEY环境变量", "timestamp": datetime.now().isoformat()}
            
        genai.configure(api_key=gemini_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        news_with_sentiment = []
        
        # 确保data是列表格式
        articles = data if isinstance(data, list) else []
        
        for article in articles:
            # 准备用于情感分析的文本
            title = article.get('title', '')
            summary = article.get('summary', '')
            
            if not title and not summary:
                logger.warning(f"跳过无内容的文章: {article}")
                continue
            
            text_for_analysis = f"标题: {title}\n描述: {summary}"
            
            # 使用Gemini进行情感分析
            prompt = f"""
            请分析以下加密货币新闻的情感倾向。
            
            重要说明：
            1. 只有在新闻中明确包含积极信息（如价格上涨、突破新高、市场信心增强等）时，才返回"positive"
            2. 只有在新闻中明确包含消极信息（如价格下跌、市场恐慌、技术故障等）时，才返回"negative"
            3. 如果新闻主要是客观事实陈述，没有明显的积极或消极倾向，必须返回"neutral"
            
            示例：
            标题：比特币价格突破新高，市场信心增强
            描述：比特币价格突破50000美元，交易量显著增加
            结果：positive（因为包含明确的积极信息：价格突破新高、交易量增加）
            
            标题：市场遭遇重挫，投资者恐慌性抛售
            描述：由于监管政策收紧，市场出现大幅下跌
            结果：negative（因为包含明确的消极信息：市场重挫、恐慌性抛售）
            
            标题：系统升级完成
            描述：系统成功完成升级，运行正常
            结果：neutral（因为只是客观陈述事实，没有明显的积极或消极倾向）
            
            请仔细分析以下新闻并只返回一个词（positive、negative或neutral）：
            {text_for_analysis}
            """
            
            try:
                response = model.generate_content(prompt)
                sentiment = response.text.strip().lower()
                logger.info(f"新闻情感分析结果: {sentiment}, 标题: {title[:50]}...")
            except Exception as e:
                logger.error(f"Gemini情感分析失败: {str(e)}")
                sentiment = "neutral"
            
            article["sentiment"] = sentiment
            # 为了保持与测试兼容，添加description字段
            article["description"] = article.get("summary", "")
            news_with_sentiment.append(article)
        
        result = {
            "news": news_with_sentiment,
            "timestamp": datetime.now().isoformat()
        }
        logger.info(f"完成{asset}相关新闻的获取和情感分析，共{len(news_with_sentiment)}条")
        return result
        
    except requests.exceptions.RequestException as e:
        logger.error(f"获取加密货币新闻失败: {str(e)}")
        return {"news": [], "error": str(e), "timestamp": datetime.now().isoformat()}

def get_supported_dex_networks() -> List[Dict[str, Any]]:
    """获取CoinMarketCap支持的DEX网络列表
    
    Returns:
        包含网络信息的列表
    """
    try:
        url = "https://pro-api.coinmarketcap.com/v4/dex/networks"
        headers = {
            "X-CMC_PRO_API_KEY": os.getenv('COINMARKETCAP_API_KEY'),
            "Accept": "application/json"
        }
        
        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            logger.error(f"获取DEX网络列表失败: HTTP {response.status_code}")
            logger.error(f"错误响应: {response.text}")
            return []
            
        data = response.json()
        return data.get("data", [])
        
    except Exception as e:
        logger.error(f"获取DEX网络列表时出错: {e}")
        return []

def get_coinmarketcap_data(asset: str, network_slug: str = "ethereum") -> Dict[str, Any]:
    """获取CoinMarketCap DEX市场数据
    
    Args:
        asset: 资产代号或合约地址
        network_slug: 网络标识符，如'ethereum', 'bsc', 'polygon'等
        
    Returns:
        包含市场数据的字典
    """
    try:
        # 使用/v4/dex/listings/quotes端点
        url = "https://pro-api.coinmarketcap.com/v4/dex/listings/quotes"
        params = {
            "start": 1,
            "limit": 50,
            "sort": "volume_24h",
            "sort_dir": "desc",
            "aux": "num_market_pairs,last_updated,market_share"
        }
        headers = {
            "X-CMC_PRO_API_KEY": os.getenv('COINMARKETCAP_API_KEY'),
            "Accept": "application/json"
        }
        
        response = requests.get(url, params=params, headers=headers)
        if response.status_code != 200:
            logger.error(f"获取DEX列表数据失败: {response.status_code}")
            logger.error(f"响应内容: {response.text}")
            return {}
            
        data = response.json()
        
        # 整合数据
        market_data = {
            "data": data.get("data", {}),
            "status": data.get("status", {}),
            "timestamp": datetime.now().isoformat()
        }
        
        return market_data
        
    except Exception as e:
        logger.error(f"获取CoinMarketCap DEX数据时出错: {e}")
        return {}

def get_geckoterminal_data(asset: str) -> Dict[str, Any]:
    """获取GeckoTerminal链上交易数据
    
    Args:
        asset: 资产代号，如'BTC'或'ETH'
        
    Returns:
        包含链上交易数据的字典
    """
    try:
        # 1. 搜索资产对应的主要交易池
        search_url = f"https://api.geckoterminal.com/api/v2/search/pools?query={asset}"
        search_response = requests.get(search_url)
        
        if search_response.status_code != 200:
            logger.error(f"搜索{asset}交易池失败: HTTP {search_response.status_code}")
            return {}
            
        pools_data = search_response.json()
        if not pools_data.get("data"):
            logger.error(f"未找到{asset}的交易池")
            return {}
            
        # 获取第一个交易池的详细信息
        pool_id = pools_data["data"][0]["id"]
        network, pool_address = pool_id.split("_", 1)
        
        # 2. 获取池的详细数据
        pool_url = f"https://api.geckoterminal.com/api/v2/networks/{network}/pools/{pool_address}"
        pool_response = requests.get(pool_url, headers={"accept": "application/json"})
        
        if pool_response.status_code != 200:
            logger.error(f"获取{asset}池数据失败: HTTP {pool_response.status_code}")
            return {}
            
        pool_data = pool_response.json()
        attrs = pool_data["data"]["attributes"]
        
        # 3. 获取交易明细
        trades_url = f"https://api.geckoterminal.com/api/v2/networks/{network}/pools/{pool_address}/trades"
        trades_response = requests.get(trades_url, headers={"accept": "application/json"})
        
        trades_data = []
        if trades_response.status_code == 200:
            trades_data = trades_response.json().get("data", [])
        
        return {
            "price_usd": attrs["base_token_price_usd"],
            "volume_24h": attrs["volume_usd"]["h24"],
            "liquidity": attrs.get("liquidity", {}),
            "transactions_24h": {
                "buys": attrs["transactions"]["h24"]["buys"],
                "sells": attrs["transactions"]["h24"]["sells"],
                "buyers": attrs["transactions"]["h24"]["buyers"],
                "sellers": attrs["transactions"]["h24"]["sellers"]
            },
            "recent_trades": trades_data[:10]  # 只返回最近10笔交易
        }
        
    except Exception as e:
        logger.error(f"获取{asset} GeckoTerminal数据时出错: {e}")
        return {}

def get_latest_trades(asset: str, network_slug: str = "ethereum", limit: int = 100) -> Dict:
    """
    获取指定资产的最新交易数据
    
    Args:
        asset: 资产符号（如ETH）
        network_slug: 网络标识符（默认为ethereum）
        limit: 返回的交易数量限制
        
    Returns:
        Dict: 包含最新交易数据的字典
    """
    try:
        # 1. 首先获取资产的合约地址
        contract_url = "https://pro-api.coinmarketcap.com/v4/dex/spot-pairs/latest"
        params = {
            "base_asset_symbol": asset,
            "network_slug": network_slug,
            "limit": 1,
            "sort": "volume_24h",
            "sort_dir": "desc",
            "aux": "contract_address,network_slug"
        }
        headers = {
            "X-CMC_PRO_API_KEY": os.getenv("COINMARKETCAP_API_KEY"),
            "Accept": "application/json"
        }
        
        response = requests.get(contract_url, params=params, headers=headers)
        if response.status_code != 200:
            logger.error(f"获取合约地址失败: {response.status_code}")
            logger.error(f"响应内容: {response.text}")
            return {}
            
        data = response.json()
        if not data.get("data"):
            logger.error(f"未找到{asset}的DEX交易对")
            return {}
            
        pair_data = data["data"][0]
        contract_address = pair_data.get("contract_address")
        
        if not contract_address:
            logger.error(f"未找到{asset}的合约地址")
            return {}
            
        # 2. 获取最新交易数据
        trades_url = "https://pro-api.coinmarketcap.com/v4/dex/pairs/trade/latest"
        params = {
            "contract_address": contract_address,
            "network_slug": network_slug,
            "limit": limit
        }
        
        response = requests.get(trades_url, params=params, headers=headers)
        if response.status_code != 200:
            logger.error(f"获取交易数据失败: {response.status_code}")
            logger.error(f"响应内容: {response.text}")
            return {}
            
        data = response.json()
        if not data.get("data"):
            logger.error(f"未找到{asset}的交易数据")
            return {}
            
        trades = data["data"]
        
        # 3. 计算交易统计信息
        total_volume = sum(float(trade.get("volume", 0)) for trade in trades)
        total_trades = len(trades)
        buy_volume = sum(float(trade.get("volume", 0)) for trade in trades if trade.get("side") == "buy")
        sell_volume = sum(float(trade.get("volume", 0)) for trade in trades if trade.get("side") == "sell")
        avg_price = sum(float(trade.get("price", 0)) for trade in trades) / total_trades if total_trades > 0 else 0
        
        return {
            "trades": trades,
            "stats": {
                "total_trades": total_trades,
                "total_volume": total_volume,
                "buy_volume": buy_volume,
                "sell_volume": sell_volume,
                "avg_price": avg_price
            },
            "pair_info": {
                "contract_address": contract_address,
                "network_slug": network_slug,
                "base_asset": asset
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"获取最新交易数据时发生错误: {str(e)}")
        return {}