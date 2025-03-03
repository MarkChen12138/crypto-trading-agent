import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import ccxt
import os
import json
import time
from src.tools.crypto_api import initialize_exchange, get_price_history

class CryptoBacktester:
    """
    加密货币回测系统 - 用于测试和评估交易策略的历史表现
    
    功能:
    1. 获取历史价格数据
    2. 模拟交易执行
    3. 计算回测绩效指标
    4. 生成回测报告和可视化
    """
    
    def __init__(self, initial_capital=10000, fee_rate=0.001):
        """
        初始化回测系统
        
        Args:
            initial_capital: 初始资金（USD）
            fee_rate: 交易手续费率（默认0.1%）
        """
        self.initial_capital = initial_capital
        self.fee_rate = fee_rate
        self.portfolio = {"cash": initial_capital, "assets": {}}
        self.trade_history = []
        self.portfolio_history = []
        self.price_data = None
        self.exchange = None
        
    def load_price_data(self, symbol, start_date, end_date=None, timeframe='1d', source='file', filename=None):
        """
        加载历史价格数据
        
        Args:
            symbol: 交易对，例如 'BTC/USDT'
            start_date: 开始日期，格式 'YYYY-MM-DD'
            end_date: 结束日期，格式 'YYYY-MM-DD'
            timeframe: K线时间周期，例如 '1d', '4h', '1h'
            source: 数据来源，'exchange' 或 'file'
            filename: 如果 source='file'，指定数据文件路径
            
        Returns:
            bool: 加载成功返回 True，否则 False
        """
        try:
            if source == 'exchange':
                # 从交易所获取数据
                self.exchange = initialize_exchange()
                if not self.exchange:
                    print("无法连接到交易所，请检查API设置")
                    return False
                    
                self.price_data = get_price_history(
                    self.exchange, 
                    symbol, 
                    start_date=start_date, 
                    end_date=end_date,
                    timeframe=timeframe
                )
                
                if self.price_data.empty:
                    print(f"无法获取{symbol}的历史数据")
                    return False
                    
            elif source == 'file':
                # 从文件加载数据
                if not filename:
                    print("需要指定数据文件路径")
                    return False
                    
                file_ext = os.path.splitext(filename)[1].lower()
                
                if file_ext == '.csv':
                    self.price_data = pd.read_csv(filename)
                elif file_ext in ['.xls', '.xlsx']:
                    self.price_data = pd.read_excel(filename)
                else:
                    print(f"不支持的文件格式: {file_ext}")
                    return False
                    
                # 确保时间列为日期类型
                if 'timestamp' in self.price_data.columns:
                    self.price_data['timestamp'] = pd.to_datetime(self.price_data['timestamp'])
                    
                # 筛选日期范围
                if 'timestamp' in self.price_data.columns:
                    start = pd.to_datetime(start_date)
                    if end_date:
                        end = pd.to_datetime(end_date)
                        self.price_data = self.price_data[
                            (self.price_data['timestamp'] >= start) & 
                            (self.price_data['timestamp'] <= end)
                        ]
                    else:
                        self.price_data = self.price_data[self.price_data['timestamp'] >= start]
            else:
                print(f"不支持的数据源: {source}")
                return False
                
            # 确保数据按时间排序
            if 'timestamp' in self.price_data.columns:
                self.price_data = self.price_data.sort_values('timestamp')
                
            # 确保必要的列存在
            required_columns = ['open', 'high', 'low', 'close']
            for col in required_columns:
                if col not in self.price_data.columns:
                    if col == 'close' and 'price' in self.price_data.columns:
                        self.price_data['close'] = self.price_data['price']
                    else:
                        print(f"数据缺少必要的列: {col}")
                        return False
                        
            print(f"成功加载{len(self.price_data)}条历史数据")
            return True
            
        except Exception as e:
            print(f"加载价格数据时出错: {e}")
            return False
            
    def generate_mock_data(self, symbol, start_date, days=365, volatility=0.02):
        """
        生成模拟价格数据（用于测试）
        
        Args:
            symbol: 交易对，例如 'BTC/USDT'
            start_date: 开始日期，格式 'YYYY-MM-DD'
            days: 生成的天数
            volatility: 价格波动率
            
        Returns:
            bool: 生成成功返回 True
        """
        try:
            # 生成日期范围
            start = pd.to_datetime(start_date)
            dates = [start + timedelta(days=i) for i in range(days)]
            
            # 生成模拟价格
            base_price = 10000  # 起始价格
            prices = [base_price]
            
            # 使用随机游走模型生成价格
            for i in range(1, days):
                # 随机日收益率，服从正态分布
                daily_return = np.random.normal(0.0005, volatility)  # 均值略大于0，有小偏向上涨
                prices.append(prices[-1] * (1 + daily_return))
                
            # 创建DataFrame
            self.price_data = pd.DataFrame({
                'timestamp': dates,
                'open': prices,
                'high': [p * (1 + np.random.uniform(0, 0.01)) for p in prices],
                'low': [p * (1 - np.random.uniform(0, 0.01)) for p in prices],
                'close': prices,
                'volume': [np.random.uniform(1000, 10000) * p for p in prices]
            })
            
            # 添加一些波动
            # 1. 添加一些趋势
            trend_period = np.random.randint(30, 90)  # 30-90天的趋势
            trend = np.sin(np.linspace(0, np.pi * days / trend_period, days))
            
            # 2. 调整价格包含趋势
            for i in range(days):
                trend_factor = 1 + trend[i] * 0.1  # 最大±10%的趋势影响
                self.price_data.loc[i, 'close'] *= trend_factor
                self.price_data.loc[i, 'open'] *= trend_factor
                self.price_data.loc[i, 'high'] *= trend_factor
                self.price_data.loc[i, 'low'] *= trend_factor
                
            print(f"成功生成{days}天的模拟价格数据")
            return True
            
        except Exception as e:
            print(f"生成模拟数据时出错: {e}")
            return False
            
    def run_backtest(self, strategy_func, *args, **kwargs):
        """
        运行回测
        
        Args:
            strategy_func: 策略函数，接收价格数据和日期索引，返回交易信号
            *args, **kwargs: 传递给策略函数的参数
            
        Returns:
            dict: 回测结果
        """
        if self.price_data is None or len(self.price_data) == 0:
            print("没有价格数据，请先加载数据")
            return None
            
        # 重置回测状态
        self.portfolio = {"cash": self.initial_capital, "assets": {}}
        self.trade_history = []
        self.portfolio_history = []
        
        symbol = kwargs.get('symbol', 'BTC/USDT')
        asset = symbol.split('/')[0]  # 例如，从 'BTC/USDT' 提取 'BTC'
        
        # 初始化资产持仓为0
        self.portfolio["assets"][asset] = 0
        
        # 遍历每个时间点
        for i in range(len(self.price_data)):
            current_date = self.price_data['timestamp'].iloc[i] if 'timestamp' in self.price_data.columns else i
            current_price = self.price_data['close'].iloc[i]
            
            # 记录每日投资组合价值
            portfolio_value = self.calculate_portfolio_value(i)
            self.portfolio_history.append({
                'date': current_date,
                'portfolio_value': portfolio_value,
                'cash': self.portfolio['cash'],
                'asset_value': portfolio_value - self.portfolio['cash'],
                'price': current_price
            })
            
            # 跳过第一天（没有足够的历史数据计算指标）
            if i == 0:
                continue
                
            # 调用策略函数获取交易信号
            historical_data = self.price_data.iloc[:i+1]
            signal = strategy_func(historical_data, i, *args, **kwargs)
            
            # 执行交易
            if signal:
                action = signal.get('action')
                quantity = signal.get('quantity', 0)
                
                if action and quantity > 0:
                    self.execute_trade(action, quantity, current_price, current_date, asset)
                    
        # 计算回测结果
        results = self.calculate_results()
        
        return results
        
    def execute_trade(self, action, quantity, price, date, asset):
        """
        执行交易
        
        Args:
            action: 交易动作，'buy' 或 'sell'
            quantity: 交易数量
            price: 交易价格
            date: 交易日期
            asset: 资产名称，例如 'BTC'
        """
        if action not in ['buy', 'sell']:
            return
            
        trade_value = quantity * price
        fee = trade_value * self.fee_rate
        
        if action == 'buy':
            # 检查现金是否足够
            total_cost = trade_value + fee
            if total_cost > self.portfolio['cash']:
                # 调整为可负担的最大数量
                affordable = self.portfolio['cash'] / (price * (1 + self.fee_rate))
                quantity = affordable
                trade_value = quantity * price
                fee = trade_value * self.fee_rate
                
            # 执行买入
            self.portfolio['cash'] -= (trade_value + fee)
            self.portfolio['assets'][asset] = self.portfolio['assets'].get(asset, 0) + quantity
            
        elif action == 'sell':
            # 检查持仓是否足够
            current_holding = self.portfolio['assets'].get(asset, 0)
            if quantity > current_holding:
                quantity = current_holding
                trade_value = quantity * price
                fee = trade_value * self.fee_rate
                
            # 执行卖出
            self.portfolio['cash'] += (trade_value - fee)
            self.portfolio['assets'][asset] = current_holding - quantity
            
        # 记录交易
        self.trade_history.append({
            'date': date,
            'action': action,
            'asset': asset,
            'quantity': quantity,
            'price': price,
            'value': trade_value,
            'fee': fee
        })
        
    def calculate_portfolio_value(self, index):
        """
        计算特定时间点的投资组合总价值
        
        Args:
            index: 价格数据的索引
            
        Returns:
            float: 投资组合价值
        """
        total_value = self.portfolio['cash']
        
        for asset, quantity in self.portfolio['assets'].items():
            price = self.price_data['close'].iloc[index]
            asset_value = quantity * price
            total_value += asset_value
            
        return total_value
        
    def calculate_results(self):
        """
        计算回测结果和性能指标
        
        Returns:
            dict: 回测结果和性能指标
        """
        if not self.portfolio_history:
            return {
                'total_return': 0,
                'total_trades': 0,
                'final_value': self.initial_capital
            }
            
        # 转换为DataFrame方便计算
        portfolio_df = pd.DataFrame(self.portfolio_history)
        
        # 计算日收益率
        portfolio_df['daily_return'] = portfolio_df['portfolio_value'].pct_change()
        
        # 计算累积收益
        initial_value = self.initial_capital
        final_value = portfolio_df['portfolio_value'].iloc[-1]
        total_return = (final_value - initial_value) / initial_value
        
        # 计算年化收益率（假设252个交易日/年）
        days = len(portfolio_df)
        annual_return = (1 + total_return) ** (252 / days) - 1 if days > 0 else 0
        
        # 计算最大回撤
        portfolio_df['cummax'] = portfolio_df['portfolio_value'].cummax()
        portfolio_df['drawdown'] = (portfolio_df['portfolio_value'] / portfolio_df['cummax'] - 1)
        max_drawdown = portfolio_df['drawdown'].min()
        
        # 计算夏普比率（假设无风险利率为0）
        sharpe_ratio = portfolio_df['daily_return'].mean() / portfolio_df['daily_return'].std() * np.sqrt(252) if len(portfolio_df) > 1 else 0
        
        # 计算交易统计
        trades_df = pd.DataFrame(self.trade_history) if self.trade_history else pd.DataFrame()
        total_trades = len(trades_df)
        buy_trades = len(trades_df[trades_df['action'] == 'buy']) if total_trades > 0 else 0
        sell_trades = len(trades_df[trades_df['action'] == 'sell']) if total_trades > 0 else 0
        total_fees = trades_df['fee'].sum() if total_trades > 0 else 0
        
        # 计算盈亏比例
        if total_trades > 0 and 'value' in trades_df.columns:
            profitable_trades = 0
            losing_trades = 0
            
            # 匹配买卖配对
            buy_history = trades_df[trades_df['action'] == 'buy']
            sell_history = trades_df[trades_df['action'] == 'sell']
            
            for _, sell in sell_history.iterrows():
                matching_buys = buy_history[buy_history['date'] < sell['date']]
                
                if not matching_buys.empty:
                    # 简化：假设先买先卖
                    avg_buy_price = matching_buys['price'].mean()
                    if sell['price'] > avg_buy_price:
                        profitable_trades += 1
                    else:
                        losing_trades += 1
                        
            win_ratio = profitable_trades / total_trades if total_trades > 0 else 0
        else:
            win_ratio = 0
            
        return {
            'initial_value': initial_value,
            'final_value': final_value,
            'total_return': total_return,
            'annual_return': annual_return,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'total_trades': total_trades,
            'buy_trades': buy_trades,
            'sell_trades': sell_trades,
            'win_ratio': win_ratio,
            'total_fees': total_fees,
            'portfolio_history': self.portfolio_history,
            'trade_history': self.trade_history
        }
        
    def plot_results(self, figsize=(12, 10)):
        """
        绘制回测结果图表
        
        Args:
            figsize: 图表大小
        """
        if not self.portfolio_history:
            print("没有回测数据可以绘制")
            return
            
        portfolio_df = pd.DataFrame(self.portfolio_history)
        
        fig, axs = plt.subplots(3, 1, figsize=figsize, gridspec_kw={'height_ratios': [3, 1, 1]})
        
        # 1. 投资组合价值 vs 价格
        ax1 = axs[0]
        portfolio_df['portfolio_value'].plot(ax=ax1, color='blue', label='投资组合价值')
        ax1_right = ax1.twinx()
        portfolio_df['price'].plot(ax=ax1_right, color='green', alpha=0.5, label='资产价格')
        
        # 添加交易点
        if self.trade_history:
            trades_df = pd.DataFrame(self.trade_history)
            buy_points = trades_df[trades_df['action'] == 'buy']
            sell_points = trades_df[trades_df['action'] == 'sell']
            
            for i, trade in buy_points.iterrows():
                trade_date = pd.to_datetime(trade['date'])
                matching_idx = portfolio_df.index[portfolio_df['date'] == trade_date]
                if not matching_idx.empty:
                    idx = matching_idx[0]
                    ax1.scatter(idx, portfolio_df['portfolio_value'].iloc[idx], 
                              color='green', marker='^', s=100)
                              
            for i, trade in sell_points.iterrows():
                trade_date = pd.to_datetime(trade['date'])
                matching_idx = portfolio_df.index[portfolio_df['date'] == trade_date]
                if not matching_idx.empty:
                    idx = matching_idx[0]
                    ax1.scatter(idx, portfolio_df['portfolio_value'].iloc[idx], 
                              color='red', marker='v', s=100)
                              
        ax1.set_title('投资组合价值与资产价格')
        ax1.set_ylabel('投资组合价值')
        ax1_right.set_ylabel('资产价格')
        ax1.legend(loc='upper left')
        ax1_right.legend(loc='upper right')
        ax1.grid(True)
        
        # 2. 资产分配
        ax2 = axs[1]
        portfolio_df['cash_ratio'] = portfolio_df['cash'] / portfolio_df['portfolio_value']
        portfolio_df['asset_ratio'] = portfolio_df['asset_value'] / portfolio_df['portfolio_value']
        
        portfolio_df[['cash_ratio', 'asset_ratio']].plot(ax=ax2, kind='area', stacked=True, 
                                                      color=['gray', 'orange'])
        ax2.set_title('资产分配')
        ax2.set_ylabel('比例')
        ax2.set_ylim(0, 1)
        ax2.grid(True)
        
        # 3. 回撤
        ax3 = axs[2]
        portfolio_df['cummax'] = portfolio_df['portfolio_value'].cummax()
        portfolio_df['drawdown'] = (portfolio_df['portfolio_value'] / portfolio_df['cummax'] - 1) * 100
        
        portfolio_df['drawdown'].plot(ax=ax3, color='red', label='回撤百分比')
        ax3.set_title('回撤')
        ax3.set_ylabel('回撤 (%)')
        ax3.set_ylim(portfolio_df['drawdown'].min() * 1.1, 1)
        ax3.grid(True)
        
        plt.tight_layout()
        return fig
        
    def generate_report(self, filename=None):
        """
        生成回测报告
        
        Args:
            filename: 报告文件名（可选）
            
        Returns:
            str: 报告文本
        """
        results = self.calculate_results()
        
        # 格式化报告文本
        report = "加密货币回测分析报告\n"
        report += "=" * 50 + "\n\n"
        
        report += "总体绩效:\n"
        report += f"初始资本: ${results['initial_value']:,.2f}\n"
        report += f"最终价值: ${results['final_value']:,.2f}\n"
        report += f"总收益率: {results['total_return']*100:.2f}%\n"
        report += f"年化收益率: {results['annual_return']*100:.2f}%\n"
        report += f"最大回撤: {results['max_drawdown']*100:.2f}%\n"
        report += f"夏普比率: {results['sharpe_ratio']:.2f}\n\n"
        
        report += "交易统计:\n"
        report += f"总交易次数: {results['total_trades']}\n"
        report += f"买入交易次数: {results['buy_trades']}\n"
        report += f"卖出交易次数: {results['sell_trades']}\n"
        report += f"胜率: {results['win_ratio']*100:.2f}%\n"
        report += f"总手续费: ${results['total_fees']:,.2f}\n\n"
        
        # 输出到文件
        if filename:
            with open(filename, 'w') as f:
                f.write(report)
                
            print(f"回测报告已保存到: {filename}")
            
        return report


# 策略示例
def simple_moving_average_strategy(data, index, fast_period=10, slow_period=30):
    """
    简单移动平均线交叉策略
    
    Args:
        data: 价格历史数据
        index: 当前数据索引
        fast_period: 快速移动平均线周期
        slow_period: 慢速移动平均线周期
        
    Returns:
        dict: 交易信号，或者 None 表示不交易
    """
    # 检查数据是否足够
    if len(data) < slow_period:
        return None
        
    # 计算移动平均线
    prices = data['close']
    fast_ma = prices.rolling(window=fast_period).mean()
    slow_ma = prices.rolling(window=slow_period).mean()
    
    # 检查当前和前一个时间点的移动平均线
    current_fast = fast_ma.iloc[index]
    current_slow = slow_ma.iloc[index]
    previous_fast = fast_ma.iloc[index-1]
    previous_slow = slow_ma.iloc[index-1]
    
    # 生成交易信号
    current_price = prices.iloc[index]
    
    # 买入信号：快线从下方穿过慢线
    if previous_fast < previous_slow and current_fast > current_slow:
        return {
            'action': 'buy',
            'quantity': 0.1,  # 固定买入数量
            'price': current_price
        }
        
    # 卖出信号：快线从上方穿过慢线
    elif previous_fast > previous_slow and current_fast < current_slow:
        return {
            'action': 'sell',
            'quantity': 0.1,  # 固定卖出数量
            'price': current_price
        }
        
    return None


def rsi_strategy(data, index, period=14, overbought=70, oversold=30):
    """
    RSI超买超卖策略
    
    Args:
        data: 价格历史数据
        index: 当前数据索引
        period: RSI计算周期
        overbought: 超买阈值
        oversold: 超卖阈值
        
    Returns:
        dict: 交易信号，或者 None 表示不交易
    """
    # 检查数据是否足够
    if len(data) < period:
        return None
        
    # 计算RSI
    prices = data['close']
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    current_rsi = rsi.iloc[index]
    previous_rsi = rsi.iloc[index-1]
    current_price = prices.iloc[index]
    
    # 生成交易信号
    # 买入信号：RSI从超卖区上穿
    if previous_rsi < oversold and current_rsi > oversold:
        return {
            'action': 'buy',
            'quantity': 0.1,
            'price': current_price
        }
        
    # 卖出信号：RSI从超买区下穿
    elif previous_rsi > overbought and current_rsi < overbought:
        return {
            'action': 'sell',
            'quantity': 0.1,
            'price': current_price
        }
        
    return None


# 使用示例
if __name__ == "__main__":
    # 创建回测器实例
    backtester = CryptoBacktester(initial_capital=10000)
    
    # 生成模拟数据
    backtester.generate_mock_data('BTC/USDT', '2023-01-01', days=365)
    
    # 运行SMA策略回测
    sma_results = backtester.run_backtest(
        simple_moving_average_strategy, 
        fast_period=5, 
        slow_period=20,
        symbol='BTC/USDT'
    )
    
    # 打印结果
    print("\nSMA策略回测结果:")
    print(f"总收益率: {sma_results['total_return']*100:.2f}%")
    print(f"最大回撤: {sma_results['max_drawdown']*100:.2f}%")
    print(f"夏普比率: {sma_results['sharpe_ratio']:.2f}")
    print(f"总交易次数: {sma_results['total_trades']}")
    
    # 绘制结果
    backtester.plot_results()
    plt.show()
    
    # 生成报告
    report = backtester.generate_report()
    print("\n" + report)