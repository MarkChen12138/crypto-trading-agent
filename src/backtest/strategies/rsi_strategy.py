"""
RSI (Relative Strength Index) 交易策略

这个策略基于以下逻辑:
1. 当RSI从超卖区域（低于指定阈值）向上突破时买入
2. 当RSI从超买区域（高于指定阈值）向下突破时卖出
3. 支持可配置的RSI周期和阈值
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

class RSIStrategy:
    """
    基于相对强弱指数(RSI)的交易策略
    """
    
    def __init__(
        self, 
        period=14, 
        overbought=70, 
        oversold=30, 
        stop_loss_pct=0.05, 
        take_profit_pct=0.15
    ):
        """
        初始化RSI策略
        
        Args:
            period: RSI计算周期
            overbought: 超买阈值
            oversold: 超卖阈值
            stop_loss_pct: 止损百分比
            take_profit_pct: 止盈百分比
        """
        self.period = period
        self.overbought = overbought
        self.oversold = oversold
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.position = 0
        self.entry_price = 0
        self.stop_loss_price = 0
        self.take_profit_price = 0
        self.last_signal = None
        
    def calculate_rsi(self, data):
        """
        计算RSI指标
        
        Args:
            data: 包含收盘价的DataFrame
            
        Returns:
            带有RSI列的DataFrame
        """
        # 确保数据是DataFrame类型
        if not isinstance(data, pd.DataFrame):
            data = pd.DataFrame(data)
            
        # 确保有close列
        if 'close' not in data.columns:
            if 'price' in data.columns:
                data['close'] = data['price']
            else:
                raise ValueError("数据中缺少'close'或'price'列")
                
        # 计算价格变化
        delta = data['close'].diff()
        
        # 分离上涨和下跌
        gain = delta.copy()
        loss = delta.copy()
        gain[gain < 0] = 0
        loss[loss > 0] = 0
        loss = -loss  # 转换为正数
        
        # 计算平均收益和损失
        avg_gain = gain.rolling(window=self.period).mean()
        avg_loss = loss.rolling(window=self.period).mean()
        
        # 处理初始值为NaN的情况
        avg_gain = avg_gain.fillna(0)
        avg_loss = avg_loss.fillna(0)
        
        # 避免除零错误
        rs = avg_gain / avg_loss.replace(0, 0.001)
        
        # 计算RSI
        data['rsi'] = 100 - (100 / (1 + rs))
        
        return data
        
    def generate_signals(self, data):
        """
        生成交易信号
        
        Args:
            data: 价格数据DataFrame
            
        Returns:
            带有信号的DataFrame
        """
        # 计算RSI
        df = self.calculate_rsi(data.copy())
        
        # 初始化信号列
        df['signal'] = 0
        
        # 生成交易信号
        for i in range(1, len(df)):
            # 检查RSI从超卖区上穿
            if df['rsi'].iloc[i-1] < self.oversold and df['rsi'].iloc[i] > self.oversold:
                df.loc[df.index[i], 'signal'] = 1  # 买入信号
                
            # 检查RSI从超买区下穿
            elif df['rsi'].iloc[i-1] > self.overbought and df['rsi'].iloc[i] < self.overbought:
                df.loc[df.index[i], 'signal'] = -1  # 卖出信号
                
            # 检查止损或止盈（如果持有仓位）
            elif self.position > 0:
                current_price = df['close'].iloc[i]
                
                # 止损检查
                if current_price <= self.stop_loss_price:
                    df.loc[df.index[i], 'signal'] = -1  # 止损卖出
                    
                # 止盈检查
                elif current_price >= self.take_profit_price:
                    df.loc[df.index[i], 'signal'] = -1  # 止盈卖出
                    
        return df
        
    def backtest(self, data, initial_capital=10000, trade_size=0.1):
        """
        回测RSI策略
        
        Args:
            data: 价格数据DataFrame
            initial_capital: 初始资金
            trade_size: 每次交易的资金比例
            
        Returns:
            带有回测结果的DataFrame
        """
        # 生成信号
        df = self.generate_signals(data.copy())
        
        # 初始化回测列
        df['position'] = 0
        df['cash'] = initial_capital
        df['holdings'] = 0
        df['total'] = initial_capital
        df['returns'] = 0
        
        # 遍历数据
        for i in range(1, len(df)):
            price = df['close'].iloc[i]
            signal = df['signal'].iloc[i]
            
            # 默认继承前一天的状态
            df.loc[df.index[i], 'position'] = df['position'].iloc[i-1]
            df.loc[df.index[i], 'cash'] = df['cash'].iloc[i-1]
            
            # 处理交易信号
            if signal == 1 and df['position'].iloc[i-1] == 0:  # 买入信号且当前无持仓
                # 计算购买数量和成本
                available_cash = df['cash'].iloc[i-1] * trade_size
                qty = available_cash / price
                cost = qty * price
                
                # 更新持仓
                df.loc[df.index[i], 'position'] = qty
                df.loc[df.index[i], 'cash'] = df['cash'].iloc[i-1] - cost
                
                # 更新实例属性
                self.position = qty
                self.entry_price = price
                self.stop_loss_price = price * (1 - self.stop_loss_pct)
                self.take_profit_price = price * (1 + self.take_profit_pct)
                self.last_signal = 'buy'
                
            elif signal == -1 and df['position'].iloc[i-1] > 0:  # 卖出信号且当前有持仓
                # 计算卖出收益
                qty = df['position'].iloc[i-1]
                revenue = qty * price
                
                # 更新持仓
                df.loc[df.index[i], 'position'] = 0
                df.loc[df.index[i], 'cash'] = df['cash'].iloc[i-1] + revenue
                
                # 更新实例属性
                self.position = 0
                self.entry_price = 0
                self.stop_loss_price = 0
                self.take_profit_price = 0
                self.last_signal = 'sell'
                
            # 计算当前持仓价值
            df.loc[df.index[i], 'holdings'] = df['position'].iloc[i] * price
            
            # 计算总资产
            df.loc[df.index[i], 'total'] = df['cash'].iloc[i] + df['holdings'].iloc[i]
            
            # 计算每日回报率
            df.loc[df.index[i], 'returns'] = df['total'].iloc[i] / df['total'].iloc[i-1] - 1
            
        # 计算累积回报率
        df['cumulative_returns'] = (1 + df['returns']).cumprod() - 1
        
        return df
        
    def plot_results(self, data):
        """
        绘制回测结果
        
        Args:
            data: 包含回测结果的DataFrame
        """
        # 创建图表
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 15), sharex=True, gridspec_kw={'height_ratios': [2, 1, 1]})
        
        # 绘制价格和信号
        ax1.plot(data.index, data['close'], label='Price')
        
        # 买入信号
        buy_signals = data[data['signal'] == 1]
        ax1.scatter(buy_signals.index, buy_signals['close'], marker='^', color='green', s=100, label='Buy')
        
        # 卖出信号
        sell_signals = data[data['signal'] == -1]
        ax1.scatter(sell_signals.index, sell_signals['close'], marker='v', color='red', s=100, label='Sell')
        
        ax1.set_title(f'RSI交易策略 (周期={self.period}, 超买={self.overbought}, 超卖={self.oversold})')
        ax1.set_ylabel('价格')
        ax1.legend()
        ax1.grid(True)
        
        # 绘制RSI
        ax2.plot(data.index, data['rsi'], color='purple', label='RSI')
        ax2.axhline(y=self.overbought, color='r', linestyle='-', alpha=0.3)
        ax2.axhline(y=self.oversold, color='g', linestyle='-', alpha=0.3)
        ax2.fill_between(data.index, y1=self.overbought, y2=100, color='r', alpha=0.1)
        ax2.fill_between(data.index, y1=0, y2=self.oversold, color='g', alpha=0.1)
        ax2.set_ylabel('RSI')
        ax2.set_ylim(0, 100)
        ax2.legend()
        ax2.grid(True)
        
        # 绘制投资组合价值
        ax3.plot(data.index, data['total'], color='blue', label='投资组合价值')
        ax3.set_ylabel('价值')
        ax3.set_xlabel('日期')
        ax3.legend()
        ax3.grid(True)
        
        plt.tight_layout()
        plt.show()
        
    def generate_statistics(self, data):
        """
        生成策略统计信息
        
        Args:
            data: 包含回测结果的DataFrame
            
        Returns:
            策略统计字典
        """
        # 计算年化收益率 (假设252个交易日)
        total_days = (data.index[-1] - data.index[0]).days
        trading_days_per_year = 252
        years = total_days / 365
        
        total_return = data['total'].iloc[-1] / data['total'].iloc[0] - 1
        annual_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
        
        # 计算夏普比率 (假设无风险利率为0)
        daily_returns = data['returns'].dropna()
        sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * np.sqrt(trading_days_per_year) if len(daily_returns) > 0 else 0
        
        # 计算最大回撤
        cumulative_max = data['total'].cummax()
        drawdown = (data['total'] / cumulative_max - 1)
        max_drawdown = drawdown.min()
        
        # 计算盈亏比
        wins = daily_returns[daily_returns > 0]
        losses = daily_returns[daily_returns < 0]
        win_rate = len(wins) / len(daily_returns) if len(daily_returns) > 0 else 0
        loss_rate = len(losses) / len(daily_returns) if len(daily_returns) > 0 else 0
        avg_win = wins.mean() if len(wins) > 0 else 0
        avg_loss = losses.mean() if len(losses) > 0 else 0
        profit_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else 0
        
        # 计算交易次数
        trades = data[data['signal'] != 0]
        trade_count = len(trades)
        
        return {
            "起始资金": data['total'].iloc[0],
            "结束资金": data['total'].iloc[-1],
            "总回报率": total_return * 100,
            "年化回报率": annual_return * 100,
            "夏普比率": sharpe_ratio,
            "最大回撤": max_drawdown * 100,
            "胜率": win_rate * 100,
            "亏损率": loss_rate * 100,
            "平均盈利": avg_win * 100,
            "平均亏损": avg_loss * 100,
            "盈亏比": profit_loss_ratio,
            "交易次数": trade_count
        }
        
    def generate_next_signal(self, data):
        """
        根据当前市场数据生成下一个交易信号
        
        Args:
            data: 最新的市场数据
            
        Returns:
            交易信号字典 {'action': 'buy'/'sell'/'hold', 'reason': '信号原因'}
        """
        # 确保数据是最新的
        df = self.calculate_rsi(data.copy())
        
        # 获取最新值
        latest_rsi = df['rsi'].iloc[-1]
        previous_rsi = df['rsi'].iloc[-2] if len(df) > 1 else None
        latest_price = df['close'].iloc[-1]
        
        # 检查RSI信号
        if previous_rsi is not None:
            # RSI从超卖区上穿
            if previous_rsi < self.oversold and latest_rsi > self.oversold:
                return {
                    'action': 'buy',
                    'price': latest_price,
                    'reason': f'RSI从超卖区({previous_rsi:.2f})向上突破({latest_rsi:.2f})',
                    'rsi': latest_rsi
                }
                
            # RSI从超买区下穿
            elif previous_rsi > self.overbought and latest_rsi < self.overbought:
                return {
                    'action': 'sell',
                    'price': latest_price,
                    'reason': f'RSI从超买区({previous_rsi:.2f})向下突破({latest_rsi:.2f})',
                    'rsi': latest_rsi
                }
        
        # 检查止损或止盈
        if self.position > 0:
            # 止损检查
            if latest_price <= self.stop_loss_price:
                return {
                    'action': 'sell',
                    'price': latest_price,
                    'reason': f'触发止损，入场价: {self.entry_price:.2f}, 止损价: {self.stop_loss_price:.2f}',
                    'rsi': latest_rsi
                }
                
            # 止盈检查
            elif latest_price >= self.take_profit_price:
                return {
                    'action': 'sell',
                    'price': latest_price,
                    'reason': f'触发止盈，入场价: {self.entry_price:.2f}, 止盈价: {self.take_profit_price:.2f}',
                    'rsi': latest_rsi
                }
        
        # 无信号
        return {
            'action': 'hold',
            'price': latest_price,
            'reason': f'RSI当前值为{latest_rsi:.2f}，处于中性区域',
            'rsi': latest_rsi
        }


# 使用示例
if __name__ == "__main__":
    # 创建一些模拟数据
    dates = pd.date_range(start='2023-01-01', periods=200, freq='D')
    np.random.seed(42)
    prices = np.random.randn(200).cumsum() + 100
    data = pd.DataFrame({
        'timestamp': dates,
        'close': prices
    }).set_index('timestamp')
    
    # 初始化策略
    strategy = RSIStrategy(period=14, overbought=70, oversold=30)
    
    # 运行回测
    results = strategy.backtest(data)
    
    # 显示统计信息
    stats = strategy.generate_statistics(results)
    for key, value in stats.items():
        print(f"{key}: {value}")
        
    # 绘制图表
    strategy.plot_results(results)
