import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.utils.logging_config import setup_logger

logger = setup_logger('ema_cross_strategy')

class EMAStrategy:
    """
    EMA交叉策略 - 基于指数移动平均线交叉的交易策略
    
    策略逻辑:
    1. 快速EMA上穿慢速EMA: 买入信号
    2. 快速EMA下穿慢速EMA: 卖出信号
    3. 其他情况: 持有现有仓位
    
    可选增强:
    - 趋势过滤器: 只在长期趋势方向上交易
    - 波动率过滤器: 在波动率过高时减少仓位
    - 止损和止盈: 设置自动止损和止盈点
    """
    
    def __init__(self, fast_period=9, slow_period=21, signal_period=9, 
                 trend_filter=False, trend_period=50, 
                 volatility_filter=False, volatility_period=20,
                 stop_loss=None, take_profit=None):
        """
        初始化EMA交叉策略
        
        Args:
            fast_period: 快速EMA周期
            slow_period: 慢速EMA周期
            signal_period: MACD信号线周期
            trend_filter: 是否启用趋势过滤器
            trend_period: 趋势EMA周期
            volatility_filter: 是否启用波动率过滤器
            volatility_period: 波动率计算周期
            stop_loss: 止损百分比 (小数，例如0.05表示5%)
            take_profit: 止盈百分比 (小数，例如0.1表示10%)
        """
        self.name = "EMA_Cross"
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
        self.trend_filter = trend_filter
        self.trend_period = trend_period
        self.volatility_filter = volatility_filter
        self.volatility_period = volatility_period
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        
        # 内部状态
        self.position = 0  # 当前仓位 (0 = 无仓位, 1 = 多头)
        self.entry_price = 0  # 入场价格
        self.exit_price = 0   # 出场价格
        self.last_signal = 'neutral'  # 最后信号
        
        logger.info(f"EMA交叉策略初始化: 快速EMA={fast_period}, 慢速EMA={slow_period}, 信号EMA={signal_period}")
        
    def prepare_data(self, df):
        """
        准备策略所需的数据
        
        Args:
            df: 价格数据DataFrame，必须包含'close'列
            
        Returns:
            DataFrame: 添加了指标的DataFrame
        """
        if 'close' not in df.columns:
            raise ValueError("数据必须包含'close'列")
            
        # 复制数据，避免修改原始数据
        result = df.copy()
        
        # 计算快速和慢速EMA
        result['ema_fast'] = result['close'].ewm(span=self.fast_period, adjust=False).mean()
        result['ema_slow'] = result['close'].ewm(span=self.slow_period, adjust=False).mean()
        
        # 计算MACD和信号线
        result['macd'] = result['ema_fast'] - result['ema_slow']
        result['macd_signal'] = result['macd'].ewm(span=self.signal_period, adjust=False).mean()
        result['macd_histogram'] = result['macd'] - result['macd_signal']
        
        # 额外指标(可选)
        if self.trend_filter:
            # 添加长期趋势EMA
            result['ema_trend'] = result['close'].ewm(span=self.trend_period, adjust=False).mean()
            
        if self.volatility_filter:
            # 计算价格变动百分比
            result['returns'] = result['close'].pct_change()
            # 计算波动率 (年化标准差)
            result['volatility'] = result['returns'].rolling(window=self.volatility_period).std() * np.sqrt(252)
            
        logger.debug(f"数据准备完成: {len(result)}行")
        return result
        
    def generate_signal(self, df, i):
        """
        生成交易信号
        
        Args:
            df: 包含指标的DataFrame
            i: 当前数据行索引
            
        Returns:
            str: 交易信号 ('buy', 'sell', 'hold')
        """
        # 确保索引有效
        if i <= self.slow_period:
            return 'hold'  # 数据不足，保持观望
            
        try:
            # 提取当前和前一个时间点的指标值
            current_fast = df['ema_fast'].iloc[i]
            current_slow = df['ema_slow'].iloc[i]
            previous_fast = df['ema_fast'].iloc[i-1]
            previous_slow = df['ema_slow'].iloc[i-1]
            
            # 检查EMA交叉
            ema_cross_up = previous_fast <= previous_slow and current_fast > current_slow
            ema_cross_down = previous_fast >= previous_slow and current_fast < current_slow
            
            # 检查MACD信号
            current_macd = df['macd'].iloc[i]
            current_signal = df['macd_signal'].iloc[i]
            previous_macd = df['macd'].iloc[i-1]
            previous_signal = df['macd_signal'].iloc[i-1]
            
            macd_cross_up = previous_macd <= previous_signal and current_macd > current_signal
            macd_cross_down = previous_macd >= previous_signal and current_macd < current_signal
            
            # 获取当前价格
            current_price = df['close'].iloc[i]
            
            # 应用趋势过滤器(可选)
            trend_condition = True
            if self.trend_filter and 'ema_trend' in df.columns:
                trend_ema = df['ema_trend'].iloc[i]
                trend_condition = current_price > trend_ema  # 只在上升趋势中做多
                
            # 应用波动率过滤器(可选)
            volatility_condition = True
            if self.volatility_filter and 'volatility' in df.columns:
                current_volatility = df['volatility'].iloc[i]
                volatility_threshold = 0.4  # 40%年化波动率阈值
                volatility_condition = current_volatility < volatility_threshold
                
            # 检查止损(如果持有仓位)
            if self.position == 1 and self.stop_loss is not None:
                stop_price = self.entry_price * (1 - self.stop_loss)
                if current_price <= stop_price:
                    logger.info(f"触发止损: 当前价格({current_price})低于止损价({stop_price})")
                    self.exit_price = current_price
                    self.position = 0
                    self.last_signal = 'sell'
                    return 'sell'
                    
            # 检查止盈(如果持有仓位)
            if self.position == 1 and self.take_profit is not None:
                take_profit_price = self.entry_price * (1 + self.take_profit)
                if current_price >= take_profit_price:
                    logger.info(f"触发止盈: 当前价格({current_price})高于止盈价({take_profit_price})")
                    self.exit_price = current_price
                    self.position = 0
                    self.last_signal = 'sell'
                    return 'sell'
                    
            # 买入信号: EMA快线上穿慢线 + MACD确认 + 趋势确认 + 波动率确认
            if (ema_cross_up or macd_cross_up) and trend_condition and volatility_condition and self.position == 0:
                self.entry_price = current_price
                self.position = 1
                self.last_signal = 'buy'
                return 'buy'
                
            # 卖出信号: EMA快线下穿慢线 + MACD确认
            elif (ema_cross_down or macd_cross_down) and self.position == 1:
                self.exit_price = current_price
                self.position = 0
                self.last_signal = 'sell'
                return 'sell'
                
            # 其他情况保持现有仓位
            else:
                self.last_signal = 'hold'
                return 'hold'
                
        except Exception as e:
            logger.error(f"生成信号时出错: {e}")
            return 'hold'  # 出错时默认持有
            
    def get_signal(self, df, i=None):
        """
        获取指定时间点的交易信号(外部接口)
        
        Args:
            df: 价格数据DataFrame
            i: 索引位置，默认为最后一行
            
        Returns:
            dict: 包含信号信息的字典
        """
        # 准备数据
        df_indicators = self.prepare_data(df)
        
        # 如果未指定索引，使用最后一行
        if i is None:
            i = len(df_indicators) - 1
            
        # 生成信号
        signal = self.generate_signal(df_indicators, i)
        
        # 获取相关指标值
        fast_ema = df_indicators['ema_fast'].iloc[i]
        slow_ema = df_indicators['ema_slow'].iloc[i]
        macd = df_indicators['macd'].iloc[i]
        macd_signal = df_indicators['macd_signal'].iloc[i]
        macd_hist = df_indicators['macd_histogram'].iloc[i]
        
        # 获取当前价格
        current_price = df_indicators['close'].iloc[i]
        
        # 构建返回结果
        result = {
            'action': signal,
            'price': current_price,
            'indicators': {
                'ema_fast': fast_ema,
                'ema_slow': slow_ema,
                'macd': macd,
                'macd_signal': macd_signal,
                'macd_histogram': macd_hist
            },
            'position': self.position,
            'entry_price': self.entry_price if self.position == 1 else None
        }
        
        # 添加可选指标
        if self.trend_filter and 'ema_trend' in df_indicators.columns:
            result['indicators']['ema_trend'] = df_indicators['ema_trend'].iloc[i]
            
        if self.volatility_filter and 'volatility' in df_indicators.columns:
            result['indicators']['volatility'] = df_indicators['volatility'].iloc[i]
            
        return result
        
    def backtest(self, df, initial_capital=10000):
        """
        回测策略
        
        Args:
            df: 价格数据DataFrame
            initial_capital: 初始资金
            
        Returns:
            tuple: (包含回测结果的DataFrame, 性能指标字典)
        """
        logger.info(f"开始回测EMA交叉策略: 初始资金={initial_capital}")
        
        # 重置内部状态
        self.position = 0
        self.entry_price = 0
        self.exit_price = 0
        
        # 准备数据
        df_indicators = self.prepare_data(df)
        
        # 初始化回测结果
        backtest_results = df_indicators.copy()
        backtest_results['signal'] = 'hold'
        backtest_results['position'] = 0
        backtest_results['cash'] = initial_capital
        backtest_results['holdings'] = 0.0
        backtest_results['portfolio_value'] = initial_capital
        
        # 记录交易
        trades = []
        
        # 遍历每一天
        for i in range(len(backtest_results)):
            if i <= self.slow_period:
                continue  # 跳过前几天，等待指标计算足够数据
                
            # 生成当天的信号
            signal = self.generate_signal(backtest_results, i)
            
            # 获取当天价格
            price = backtest_results['close'].iloc[i]
            
            # 记录信号
            backtest_results.loc[backtest_results.index[i], 'signal'] = signal
            
            # 前一天的现金和持仓
            if i > 0:
                backtest_results.loc[backtest_results.index[i], 'cash'] = backtest_results['cash'].iloc[i-1]
                backtest_results.loc[backtest_results.index[i], 'position'] = backtest_results['position'].iloc[i-1]
                backtest_results.loc[backtest_results.index[i], 'holdings'] = backtest_results['holdings'].iloc[i-1]
                
            # 根据信号执行交易
            if signal == 'buy' and backtest_results['position'].iloc[i] == 0:
                # 计算可买入的数量
                available_cash = backtest_results['cash'].iloc[i]
                quantity = available_cash / price
                
                # 执行买入
                backtest_results.loc[backtest_results.index[i], 'cash'] = 0
                backtest_results.loc[backtest_results.index[i], 'holdings'] = quantity * price
                backtest_results.loc[backtest_results.index[i], 'position'] = 1
                
                # 记录交易
                trades.append({
                    'date': backtest_results.index[i],
                    'action': 'buy',
                    'price': price,
                    'quantity': quantity,
                    'value': quantity * price
                })
                
                logger.debug(f"买入信号: 日期={backtest_results.index[i]}, 价格={price:.2f}, 数量={quantity:.6f}")
                
            elif signal == 'sell' and backtest_results['position'].iloc[i] == 1:
                # 计算持仓价值
                holdings_value = backtest_results['holdings'].iloc[i]
                quantity = holdings_value / price
                
                # 执行卖出
                backtest_results.loc[backtest_results.index[i], 'cash'] = holdings_value
                backtest_results.loc[backtest_results.index[i], 'holdings'] = 0
                backtest_results.loc[backtest_results.index[i], 'position'] = 0
                
                # 记录交易
                trades.append({
                    'date': backtest_results.index[i],
                    'action': 'sell',
                    'price': price,
                    'quantity': quantity,
                    'value': holdings_value
                })
                
                logger.debug(f"卖出信号: 日期={backtest_results.index[i]}, 价格={price:.2f}, 数量={quantity:.6f}")
                
            # 更新持仓价值
            backtest_results.loc[backtest_results.index[i], 'holdings'] = backtest_results['position'].iloc[i] * backtest_results['close'].iloc[i] * (backtest_results['holdings'].iloc[i] / price if backtest_results['position'].iloc[i] == 1 and price > 0 else 0)
            
            # 计算投资组合总价值
            backtest_results.loc[backtest_results.index[i], 'portfolio_value'] = backtest_results['cash'].iloc[i] + backtest_results['holdings'].iloc[i]
            
        # 计算回测性能指标
        performance = self._calculate_performance(backtest_results, trades)
        
        logger.info(f"回测完成: 总收益率={performance['total_return']:.2%}, 夏普比率={performance['sharpe_ratio']:.2f}")
        
        return backtest_results, performance, trades
        
    def _calculate_performance(self, results, trades):
        """
        计算回测性能指标
        
        Args:
            results: 回测结果DataFrame
            trades: 交易记录列表
            
        Returns:
            dict: 性能指标字典
        """
        # 计算每日回报
        results['daily_return'] = results['portfolio_value'].pct_change()
        
        # 初始和最终投资组合价值
        initial_value = results['portfolio_value'].iloc[0]
        final_value = results['portfolio_value'].iloc[-1]
        
        # 总收益率
        total_return = (final_value / initial_value) - 1
        
        # 年化收益率（假设252个交易日/年）
        days = len(results)
        annual_return = (1 + total_return) ** (252 / days) - 1
        
        # 计算波动率（年化）
        volatility = results['daily_return'].std() * np.sqrt(252)
        
        # 计算夏普比率（假设无风险收益率为0）
        sharpe_ratio = annual_return / volatility if volatility > 0 else 0
        
        # 计算最大回撤
        results['cum_max'] = results['portfolio_value'].cummax()
        results['drawdown'] = (results['portfolio_value'] / results['cum_max']) - 1
        max_drawdown = results['drawdown'].min()
        
        # 计算交易统计
        num_trades = len(trades)
        num_wins = 0
        gross_profit = 0
        gross_loss = 0
        
        for i in range(0, len(trades), 2):
            if i + 1 < len(trades):  # 确保有配对的买入/卖出
                entry = trades[i]
                exit = trades[i+1]
                
                profit = exit['value'] - entry['value']
                
                if profit > 0:
                    num_wins += 1
                    gross_profit += profit
                else:
                    gross_loss += abs(profit)
        
        win_rate = num_wins / (num_trades / 2) if num_trades > 0 else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # 返回性能指标
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'num_trades': num_trades,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'gross_profit': gross_profit,
            'gross_loss': gross_loss
        }
        
    def plot_backtest(self, results, trades=None, figsize=(15, 10)):
        """
        绘制回测结果图表
        
        Args:
            results: 回测结果DataFrame
            trades: 交易记录列表
            figsize: 图表大小
            
        Returns:
            matplotlib.figure.Figure: 图表对象
        """
        plt.figure(figsize=figsize)
        
        # 创建子图
        ax1 = plt.subplot2grid((4, 1), (0, 0), rowspan=2)
        ax2 = plt.subplot2grid((4, 1), (2, 0))
        ax3 = plt.subplot2grid((4, 1), (3, 0))
        
        # 绘制价格和EMA
        ax1.plot(results.index, results['close'], label='价格', alpha=0.5)
        ax1.plot(results.index, results['ema_fast'], label=f'快速EMA ({self.fast_period})', alpha=0.8)
        ax1.plot(results.index, results['ema_slow'], label=f'慢速EMA ({self.slow_period})', alpha=0.8)
        
        if self.trend_filter and 'ema_trend' in results.columns:
            ax1.plot(results.index, results['ema_trend'], label=f'趋势EMA ({self.trend_period})', alpha=0.8, linestyle='--')
            
        # 绘制买入和卖出点
        if trades:
            buy_dates = [trade['date'] for trade in trades if trade['action'] == 'buy']
            buy_prices = [trade['price'] for trade in trades if trade['action'] == 'buy']
            
            sell_dates = [trade['date'] for trade in trades if trade['action'] == 'sell']
            sell_prices = [trade['price'] for trade in trades if trade['action'] == 'sell']
            
            ax1.scatter(buy_dates, buy_prices, marker='^', color='g', s=100, label='买入')
            ax1.scatter(sell_dates, sell_prices, marker='v', color='r', s=100, label='卖出')
            
        # 设置标题和图例
        ax1.set_title('EMA交叉策略回测')
        ax1.set_ylabel('价格')
        ax1.legend()
        ax1.grid(True)
        
        # 绘制MACD
        ax2.plot(results.index, results['macd'], label='MACD')
        ax2.plot(results.index, results['macd_signal'], label=f'信号线 ({self.signal_period})')
        ax2.bar(results.index, results['macd_histogram'], label='直方图', alpha=0.3)
        ax2.set_ylabel('MACD')
        ax2.legend()
        ax2.grid(True)
        
        # 绘制投资组合价值
        ax3.plot(results.index, results['portfolio_value'], label='投资组合价值', color='purple')
        ax3.set_ylabel('价值')
        ax3.set_xlabel('日期')
        ax3.legend()
        ax3.grid(True)
        
        plt.tight_layout()
        
        return plt.gcf()


# 示例用法
if __name__ == "__main__":
    # 创建示例数据
    dates = pd.date_range('2022-01-01', periods=200, freq='D')
    prices = np.random.normal(0, 1, 200).cumsum() + 100  # 随机价格序列
    
    # 创建DataFrame
    df = pd.DataFrame({
        'open': prices * 0.99,
        'high': prices * 1.02,
        'low': prices * 0.98,
        'close': prices,
        'volume': np.random.randint(1000, 100000, 200)
    }, index=dates)
    
    # 创建策略
    strategy = EMAStrategy(fast_period=9, slow_period=21, signal_period=9,
                         trend_filter=True, trend_period=50,
                         stop_loss=0.05, take_profit=0.15)
    
    # 执行回测
    results, performance, trades = strategy.backtest(df)
    
    # 打印性能
    print("回测性能:")
    for key, value in performance.items():
        print(f"{key}: {value}")
        
    # 绘制回测图表
    strategy.plot_backtest(results, trades)
    plt.show()
