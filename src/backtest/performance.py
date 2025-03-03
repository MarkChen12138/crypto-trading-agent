"""
回测性能分析工具

提供全面的交易策略性能指标计算和结果可视化功能
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter
import empyrical

class PerformanceAnalyzer:
    """
    回测结果性能分析工具，计算各种性能指标，并提供可视化
    """
    
    def __init__(self, results, benchmark_returns=None):
        """
        初始化性能分析器
        
        Args:
            results: 包含回测结果的DataFrame，必须包含'total'和'returns'列
            benchmark_returns: 可选，基准回报率序列，用于比较分析
        """
        self.results = results
        
        # 验证必要的列
        required_columns = ['total']
        if not all(col in results.columns for col in required_columns):
            raise ValueError(f"结果数据必须包含以下列: {required_columns}")
            
        # 如果没有returns列，计算它
        if 'returns' not in results.columns:
            self.results['returns'] = self.results['total'].pct_change()
            
        # 如果未提供基准，使用零回报率（相当于现金基准）
        if benchmark_returns is None:
            self.benchmark_returns = pd.Series(0, index=results.index)
        else:
            self.benchmark_returns = benchmark_returns
            
        # 计算累积回报率
        if 'cumulative_returns' not in self.results.columns:
            self.results['cumulative_returns'] = (1 + self.results['returns']).cumprod() - 1
            
        # 计算性能指标
        self.metrics = self.calculate_metrics()
        
    def calculate_metrics(self):
        """
        计算各种性能指标
        
        Returns:
            dict: 包含各种性能指标的字典
        """
        # 提取回报率序列
        returns = self.results['returns'].dropna()
        
        # 基础指标
        total_return = self.results['total'].iloc[-1] / self.results['total'].iloc[0] - 1
        
        # 风险指标
        try:
            # 使用empyrical库计算各种金融指标
            annual_return = empyrical.annual_return(returns)
            annual_volatility = empyrical.annual_volatility(returns)
            sharpe_ratio = empyrical.sharpe_ratio(returns)
            sortino_ratio = empyrical.sortino_ratio(returns)
            calmar_ratio = empyrical.calmar_ratio(returns)
            max_drawdown = empyrical.max_drawdown(returns)
            omega_ratio = empyrical.omega_ratio(returns)
            
            # 统计回报率分析
            best_day = returns.max()
            worst_day = returns.min()
            
            # 盈亏比分析
            winning_days = returns[returns > 0]
            losing_days = returns[returns < 0]
            win_rate = len(winning_days) / len(returns) if len(returns) > 0 else 0
            
            avg_profit = winning_days.mean() if len(winning_days) > 0 else 0
            avg_loss = losing_days.mean() if len(losing_days) > 0 else 0
            profit_factor = abs(winning_days.sum() / losing_days.sum()) if losing_days.sum() != 0 else float('inf')
            
            # 波动率指标
            downside_risk = empyrical.downside_risk(returns)
            
            # 回撤分析
            cum_returns = (1 + returns).cumprod()
            running_max = cum_returns.cummax()
            drawdown = (cum_returns / running_max - 1)
            max_drawdown_value = drawdown.min()
            
            # 获取最大回撤期间
            end_idx = drawdown.idxmin()
            peak_idx = running_max.loc[:end_idx].idxmax()
            recovery_idx = None
            
            if end_idx < drawdown.index[-1]:
                recovery_period = drawdown.loc[end_idx:].ge(0).idxmax()
                if recovery_period != drawdown.loc[end_idx:].index[0]:  # 检查是否有恢复
                    recovery_idx = recovery_period
            
            drawdown_length = (end_idx - peak_idx).days
            recovery_length = (recovery_idx - end_idx).days if recovery_idx else None
            
            metrics = {
                "总回报率": total_return * 100,
                "年化回报率": annual_return * 100,
                "年化波动率": annual_volatility * 100,
                "夏普比率": sharpe_ratio,
                "索提诺比率": sortino_ratio,
                "卡玛比率": calmar_ratio,
                "最大回撤": max_drawdown * 100,
                "欧米伽比率": omega_ratio,
                "最佳单日回报": best_day * 100,
                "最差单日回报": worst_day * 100,
                "胜率": win_rate * 100,
                "平均盈利": avg_profit * 100,
                "平均亏损": avg_loss * 100,
                "盈亏比": profit_factor,
                "下行风险": downside_risk * 100,
                "最大回撤值": max_drawdown_value * 100,
                "最大回撤持续天数": drawdown_length,
                "回撤恢复天数": recovery_length
            }
            
            # 添加相对基准的指标
            if not self.benchmark_returns.equals(pd.Series(0, index=returns.index)):
                alpha = empyrical.alpha(returns, self.benchmark_returns)
                beta = empyrical.beta(returns, self.benchmark_returns)
                information_ratio = empyrical.excess_sharpe(returns, self.benchmark_returns)
                
                metrics.update({
                    "Alpha": alpha * 100,
                    "Beta": beta,
                    "信息比率": information_ratio
                })
                
            return metrics
            
        except Exception as e:
            print(f"计算性能指标时出错: {e}")
            return {
                "总回报率": total_return * 100
            }
            
    def plot_equity_curve(self, benchmark=True, log_scale=False):
        """
        绘制权益曲线
        
        Args:
            benchmark: 是否包含基准曲线
            log_scale: 是否使用对数刻度
        """
        plt.figure(figsize=(12, 6))
        
        # 绘制策略权益曲线
        plt.plot(self.results.index, self.results['total'], label='策略', linewidth=2)
        
        # 如果有基准且不是全零，则绘制基准曲线
        if benchmark and not np.all(self.benchmark_returns == 0):
            benchmark_equity = (1 + self.benchmark_returns).cumprod() * self.results['total'].iloc[0]
            plt.plot(self.results.index, benchmark_equity, label='基准', linewidth=2, alpha=0.7)
            
        plt.title('权益曲线')
        plt.xlabel('日期')
        plt.ylabel('资金')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        if log_scale:
            plt.yscale('log')
            
        # 格式化x轴日期
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.gcf().autofmt_xdate()
        
        # 格式化y轴为货币格式
        def currency_formatter(x, pos):
            return f"${x:,.0f}"
            
        plt.gca().yaxis.set_major_formatter(FuncFormatter(currency_formatter))
        
        plt.tight_layout()
        
    def plot_returns_distribution(self):
        """
        绘制回报率分布图
        """
        returns = self.results['returns'].dropna()
        
        plt.figure(figsize=(12, 6))
        
        # 绘制回报率分布直方图
        sns.histplot(returns, bins=50, kde=True)
        
        # 添加均值和中位数线条
        mean_return = returns.mean()
        median_return = returns.median()
        
        plt.axvline(mean_return, color='r', linestyle='--', linewidth=2, label=f'均值: {mean_return:.2%}')
        plt.axvline(median_return, color='g', linestyle='-.', linewidth=2, label=f'中位数: {median_return:.2%}')
        plt.axvline(0, color='k', linestyle='-', linewidth=1)
        
        plt.title('每日回报率分布')
        plt.xlabel('回报率')
        plt.ylabel('频率')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # 格式化x轴为百分比
        def percentage_formatter(x, pos):
            return f"{x:.1%}"
            
        plt.gca().xaxis.set_major_formatter(FuncFormatter(percentage_formatter))
        
        plt.tight_layout()
        
    def plot_drawdowns(self, top_n=5):
        """
        绘制回撤图
        
        Args:
            top_n: 显示的最大回撤数量
        """
        returns = self.results['returns'].dropna()
        
        # 计算回撤
        cum_returns = (1 + returns).cumprod()
        running_max = cum_returns.cummax()
        drawdown = (cum_returns / running_max - 1)
        
        plt.figure(figsize=(12, 6))
        
        # 绘制回撤曲线
        plt.plot(drawdown.index, drawdown * 100, linewidth=2)
        
        # 标记最大回撤
        max_drawdown_idx = drawdown.idxmin()
        plt.scatter(max_drawdown_idx, drawdown.min() * 100, color='red', s=100, 
                    label=f'最大回撤: {drawdown.min():.2%}', zorder=5)
        
        # 寻找前N个最大回撤
        if top_n > 1:
            # 创建回撤区间
            drawdown_periods = []
            in_drawdown = False
            start_idx = None
            
            for i, (date, value) in enumerate(drawdown.items()):
                if value < 0 and not in_drawdown:
                    in_drawdown = True
                    start_idx = date
                elif value >= 0 and in_drawdown:
                    in_drawdown = False
                    # 找出这个回撤区间内的最小值
                    period_min = drawdown.loc[start_idx:date].min()
                    period_min_idx = drawdown.loc[start_idx:date].idxmin()
                    drawdown_periods.append((start_idx, period_min_idx, date, period_min))
                    
            # 如果最后一个周期仍在回撤中
            if in_drawdown:
                period_min = drawdown.loc[start_idx:].min()
                period_min_idx = drawdown.loc[start_idx:].idxmin()
                drawdown_periods.append((start_idx, period_min_idx, drawdown.index[-1], period_min))
                
            # 根据回撤大小排序
            drawdown_periods.sort(key=lambda x: x[3])
            
            # 标记前N个最大回撤
            for i, (start, bottom, end, value) in enumerate(drawdown_periods[:top_n-1]):
                if i == 0:  # 跳过已经标记的最大回撤
                    continue
                plt.scatter(bottom, value * 100, color='orange', s=80, 
                            label=f'回撤 {i+1}: {value:.2%}', zorder=5)
        
        plt.title('回撤分析')
        plt.xlabel('日期')
        plt.ylabel('回撤 (%)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # 填充回撤区域
        plt.fill_between(drawdown.index, 0, drawdown * 100, color='red', alpha=0.3)
        
        # 格式化x轴日期
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.gcf().autofmt_xdate()
        
        plt.tight_layout()
        
    def plot_monthly_returns_heatmap(self):
        """
        绘制月度回报率热图
        """
        # 转换回报率为月度值
        returns = self.results['returns'].dropna()
        monthly_returns = returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
        
        # 创建月度回报率数据框
        monthly_return_table = pd.DataFrame({
            'Year': monthly_returns.index.year,
            'Month': monthly_returns.index.month,
            'Return': monthly_returns.values
        })
        
        # 数据透视表转换
        pivot_table = monthly_return_table.pivot_table(
            index='Year', 
            columns='Month', 
            values='Return'
        )
        
        # 填充缺失值
        pivot_table = pivot_table.fillna(0)
        
        # 设置月份名称
        month_names = ['一月', '二月', '三月', '四月', '五月', '六月', '七月', '八月', '九月', '十月', '十一月', '十二月']
        pivot_table.columns = month_names[:len(pivot_table.columns)]
        
        plt.figure(figsize=(14, 8))
        
        # 创建热图
        ax = sns.heatmap(
            pivot_table * 100, 
            annot=True, 
            fmt='.2f', 
            cmap='RdYlGn', 
            center=0, 
            cbar_kws={'label': '回报率 (%)'}
        )
        
        # 设置标题和标签
        plt.title('月度回报率热图 (%)')
        plt.ylabel('年份')
        plt.xlabel('月份')
        
        # 添加年度汇总列
        yearly_returns = pivot_table.sum(axis=1)
        yearly_text = [f"{year}: {ret:.2%}" for year, ret in yearly_returns.items()]
        yearly_summary = "\n".join(yearly_text)
        
        plt.figtext(
            0.92, 0.5, f"年度回报率:\n\n{yearly_summary}", 
            bbox=dict(facecolor='white', alpha=0.5, edgecolor='black', boxstyle='round,pad=0.5'),
            verticalalignment='center'
        )
        
        plt.tight_layout(rect=[0, 0, 0.85, 1])
        
    def plot_rolling_statistics(self, window=30):
        """
        绘制滚动统计数据
        
        Args:
            window: 滚动窗口大小(天数)
        """
        returns = self.results['returns'].dropna()
        
        # 计算滚动统计数据
        rolling_mean = returns.rolling(window=window).mean() * 252  # 年化
        rolling_vol = returns.rolling(window=window).std() * np.sqrt(252)  # 年化
        rolling_sharpe = rolling_mean / rolling_vol
        
        # 创建4个子图
        fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
        
        # 1. 滚动平均回报率
        axes[0].plot(rolling_mean.index, rolling_mean * 100, color='blue')
        axes[0].axhline(y=0, color='black', linestyle='-', linewidth=1)
        axes[0].set_title(f'{window}天滚动年化回报率')
        axes[0].set_ylabel('年化回报率 (%)')
        axes[0].grid(True, alpha=0.3)
        
        # 格式化y轴为百分比
        def percentage_formatter(x, pos):
            return f"{x:.0f}%"
            
        axes[0].yaxis.set_major_formatter(FuncFormatter(percentage_formatter))
        
        # 2. 滚动波动率
        axes[1].plot(rolling_vol.index, rolling_vol * 100, color='red')
        axes[1].set_title(f'{window}天滚动年化波动率')
        axes[1].set_ylabel('年化波动率 (%)')
        axes[1].grid(True, alpha=0.3)
        
        axes[1].yaxis.set_major_formatter(FuncFormatter(percentage_formatter))
        
        # 3. 滚动夏普比率
        axes[2].plot(rolling_sharpe.index, rolling_sharpe, color='green')
        axes[2].axhline(y=0, color='black', linestyle='-', linewidth=1)
        axes[2].axhline(y=1, color='red', linestyle='--', linewidth=1)
        axes[2].set_title(f'{window}天滚动夏普比率')
        axes[2].set_ylabel('夏普比率')
        axes[2].grid(True, alpha=0.3)
        
        # 格式化x轴日期
        for ax in axes:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
            
        fig.autofmt_xdate()
        plt.tight_layout()
        
    def generate_report(self, save_path=None, include_plots=True):
        """
        生成PDF报告
        
        Args:
            save_path: 报告保存路径
            include_plots: 是否包含图表
            
        Returns:
            str: 生成的报告文本
        """
        try:
            from fpdf import FPDF
            
            # 如果包含图表，先生成图表并保存为临时文件
            temp_files = []
            if include_plots:
                # 创建临时图表文件
                import tempfile
                import os
                
                # 创建图表
                self.plot_equity_curve()
                temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
                plt.savefig(temp_file.name)
                temp_files.append(temp_file.name)
                plt.close()
                
                self.plot_returns_distribution()
                temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
                plt.savefig(temp_file.name)
                temp_files.append(temp_file.name)
                plt.close()
                
                self.plot_drawdowns()
                temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
                plt.savefig(temp_file.name)
                temp_files.append(temp_file.name)
                plt.close()
                
                self.plot_monthly_returns_heatmap()
                temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
                plt.savefig(temp_file.name)
                temp_files.append(temp_file.name)
                plt.close()
                
                self.plot_rolling_statistics()
                temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
                plt.savefig(temp_file.name)
                temp_files.append(temp_file.name)
                plt.close()
                
            # 创建PDF
            pdf = FPDF()
            pdf.add_page()
            
            # 添加标题
            pdf.set_font('Arial', 'B', 16)
            pdf.cell(0, 10, '交易策略回测报告', 0, 1, 'C')
            pdf.ln(5)
            
            # 添加日期
            pdf.set_font('Arial', '', 10)
            pdf.cell(0, 10, f'报告生成日期: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', 0, 1)
            pdf.ln(5)
            
            # 添加性能指标
            pdf.set_font('Arial', 'B', 14)
            pdf.cell(0, 10, '性能指标', 0, 1)
            pdf.ln(2)
            
            pdf.set_font('Arial', '', 10)
            for key, value in self.metrics.items():
                if isinstance(value, float):
                    pdf.cell(0, 10, f'{key}: {value:.2f}', 0, 1)
                else:
                    pdf.cell(0, 10, f'{key}: {value}', 0, 1)
                    
            # 如果包含图表，添加图表
            if include_plots:
                for i, file in enumerate(temp_files):
                    pdf.add_page()
                    
                    if i == 0:
                        pdf.set_font('Arial', 'B', 14)
                        pdf.cell(0, 10, '权益曲线', 0, 1)
                    elif i == 1:
                        pdf.set_font('Arial', 'B', 14)
                        pdf.cell(0, 10, '回报率分布', 0, 1)
                    elif i == 2:
                        pdf.set_font('Arial', 'B', 14)
                        pdf.cell(0, 10, '回撤分析', 0, 1)
                    elif i == 3:
                        pdf.set_font('Arial', 'B', 14)
                        pdf.cell(0, 10, '月度回报率热图', 0, 1)
                    elif i == 4:
                        pdf.set_font('Arial', 'B', 14)
                        pdf.cell(0, 10, '滚动统计数据', 0, 1)
                        
                    pdf.ln(2)
                    pdf.image(file, x=10, y=None, w=190)
                    
            # 保存PDF
            if save_path:
                pdf.output(save_path)
                print(f"报告已保存至: {save_path}")
                
            # 清理临时文件
            for file in temp_files:
                os.unlink(file)
                
            # 返回文本形式的摘要
            report_text = "交易策略回测报告\n"
            report_text += "=" * 50 + "\n\n"
            report_text += f"报告生成日期: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            report_text += "性能指标:\n"
            report_text += "-" * 50 + "\n"
            
            for key, value in self.metrics.items():
                if isinstance(value, float):
                    report_text += f"{key}: {value:.2f}\n"
                else:
                    report_text += f"{key}: {value}\n"
                    
            return report_text
            
        except ImportError:
            # 如果没有fpdf，只返回文本报告
            report_text = "交易策略回测报告\n"
            report_text += "=" * 50 + "\n\n"
            report_text += f"报告生成日期: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            report_text += "性能指标:\n"
            report_text += "-" * 50 + "\n"
            
            for key, value in self.metrics.items():
                if isinstance(value, float):
                    report_text += f"{key}: {value:.2f}\n"
                else:
                    report_text += f"{key}: {value}\n"
                    
            print("提示: 安装fpdf库可以生成PDF报告")
            return report_text
            
        except Exception as e:
            print(f"生成报告时出错: {e}")
            return str(self.metrics)


# 使用示例
if __name__ == "__main__":
    # 创建一些模拟数据
    dates = pd.date_range(start='2023-01-01', periods=365, freq='D')
    
    # 模拟策略回报率
    np.random.seed(42)
    returns = np.random.normal(0.0005, 0.01, 365)  # 平均每日0.05%回报，1%标准差
    
    # 创建一个偏正的回报分布
    returns = returns + 0.0005  # 添加一个小的正偏移
    
    # 模拟每天的资产总值
    initial_value = 10000
    total = [initial_value]
    for r in returns:
        total.append(total[-1] * (1 + r))
    total = total[1:]  # 移除初始值，使长度匹配
    
    # 创建结果数据框
    results = pd.DataFrame({
        'returns': returns,
        'total': total
    }, index=dates)
    
    # 创建基准回报率（例如，股票指数）
    benchmark_returns = pd.Series(np.random.normal(0.0003, 0.008, 365), index=dates)
    
    # 初始化性能分析器
    analyzer = PerformanceAnalyzer(results, benchmark_returns)
    
    # 显示性能指标
    print("性能指标:")
    for key, value in analyzer.metrics.items():
        print(f"{key}: {value}")
        
    # 绘制分析图表
    analyzer.plot_equity_curve(benchmark=True)
    analyzer.plot_returns_distribution()
    analyzer.plot_drawdowns()
    analyzer.plot_monthly_returns_heatmap()
    analyzer.plot_rolling_statistics()
    
    # 生成报告
    report = analyzer.generate_report(save_path="backtest_report.pdf")
    print("\n报告摘要:")
    print(report)
    
    plt.show()
