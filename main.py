#!/usr/bin/env python3
import os
import sys
import json
import argparse
import logging
import time
from datetime import datetime
import yaml
from pathlib import Path

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

from workflow.controller import CryptoTradingWorkflow

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(f"logs/crypto_agent_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    ]
)
logger = logging.getLogger('crypto_agent')

def load_config(config_path):
    """
    加载配置文件
    
    Args:
        config_path: 配置文件路径
    
    Returns:
        dict: 配置字典
    """
    if not os.path.exists(config_path):
        logger.error(f"配置文件不存在: {config_path}")
        sys.exit(1)
        
    _, ext = os.path.splitext(config_path)
    if ext.lower() == '.json':
        with open(config_path, 'r') as f:
            return json.load(f)
    elif ext.lower() in ['.yaml', '.yml']:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    else:
        logger.error(f"不支持的配置文件格式: {ext}")
        sys.exit(1)

def generate_default_config(output_path):
    """
    生成默认配置文件
    
    Args:
        output_path: 输出文件路径
    """
    default_config = {
        "mode": "analysis",  # 'analysis' 或 'trading'
        "symbol": "BTC/USDT",
        "exchange_id": "binance",
        "timeframe": "1d",
        "portfolio": {
            "cash": 10000,
            "stock": 0
        },
        "execution_mode": "simulation",  # 'simulation', 'paper', 'live'
        "show_reasoning": True,
        "save_results": True,
        "results_dir": "results",
        "api_keys": {
            "binance": {
                "api_key": "YOUR_API_KEY_HERE",
                "api_secret": "YOUR_API_SECRET_HERE"
            },
            "twitter": {
                "bearer_token": "YOUR_TWITTER_BEARER_TOKEN"
            }
        },
        "notification": {
            "enabled": False,
            "telegram": {
                "bot_token": "YOUR_TELEGRAM_BOT_TOKEN",
                "chat_id": "YOUR_TELEGRAM_CHAT_ID"
            },
            "email": {
                "enabled": False,
                "smtp_server": "smtp.gmail.com",
                "smtp_port": 587,
                "sender_email": "your_email@gmail.com",
                "receiver_email": "your_email@gmail.com",
                "password": "YOUR_EMAIL_PASSWORD"
            }
        },
        "backtesting": {
            "start_date": "2023-01-01",
            "end_date": "2023-12-31"
        }
    }
    
    # 确保目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 保存配置
    _, ext = os.path.splitext(output_path)
    if ext.lower() == '.json':
        with open(output_path, 'w') as f:
            json.dump(default_config, f, indent=2)
    elif ext.lower() in ['.yaml', '.yml']:
        with open(output_path, 'w') as f:
            yaml.dump(default_config, f, sort_keys=False)
    else:
        logger.error(f"不支持的配置文件格式: {ext}")
        sys.exit(1)
        
    logger.info(f"默认配置已保存到: {output_path}")

def setup_environment():
    """
    设置环境，创建必要的目录
    """
    directories = ['logs', 'results', 'data', 'configs']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        
    logger.info("环境设置完成")

def print_summary(summary):
    """
    打印分析或交易摘要
    
    Args:
        summary: 摘要字典
    """
    print("\n" + "="*50)
    print(f"加密货币交易代理 - {summary.get('symbol', 'Unknown')} 摘要")
    print("="*50)
    
    # 打印价格信息
    print(f"价格: ${summary.get('price', 0):.2f}")
    print(f"24小时变化: {summary.get('change_24h_percent', 0):.2f}%")
    print(f"24小时成交量: ${summary.get('volume_24h', 0):,.0f}")
    
    print("\n分析结果:")
    
    # 打印技术分析
    if 'technical_analysis' in summary:
        tech = summary['technical_analysis']
        print(f"技术分析: {tech['signal'].upper()} (信心: {tech['confidence']})")
        
    # 打印链上分析
    if 'onchain_analysis' in summary:
        onchain = summary['onchain_analysis']
        print(f"链上分析: {onchain['signal'].upper()} (信心: {onchain['confidence']})")
        
    # 打印估值分析
    if 'valuation_analysis' in summary:
        valuation = summary['valuation_analysis']
        print(f"估值分析: {valuation['signal'].upper()} (信心: {valuation['confidence']})")
        
    # 打印情绪分析
    if 'sentiment_analysis' in summary:
        sentiment = summary['sentiment_analysis']
        print(f"情绪分析: {sentiment['signal'].upper()} (信心: {sentiment['confidence']})")
        
    # 打印辩论结果
    if 'debate_conclusion' in summary:
        debate = summary['debate_conclusion']
        print(f"\n最终结论: {debate['signal'].upper()} (信心: {debate['confidence']:.2f})")
        print(f"看多信心: {debate['bull_confidence']:.2f}, 看空信心: {debate['bear_confidence']:.2f}")
        print(f"分析: {debate['reasoning']}")
        
    # 打印风险分析
    if 'risk_analysis' in summary:
        risk = summary['risk_analysis']
        print(f"\n风险评分: {risk['risk_score']}/10")
        print(f"建议操作: {risk['trading_action'].upper()}")
        print(f"最大仓位: ${risk['max_position_size']:,.2f}")
        
    # 如果包含交易决策，打印交易摘要
    if 'portfolio_decision' in summary:
        decision = summary['portfolio_decision']
        print("\n" + "-"*50)
        print("交易决策:")
        print(f"操作: {decision['action'].upper()}")
        print(f"数量: {decision['quantity']}")
        print(f"信心: {decision['confidence']:.2f}")
        print(f"理由: {decision['reasoning']}")
        
    # 如果包含执行结果，打印执行摘要
    if 'execution_result' in summary:
        execution = summary['execution_result']
        print("\n" + "-"*50)
        print("交易执行:")
        print(f"状态: {execution['status'].upper()}")
        print(f"价格: ${execution['price']:.2f}")
        print(f"执行时间: {execution['execution_time']}")
        print(f"消息: {execution['message']}")
        
    # 如果包含当前投资组合状态，打印投资组合摘要
    if 'current_portfolio' in summary:
        portfolio = summary['current_portfolio']
        print("\n" + "-"*50)
        print("当前投资组合:")
        print(f"现金: ${portfolio.get('cash', 0):,.2f}")
        print(f"持仓: {portfolio.get('stock', 0)}")
        
        # 计算总价值
        if 'price' in summary:
            total_value = portfolio.get('cash', 0) + (portfolio.get('stock', 0) * summary['price'])
            print(f"总价值: ${total_value:,.2f}")
            
    print("\n" + "="*50)

def main():
    """
    主函数
    """
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='加密货币交易代理系统')
    parser.add_argument('--config', type=str, help='配置文件路径')
    parser.add_argument('--mode', choices=['analysis', 'trading', 'backtest'], help='运行模式')
    parser.add_argument('--symbol', type=str, help='交易对 (例如 BTC/USDT)')
    parser.add_argument('--generate-config', type=str, help='生成默认配置文件并退出')
    parser.add_argument('--setup', action='store_true', help='设置环境并退出')
    
    args = parser.parse_args()
    
    # 设置环境
    if args.setup:
        setup_environment()
        sys.exit(0)
        
    # 生成默认配置
    if args.generate_config:
        generate_default_config(args.generate_config)
        sys.exit(0)
        
    # 确保日志目录存在
    os.makedirs('logs', exist_ok=True)
    
    # 加载配置
    config = {}
    if args.config:
        config = load_config(args.config)
        
    # 命令行参数覆盖配置文件
    if args.mode:
        config['mode'] = args.mode
    if args.symbol:
        config['symbol'] = args.symbol
        
    # 初始化工作流
    workflow = CryptoTradingWorkflow(config)
    
    # 运行工作流
    start_time = time.time()
    
    if config.get('mode') == 'trading':
        logger.info("开始交易工作流")
        workflow.run_trading_workflow()
        summary = workflow.get_trading_summary()
    elif config.get('mode') == 'backtest':
        logger.info("开始回测")
        # 回测逻辑在这里实现或调用回测模块
        sys.exit(0)
    else:
        logger.info("开始分析工作流")
        workflow.run_analysis_workflow()
        summary = workflow.get_analysis_summary()
        
    end_time = time.time()
    
    # 打印摘要
    print_summary(summary)
    
    # 打印执行时间
    logger.info(f"工作流执行完成，耗时: {end_time - start_time:.2f} 秒")

if __name__ == "__main__":
    main()
