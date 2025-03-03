from langchain_core.messages import SystemMessage
from src.agents.state import AgentState
from datetime import datetime
import os
import json
import logging

# 导入代理模块
from src.agents.market_data import market_data_agent
from src.agents.technicals import technical_analyst_agent
from src.agents.onchain_analysis import onchain_analysis_agent
from src.agents.sentiment import sentiment_agent
from src.agents.valuation import valuation_agent
from src.agents.researcher_bull import researcher_bull_agent
from src.agents.researcher_bear import researcher_bear_agent
from src.agents.debate_room import debate_room_agent
from src.agents.risk_manager import risk_management_agent
from src.agents.portfolio_manager import portfolio_management_agent
from src.agents.execution import execution_agent

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('workflow_controller')

class CryptoTradingWorkflow:
    """
    加密货币交易系统工作流控制器 - 协调各代理的执行顺序和数据流
    
    提供两种操作模式:
    1. 分析模式 (analysis): 仅执行分析，不进行实际交易
    2. 交易模式 (trading): 执行完整工作流，包括交易执行
    """
    
    def __init__(self, config=None):
        """
        初始化工作流控制器
        
        Args:
            config: 配置字典或配置文件路径
        """
        self.config = self._load_config(config)
        self.state = self._initialize_state()
        
    def _load_config(self, config):
        """
        加载配置
        
        Args:
            config: 配置字典或配置文件路径
            
        Returns:
            dict: 配置字典
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
            "results_dir": "results"
        }
        
        if config is None:
            return default_config
            
        # 如果是文件路径，加载JSON或YAML配置
        if isinstance(config, str):
            if os.path.exists(config):
                _, ext = os.path.splitext(config)
                if ext.lower() == '.json':
                    with open(config, 'r') as f:
                        loaded_config = json.load(f)
                elif ext.lower() in ['.yaml', '.yml']:
                    import yaml
                    with open(config, 'r') as f:
                        loaded_config = yaml.safe_load(f)
                else:
                    logger.warning(f"不支持的配置文件格式: {ext}，使用默认配置")
                    loaded_config = {}
            else:
                logger.warning(f"配置文件不存在: {config}，使用默认配置")
                loaded_config = {}
        else:
            # 直接使用传入的字典
            loaded_config = config
            
        # 合并配置
        merged_config = {**default_config, **loaded_config}
        return merged_config
        
    def _initialize_state(self):
        """
        初始化代理状态
        
        Returns:
            AgentState: 初始状态
        """
        # 提取配置参数
        symbol = self.config["symbol"]
        asset = symbol.split('/')[0]
        
        # 创建初始状态
        state = AgentState(
            messages=[
                SystemMessage(content=f"You are a crypto trading analysis system analyzing {symbol}.")
            ],
            data={
                "symbol": symbol,
                "asset": asset,
                "exchange_id": self.config["exchange_id"],
                "timeframe": self.config["timeframe"],
                "start_date": None,  # 由市场数据代理设置
                "end_date": None,    # 由市场数据代理设置
                "portfolio": self.config["portfolio"].copy(),
                "execution_mode": self.config["execution_mode"]
            },
            metadata={
                "show_reasoning": self.config.get("show_reasoning", False),
                "mode": self.config["mode"],
                "run_id": datetime.now().strftime("%Y%m%d_%H%M%S"),
                "save_results": self.config["save_results"],
                "results_dir": self.config["results_dir"]
            }
        )
        
        return state
        
    def run_analysis_workflow(self):
        """
        运行分析工作流 (不包括交易执行)
        
        Returns:
            AgentState: 结束状态
        """
        logger.info("开始运行分析工作流")
        
        # 1. 市场数据收集阶段
        logger.info("阶段 1/9: 市场数据收集")
        self.state = market_data_agent(self.state)
        
        # 2. 技术分析阶段
        logger.info("阶段 2/9: 技术分析")
        self.state = technical_analyst_agent(self.state)
        
        # 3. 链上分析阶段
        logger.info("阶段 3/9: 链上分析")
        self.state = onchain_analysis_agent(self.state)
        
        # 4. 情绪分析阶段
        logger.info("阶段 4/9: 情绪分析")
        self.state = sentiment_agent(self.state)
        
        # 5. 估值分析阶段
        logger.info("阶段 5/9: 估值分析")
        self.state = valuation_agent(self.state)
        
        # 6. 看多研究阶段
        logger.info("阶段 6/9: 看多研究")
        self.state = researcher_bull_agent(self.state)
        
        # 7. 看空研究阶段
        logger.info("阶段 7/9: 看空研究")
        self.state = researcher_bear_agent(self.state)
        
        # 8. 辩论总结阶段
        logger.info("阶段 8/9: 辩论总结")
        self.state = debate_room_agent(self.state)
        
        # 9. 风险管理阶段
        logger.info("阶段 9/9: 风险管理")
        self.state = risk_management_agent(self.state)
        
        # 保存结果
        if self.state.get("metadata", {}).get("save_results", False):
            self._save_results("analysis")
            
        logger.info("分析工作流完成")
        return self.state
        
    def run_trading_workflow(self):
        """
        运行完整交易工作流 (包括交易执行)
        
        Returns:
            AgentState: 结束状态
        """
        # 先运行分析工作流
        self.run_analysis_workflow()
        
        # 然后执行交易决策和执行阶段
        logger.info("阶段 10/11: 投资组合管理")
        self.state = portfolio_management_agent(self.state)
        
        logger.info("阶段 11/11: 交易执行")
        self.state = execution_agent(self.state)
        
        # 保存交易结果
        if self.state["metadata"]["save_results"]:
            self._save_results("trading")
            
        logger.info("交易工作流完成")
        return self.state
        
    def run(self):
        """
        运行工作流 (根据配置选择分析或交易模式)
        
        Returns:
            AgentState: 结束状态
        """
        if self.config["mode"] == "trading":
            return self.run_trading_workflow()
        else:
            return self.run_analysis_workflow()
            
    def get_analysis_summary(self):
        """
        获取分析摘要
        
        Returns:
            dict: 分析摘要
        """
        # 检查是否已运行分析
        if not any(msg.name == "debate_room_agent" for msg in self.state["messages"]):
            return {"error": "尚未运行分析，请先调用run_analysis_workflow()"}
            
        # 收集各代理的关键信息
        summary = {
            "symbol": self.state["data"]["symbol"],
            "asset": self.state["data"]["asset"],
            "run_time": datetime.now().isoformat()
        }
        
        # 添加市场数据
        if "market_data" in self.state["data"]:
            market_data = self.state["data"]["market_data"]
            summary["price"] = market_data.get("price", 0)
            summary["volume_24h"] = market_data.get("volume_24h", 0)
            summary["change_24h_percent"] = market_data.get("change_24h_percent", 0)
            
        # 添加技术分析
        technical_message = next((msg for msg in self.state["messages"] if msg.name == "technical_analyst_agent"), None)
        if technical_message:
            try:
                technical_content = json.loads(technical_message.content)
                summary["technical_analysis"] = {
                    "signal": technical_content.get("signal", "neutral"),
                    "confidence": technical_content.get("confidence", "50%")
                }
            except:
                pass
                
        # 添加链上分析
        onchain_message = next((msg for msg in self.state["messages"] if msg.name == "onchain_analysis_agent"), None)
        if onchain_message:
            try:
                onchain_content = json.loads(onchain_message.content)
                summary["onchain_analysis"] = {
                    "signal": onchain_content.get("signal", "neutral"),
                    "confidence": onchain_content.get("confidence", "50%")
                }
            except:
                pass
                
        # 添加估值分析
        valuation_message = next((msg for msg in self.state["messages"] if msg.name == "valuation_agent"), None)
        if valuation_message:
            try:
                valuation_content = json.loads(valuation_message.content)
                summary["valuation_analysis"] = {
                    "signal": valuation_content.get("signal", "neutral"),
                    "confidence": valuation_content.get("confidence", "50%")
                }
            except:
                pass
                
        # 添加情绪分析
        sentiment_message = next((msg for msg in self.state["messages"] if msg.name == "sentiment_agent"), None)
        if sentiment_message:
            try:
                sentiment_content = json.loads(sentiment_message.content)
                summary["sentiment_analysis"] = {
                    "signal": sentiment_content.get("signal", "neutral"),
                    "confidence": sentiment_content.get("confidence", "50%")
                }
            except:
                pass
                
        # 添加辩论结果
        debate_message = next((msg for msg in self.state["messages"] if msg.name == "debate_room_agent"), None)
        if debate_message:
            try:
                debate_content = json.loads(debate_message.content)
                summary["debate_conclusion"] = {
                    "signal": debate_content.get("signal", "neutral"),
                    "confidence": debate_content.get("confidence", 0.5),
                    "bull_confidence": debate_content.get("bull_confidence", 0.5),
                    "bear_confidence": debate_content.get("bear_confidence", 0.5),
                    "reasoning": debate_content.get("reasoning", "")
                }
            except:
                pass
                
        # 添加风险分析
        risk_message = next((msg for msg in self.state["messages"] if msg.name == "risk_management_agent"), None)
        if risk_message:
            try:
                risk_content = json.loads(risk_message.content)
                summary["risk_analysis"] = {
                    "risk_score": risk_content.get("risk_score", 5),
                    "trading_action": risk_content.get("trading_action", "hold"),
                    "max_position_size": risk_content.get("max_position_size", 0)
                }
            except:
                pass
                
        return summary
        
    def get_trading_summary(self):
        """
        获取交易摘要
        
        Returns:
            dict: 交易摘要
        """
        # 检查是否已运行交易
        if not any(msg.name == "portfolio_management" for msg in self.state["messages"]):
            return {"error": "尚未运行交易工作流，请先调用run_trading_workflow()"}
            
        # 获取分析摘要
        summary = self.get_analysis_summary()
        
        # 添加投资组合管理决策
        pm_message = next((msg for msg in self.state["messages"] if msg.name == "portfolio_management"), None)
        if pm_message:
            try:
                pm_content = json.loads(pm_message.content)
                summary["portfolio_decision"] = {
                    "action": pm_content.get("action", "hold"),
                    "quantity": pm_content.get("quantity", 0),
                    "confidence": pm_content.get("confidence", 0.5),
                    "reasoning": pm_content.get("reasoning", "")
                }
            except:
                pass
                
        # 添加交易执行结果
        exec_message = next((msg for msg in self.state["messages"] if msg.name == "execution_agent"), None)
        if exec_message:
            try:
                exec_content = json.loads(exec_message.content)
                summary["execution_result"] = {
                    "status": exec_content.get("status", "unknown"),
                    "message": exec_content.get("message", ""),
                    "price": exec_content.get("price", 0),
                    "execution_time": exec_content.get("execution_time", "")
                }
                
                # 添加当前投资组合状态
                summary["current_portfolio"] = exec_content.get("portfolio", {})
            except:
                pass
                
        return summary
        
    def _save_results(self, result_type):
        """
        保存结果到文件
        
        Args:
            result_type: 结果类型 ('analysis' 或 'trading')
        """
        # 创建结果目录
        results_dir = self.state["metadata"]["results_dir"]
        os.makedirs(results_dir, exist_ok=True)
        
        # 获取运行ID
        run_id = self.state["metadata"]["run_id"]
        
        # 获取摘要
        if result_type == "analysis":
            summary = self.get_analysis_summary()
        else:
            summary = self.get_trading_summary()
            
        # 保存摘要
        filename = f"{results_dir}/{run_id}_{result_type}_summary.json"
        with open(filename, 'w') as f:
            json.dump(summary, f, indent=2)
            
        logger.info(f"结果已保存到 {filename}")
        
        # 保存完整状态（可选，用于调试）
        # 注意：状态可能很大，包含所有消息和数据
        if self.config.get("save_full_state", False):
            # 将状态转换为可序列化的格式
            serializable_state = {
                "data": self.state["data"],
                "metadata": self.state["metadata"],
                "messages": [
                    {
                        "name": msg.name,
                        "content": msg.content,
                        "type": type(msg).__name__
                    }
                    for msg in self.state["messages"]
                ]
            }
            
            state_filename = f"{results_dir}/{run_id}_{result_type}_full_state.json"
            with open(state_filename, 'w') as f:
                json.dump(serializable_state, f, indent=2)
                
            logger.info(f"完整状态已保存到 {state_filename}")
