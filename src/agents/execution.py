from langchain_core.messages import HumanMessage
from src.agents.state import AgentState, show_agent_reasoning, show_workflow_status
from src.tools.crypto_api import initialize_exchange, execute_trade, get_account_balance
import json
import time
import pandas as pd
from datetime import datetime
import logging

# 设置日志
logger = logging.getLogger('execution_agent')

class ExecutionAgent:
    """
    交易执行代理 - 负责执行交易决策并记录交易历史
    
    功能:
    1. 接收投资组合管理代理的交易决策
    2. 执行实际交易或模拟交易
    3. 记录交易历史和投资组合状态
    4. 发送交易通知
    """
    
    def __init__(self):
        """初始化执行代理"""
        self.exchange = None
        self.symbol = None
        self.portfolio = None
        self.mode = "simulation"  # 默认为模拟模式
        self.trade_history_file = "trade_history.csv"
        
    def execute(self, state: AgentState):
        """执行交易决策"""
        show_workflow_status("交易执行代理")
        if "metadata" in state and "show_reasoning" in state.get("metadata", {}):
            show_reasoning = state["metadata"]["show_reasoning"]
        else:
            show_reasoning = False
        data = state["data"]
        
        # 获取设置
        self.symbol = data.get("symbol", "BTC/USDT")
        self.portfolio = data.get("portfolio", {"cash": 10000, "stock": 0})
        self.mode = data.get("execution_mode", "simulation")
        
        # 获取投资组合管理代理的消息
        pm_message = None
        
        for message in state["messages"]:
            if message.name == "portfolio_management":
                pm_message = message
                break
                
        # 如果没有投资组合管理决策，返回无操作消息
        if not pm_message:
            message_content = {
                "status": "no_action",
                "message": "无交易决策可执行",
                "execution_time": datetime.now().isoformat(),
                "portfolio": self.portfolio
            }
            
            message = HumanMessage(
                content=json.dumps(message_content),
                name="execution_agent",
            )
            
            if show_reasoning:
                show_agent_reasoning(message_content, "交易执行代理")
                
            show_workflow_status("交易执行代理", "completed")
            return {
                "messages": state["messages"] + [message],
                "data": data
            }
            
        # 解析投资组合管理决策
        try:
            pm_decision = json.loads(pm_message.content)
        except Exception as e:
            logger.error(f"解析投资组合管理决策时出错: {e}")
            message_content = {
                "status": "error",
                "message": f"解析交易决策时出错: {e}",
                "execution_time": datetime.now().isoformat(),
                "portfolio": self.portfolio
            }
            
            message = HumanMessage(
                content=json.dumps(message_content),
                name="execution_agent",
            )
            
            if show_reasoning:
                show_agent_reasoning(message_content, "交易执行代理")
                
            show_workflow_status("交易执行代理", "completed")
            return {
                "messages": state["messages"] + [message],
                "data": data
            }
            
        # 获取交易决策参数
        action = pm_decision.get("action", "hold")
        quantity = pm_decision.get("quantity", 0)
        
        # 如果是持有决策或数量为0，返回无操作消息
        if action == "hold" or quantity <= 0:
            message_content = {
                "status": "no_trade",
                "message": f"根据投资组合管理决策，维持当前持仓，不执行交易",
                "action": action,
                "quantity": quantity,
                "execution_time": datetime.now().isoformat(),
                "portfolio": self.portfolio
            }
            
            message = HumanMessage(
                content=json.dumps(message_content),
                name="execution_agent",
            )
            
            if show_reasoning:
                show_agent_reasoning(message_content, "交易执行代理")
                
            show_workflow_status("交易执行代理", "completed")
            return {
                "messages": state["messages"] + [message],
                "data": data
            }
            
        # 执行交易
        execution_result = None
        
        # 获取当前市场价格
        current_price = 0
        
        # 从市场数据获取价格
        if "market_data" in data and "price" in data["market_data"]:
            current_price = data["market_data"]["price"]
        # 从价格历史获取价格
        elif "prices" in data and len(data["prices"]) > 0:
            prices = data["prices"]
            if isinstance(prices, list) and len(prices) > 0:
                if isinstance(prices[-1], dict) and "close" in prices[-1]:
                    current_price = prices[-1]["close"]
                    
        # 如果仍然无法获取价格，使用默认价格
        if current_price <= 0:
            asset = self.symbol.split('/')[0]
            if asset == "BTC":
                current_price = 50000
            elif asset == "ETH":
                current_price = 3000
            else:
                current_price = 100
                
            logger.warning(f"无法获取{self.symbol}的当前价格，使用默认价格 ${current_price}")
            
        # 根据操作模式执行交易
        if self.mode == "live":
            # 真实交易模式
            execution_result = self._execute_live_trade(action, quantity, current_price)
        elif self.mode == "paper":
            # 模拟账户交易模式
            execution_result = self._execute_paper_trade(action, quantity, current_price)
        else:
            # 默认模拟交易模式
            execution_result = self._execute_simulation_trade(action, quantity, current_price)
            
        # 记录交易历史
        if execution_result["status"] == "success":
            self._record_trade_history(execution_result)
            
        # 更新投资组合状态
        data["portfolio"] = self.portfolio
        
        # 创建执行代理消息
        message = HumanMessage(
            content=json.dumps(execution_result),
            name="execution_agent",
        )
        
        if show_reasoning:
            show_agent_reasoning(execution_result, "交易执行代理")
            
        show_workflow_status("交易执行代理", "completed")
        return {
            "messages": state["messages"] + [message],
            "data": data
        }
        
    def _execute_live_trade(self, action, quantity, price):
        """
        在实盘环境执行交易
        
        Args:
            action: 交易动作 ('buy' 或 'sell')
            quantity: 交易数量
            price: 当前价格 (用于日志记录)
            
        Returns:
            dict: 交易执行结果
        """
        # 初始化交易所连接
        if not self.exchange:
            self.exchange = initialize_exchange()
            
        if not self.exchange:
            return {
                "status": "error",
                "message": "无法连接到交易所，交易失败",
                "action": action,
                "quantity": quantity,
                "price": price,
                "execution_time": datetime.now().isoformat(),
                "portfolio": self.portfolio
            }
            
        # 执行交易
        try:
            # 使用市价单执行交易
            order_result = execute_trade(
                self.exchange, 
                self.symbol, 
                order_type="market", 
                side=action, 
                amount=quantity
            )
            
            if order_result and "error" not in order_result:
                # 交易成功
                # 获取实际成交价格和数量
                executed_price = order_result.get("price", price)
                executed_amount = order_result.get("amount", quantity)
                order_id = order_result.get("id", "unknown")
                fee = order_result.get("fee", {"cost": 0, "currency": "USDT"})
                
                # 更新投资组合
                if action == "buy":
                    cost = executed_amount * executed_price + fee["cost"]
                    self.portfolio["cash"] -= cost
                    self.portfolio["stock"] += executed_amount
                else:  # sell
                    proceeds = executed_amount * executed_price - fee["cost"]
                    self.portfolio["cash"] += proceeds
                    self.portfolio["stock"] -= executed_amount
                    
                return {
                    "status": "success",
                    "message": f"实盘交易执行成功: {action} {executed_amount} @ ${executed_price}",
                    "action": action,
                    "quantity": executed_amount,
                    "price": executed_price,
                    "order_id": order_id,
                    "fee": fee["cost"],
                    "execution_time": datetime.now().isoformat(),
                    "portfolio": self.portfolio
                }
            else:
                # 交易失败
                error_message = order_result.get("error", "未知错误") if order_result else "交易执行失败"
                
                return {
                    "status": "error",
                    "message": f"实盘交易执行失败: {error_message}",
                    "action": action,
                    "quantity": quantity,
                    "price": price,
                    "execution_time": datetime.now().isoformat(),
                    "portfolio": self.portfolio
                }
                
        except Exception as e:
            logger.error(f"执行实盘交易时出错: {e}")
            
            return {
                "status": "error",
                "message": f"执行实盘交易时出错: {e}",
                "action": action,
                "quantity": quantity,
                "price": price,
                "execution_time": datetime.now().isoformat(),
                "portfolio": self.portfolio
            }
            
    def _execute_paper_trade(self, action, quantity, price):
        """
        在模拟账户上执行交易（使用交易所的模拟交易功能）
        
        Args:
            action: 交易动作 ('buy' 或 'sell')
            quantity: 交易数量
            price: 当前价格
            
        Returns:
            dict: 交易执行结果
        """
        # 初始化交易所连接（使用测试模式）
        if not self.exchange:
            self.exchange = initialize_exchange(test_mode=True)
            
        if not self.exchange:
            return {
                "status": "error",
                "message": "无法连接到交易所（模拟交易模式），交易失败",
                "action": action,
                "quantity": quantity,
                "price": price,
                "execution_time": datetime.now().isoformat(),
                "portfolio": self.portfolio
            }
            
        # 执行模拟交易
        try:
            # 使用市价单执行交易
            order_result = execute_trade(
                self.exchange, 
                self.symbol, 
                order_type="market", 
                side=action, 
                amount=quantity
            )
            
            if order_result and "error" not in order_result:
                # 交易成功
                # 获取实际成交价格和数量
                executed_price = order_result.get("price", price)
                executed_amount = order_result.get("amount", quantity)
                order_id = order_result.get("id", "unknown")
                fee = order_result.get("fee", {"cost": 0, "currency": "USDT"})
                
                # 更新投资组合
                if action == "buy":
                    cost = executed_amount * executed_price + fee["cost"]
                    self.portfolio["cash"] -= cost
                    self.portfolio["stock"] += executed_amount
                else:  # sell
                    proceeds = executed_amount * executed_price - fee["cost"]
                    self.portfolio["cash"] += proceeds
                    self.portfolio["stock"] -= executed_amount
                    
                return {
                    "status": "success",
                    "message": f"模拟账户交易执行成功: {action} {executed_amount} @ ${executed_price}",
                    "action": action,
                    "quantity": executed_amount,
                    "price": executed_price,
                    "order_id": order_id,
                    "fee": fee["cost"],
                    "execution_time": datetime.now().isoformat(),
                    "portfolio": self.portfolio
                }
            else:
                # 交易失败
                error_message = order_result.get("error", "未知错误") if order_result else "交易执行失败"
                
                return {
                    "status": "error",
                    "message": f"模拟账户交易执行失败: {error_message}",
                    "action": action,
                    "quantity": quantity,
                    "price": price,
                    "execution_time": datetime.now().isoformat(),
                    "portfolio": self.portfolio
                }
                
        except Exception as e:
            logger.error(f"执行模拟账户交易时出错: {e}")
            
            return {
                "status": "error",
                "message": f"执行模拟账户交易时出错: {e}",
                "action": action,
                "quantity": quantity,
                "price": price,
                "execution_time": datetime.now().isoformat(),
                "portfolio": self.portfolio
            }
            
    def _execute_simulation_trade(self, action, quantity, price):
        """
        在软件层面模拟执行交易（不连接交易所）
        
        Args:
            action: 交易动作 ('buy' 或 'sell')
            quantity: 交易数量
            price: 当前价格
            
        Returns:
            dict: 交易执行结果
        """
        # 检查交易是否可行
        if action == "buy":
            # 计算交易成本
            fee_rate = 0.001  # 0.1%的交易手续费
            cost = quantity * price * (1 + fee_rate)
            
            # 检查现金是否足够
            if cost > self.portfolio["cash"]:
                # 调整为可负担的最大数量
                max_quantity = self.portfolio["cash"] / (price * (1 + fee_rate))
                quantity = max_quantity * 0.99  # 留一点余量
                cost = quantity * price * (1 + fee_rate)
                
                if quantity <= 0:
                    return {
                        "status": "error",
                        "message": "模拟交易失败: 现金不足，无法执行买入交易",
                        "action": action,
                        "quantity": quantity,
                        "price": price,
                        "execution_time": datetime.now().isoformat(),
                        "portfolio": self.portfolio
                    }
                    
            # 执行买入
            self.portfolio["cash"] -= cost
            self.portfolio["stock"] += quantity
            fee = quantity * price * fee_rate
            
            return {
                "status": "success",
                "message": f"模拟交易执行成功: 买入 {quantity:.8f} @ ${price:.2f}",
                "action": "buy",
                "quantity": quantity,
                "price": price,
                "fee": fee,
                "execution_time": datetime.now().isoformat(),
                "portfolio": self.portfolio
            }
            
        elif action == "sell":
            # 检查持仓是否足够
            if quantity > self.portfolio["stock"]:
                quantity = self.portfolio["stock"]
                
                if quantity <= 0:
                    return {
                        "status": "error",
                        "message": "模拟交易失败: 持仓不足，无法执行卖出交易",
                        "action": action,
                        "quantity": 0,
                        "price": price,
                        "execution_time": datetime.now().isoformat(),
                        "portfolio": self.portfolio
                    }
                    
            # 计算交易收益
            fee_rate = 0.001  # 0.1%的交易手续费
            fee = quantity * price * fee_rate
            proceeds = quantity * price - fee
            
            # 执行卖出
            self.portfolio["cash"] += proceeds
            self.portfolio["stock"] -= quantity
            
            return {
                "status": "success",
                "message": f"模拟交易执行成功: 卖出 {quantity:.8f} @ ${price:.2f}",
                "action": "sell",
                "quantity": quantity,
                "price": price,
                "fee": fee,
                "execution_time": datetime.now().isoformat(),
                "portfolio": self.portfolio
            }
            
        else:
            return {
                "status": "error",
                "message": f"模拟交易失败: 不支持的交易动作 '{action}'",
                "action": action,
                "quantity": quantity,
                "price": price,
                "execution_time": datetime.now().isoformat(),
                "portfolio": self.portfolio
            }
            
    def _record_trade_history(self, trade_result):
        """
        记录交易历史到CSV文件
        
        Args:
            trade_result: 交易执行结果
        """
        try:
            # 准备交易记录
            trade_record = {
                "timestamp": trade_result["execution_time"],
                "symbol": self.symbol,
                "action": trade_result["action"],
                "quantity": trade_result["quantity"],
                "price": trade_result["price"],
                "fee": trade_result.get("fee", 0),
                "total_value": trade_result["quantity"] * trade_result["price"],
                "cash_after": self.portfolio["cash"],
                "stock_after": self.portfolio["stock"],
                "portfolio_value": self.portfolio["cash"] + (self.portfolio["stock"] * trade_result["price"])
            }
            
            # 检查交易历史文件是否存在
            try:
                trade_history = pd.read_csv(self.trade_history_file)
                # 追加新记录
                trade_history = pd.concat([trade_history, pd.DataFrame([trade_record])], ignore_index=True)
            except (FileNotFoundError, pd.errors.EmptyDataError):
                # 创建新的交易历史
                trade_history = pd.DataFrame([trade_record])
                
            # 保存交易历史
            trade_history.to_csv(self.trade_history_file, index=False)
            logger.info(f"交易记录已保存到 {self.trade_history_file}")
            
        except Exception as e:
            logger.error(f"记录交易历史时出错: {e}")


def execution_agent(state: AgentState):
    """交易执行代理入口函数"""
    agent = ExecutionAgent()
    return agent.execute(state)
