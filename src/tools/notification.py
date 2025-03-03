import os
import smtplib
import requests
import logging
import json
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime

logger = logging.getLogger('notification')

class NotificationManager:
    """
    通知管理器 - 负责发送交易通知
    
    支持的通知渠道:
    - Telegram
    - Email
    - Discord Webhook
    """
    
    def __init__(self, config=None):
        """
        初始化通知管理器
        
        Args:
            config: 通知配置
        """
        self.config = config or {}
        self.enabled = self.config.get('enabled', False)
        
        if not self.enabled:
            logger.info("通知系统已禁用")
            return
            
        # 加载Telegram配置
        self.telegram_enabled = False
        if 'telegram' in self.config:
            telegram_config = self.config['telegram']
            self.telegram_bot_token = telegram_config.get('bot_token')
            self.telegram_chat_id = telegram_config.get('chat_id')
            
            if self.telegram_bot_token and self.telegram_chat_id:
                self.telegram_enabled = True
                logger.info("Telegram通知已启用")
                
        # 加载Email配置
        self.email_enabled = False
        if 'email' in self.config and self.config['email'].get('enabled', False):
            email_config = self.config['email']
            self.smtp_server = email_config.get('smtp_server')
            self.smtp_port = email_config.get('smtp_port', 587)
            self.sender_email = email_config.get('sender_email')
            self.receiver_email = email_config.get('receiver_email')
            self.email_password = email_config.get('password')
            
            if all([self.smtp_server, self.sender_email, self.receiver_email, self.email_password]):
                self.email_enabled = True
                logger.info("Email通知已启用")
                
        # 加载Discord配置
        self.discord_enabled = False
        if 'discord' in self.config and self.config['discord'].get('enabled', False):
            discord_config = self.config['discord']
            self.discord_webhook_url = discord_config.get('webhook_url')
            
            if self.discord_webhook_url:
                self.discord_enabled = True
                logger.info("Discord通知已启用")
                
    def send_notification(self, title, message, trade_data=None):
        """
        发送通知
        
        Args:
            title: 通知标题
            message: 通知内容
            trade_data: 交易数据（可选）
        
        Returns:
            bool: 是否成功发送通知
        """
        if not self.enabled:
            return False
            
        success = False
        
        # 格式化交易数据
        formatted_message = message
        if trade_data:
            formatted_message += "\n\n交易详情:"
            formatted_message += f"\n资产: {trade_data.get('symbol', 'Unknown')}"
            formatted_message += f"\n操作: {trade_data.get('action', 'Unknown')}"
            formatted_message += f"\n数量: {trade_data.get('quantity', 0)}"
            formatted_message += f"\n价格: ${trade_data.get('price', 0)}"
            formatted_message += f"\n时间: {trade_data.get('execution_time', datetime.now().isoformat())}"
            
        # 发送Telegram通知
        if self.telegram_enabled:
            telegram_success = self._send_telegram_notification(title, formatted_message)
            success = success or telegram_success
            
        # 发送Email通知
        if self.email_enabled:
            email_success = self._send_email_notification(title, formatted_message)
            success = success or email_success
            
        # 发送Discord通知
        if self.discord_enabled:
            discord_success = self._send_discord_notification(title, formatted_message)
            success = success or discord_success
            
        return success
        
    def _send_telegram_notification(self, title, message):
        """
        发送Telegram通知
        
        Args:
            title: 通知标题
            message: 通知内容
            
        Returns:
            bool: 是否成功发送通知
        """
        try:
            # 格式化消息
            telegram_message = f"*{title}*\n\n{message}"
            
            # 发送请求
            url = f"https://api.telegram.org/bot{self.telegram_bot_token}/sendMessage"
            payload = {
                "chat_id": self.telegram_chat_id,
                "text": telegram_message,
                "parse_mode": "Markdown"
            }
            
            response = requests.post(url, json=payload)
            
            if response.status_code == 200:
                logger.info("Telegram通知发送成功")
                return True
            else:
                logger.error(f"Telegram通知发送失败: {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"发送Telegram通知时出错: {e}")
            return False
            
    def _send_email_notification(self, title, message):
        """
        发送Email通知
        
        Args:
            title: 通知标题
            message: 通知内容
            
        Returns:
            bool: 是否成功发送通知
        """
        try:
            # 创建邮件
            msg = MIMEMultipart()
            msg['From'] = self.sender_email
            msg['To'] = self.receiver_email
            msg['Subject'] = f"加密货币交易代理: {title}"
            
            # 添加正文
            msg.attach(MIMEText(message, 'plain'))
            
            # 发送邮件
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.sender_email, self.email_password)
                server.send_message(msg)
                
            logger.info("Email通知发送成功")
            return True
            
        except Exception as e:
            logger.error(f"发送Email通知时出错: {e}")
            return False
            
    def _send_discord_notification(self, title, message):
        """
        发送Discord通知
        
        Args:
            title: 通知标题
            message: 通知内容
            
        Returns:
            bool: 是否成功发送通知
        """
        try:
            # 格式化消息
            discord_message = {
                "content": f"**{title}**\n\n{message}",
                "username": "加密货币交易代理"
            }
            
            # 发送请求
            response = requests.post(
                self.discord_webhook_url,
                json=discord_message
            )
            
            if response.status_code == 204:
                logger.info("Discord通知发送成功")
                return True
            else:
                logger.error(f"Discord通知发送失败: {response.status_code}, {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"发送Discord通知时出错: {e}")
            return False
            
    def send_trade_notification(self, trade_result):
        """
        发送交易通知
        
        Args:
            trade_result: 交易结果
            
        Returns:
            bool: 是否成功发送通知
        """
        if not self.enabled:
            return False
            
        # 提取交易数据
        action = trade_result.get('action', 'unknown')
        symbol = trade_result.get('symbol', 'unknown')
        quantity = trade_result.get('quantity', 0)
        price = trade_result.get('price', 0)
        status = trade_result.get('status', 'unknown')
        
        # 构建通知标题和内容
        if status == 'success':
            if action == 'buy':
                title = f"买入成功: {symbol}"
                message = f"成功买入 {quantity} {symbol} @ ${price}"
            elif action == 'sell':
                title = f"卖出成功: {symbol}"
                message = f"成功卖出 {quantity} {symbol} @ ${price}"
            else:
                title = f"交易成功: {symbol}"
                message = f"完成 {action} 操作 {quantity} {symbol} @ ${price}"
        else:
            title = f"交易失败: {symbol}"
            message = f"尝试 {action} {quantity} {symbol} 失败，原因: {trade_result.get('message', '未知')}"
            
        # 发送通知
        return self.send_notification(title, message, trade_result)
        
    def send_analysis_notification(self, analysis_result):
        """
        发送分析结果通知
        
        Args:
            analysis_result: 分析结果
            
        Returns:
            bool: 是否成功发送通知
        """
        if not self.enabled:
            return False
            
        # 提取分析数据
        symbol = analysis_result.get('symbol', 'unknown')
        price = analysis_result.get('price', 0)
        signal = analysis_result.get('debate_conclusion', {}).get('signal', 'neutral')
        confidence = analysis_result.get('debate_conclusion', {}).get('confidence', 0.5)
        risk_score = analysis_result.get('risk_analysis', {}).get('risk_score', 5)
        
        # 构建通知标题和内容
        title = f"分析结果: {symbol} - {signal.upper()}"
        
        message = f"{symbol} 当前价格: ${price}\n"
        message += f"分析信号: {signal.upper()}\n"
        message += f"信心指数: {confidence:.2f}\n"
        message += f"风险评分: {risk_score}/10"
        
        # 添加各个分析代理的信号
        if 'technical_analysis' in analysis_result:
            tech = analysis_result['technical_analysis']
            message += f"\n\n技术分析: {tech['signal'].upper()} (信心: {tech['confidence']})"
            
        if 'onchain_analysis' in analysis_result:
            onchain = analysis_result['onchain_analysis']
            message += f"\n链上分析: {onchain['signal'].upper()} (信心: {onchain['confidence']})"
            
        if 'sentiment_analysis' in analysis_result:
            sentiment = analysis_result['sentiment_analysis']
            message += f"\n情绪分析: {sentiment['signal'].upper()} (信心: {sentiment['confidence']})"
            
        if 'valuation_analysis' in analysis_result:
            valuation = analysis_result['valuation_analysis']
            message += f"\n估值分析: {valuation['signal'].upper()} (信心: {valuation['confidence']})"
            
        # 发送通知
        return self.send_notification(title, message, {'symbol': symbol, 'price': price})
