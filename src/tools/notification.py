import logging
import smtplib
import requests
import json
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime

logger = logging.getLogger('notification')

class NotificationManager:
    """
    通知管理器 - 负责发送各种通知
    
    支持的通知类型:
    1. Telegram
    2. Email
    3. 自定义webhook
    """
    
    def __init__(self, config=None):
        """
        初始化通知管理器
        
        Args:
            config: 通知配置字典
        """
        self.config = config or {}
        self.enabled = self.config.get('enabled', False)
        
        if not self.enabled:
            logger.info("通知功能已禁用")
            
    def send_notification(self, message, subject=None, notification_type=None):
        """
        发送通知
        
        Args:
            message: 通知内容
            subject: 通知主题(用于邮件)
            notification_type: 通知类型('telegram', 'email', 'webhook', None=all)
            
        Returns:
            bool: 是否成功发送
        """
        if not self.enabled:
            logger.debug("通知功能已禁用，跳过发送")
            return False
            
        if not message:
            logger.warning("通知内容为空，跳过发送")
            return False
            
        # 默认发送所有启用的通知类型
        if notification_type is None:
            results = []
            
            if self.config.get('telegram', {}).get('enabled', False):
                results.append(self.send_telegram(message))
                
            if self.config.get('email', {}).get('enabled', False):
                results.append(self.send_email(message, subject))
                
            if self.config.get('webhook', {}).get('enabled', False):
                results.append(self.send_webhook(message))
                
            return any(results)  # 只要有一个成功发送就返回True
            
        # 发送特定类型的通知
        if notification_type == 'telegram':
            return self.send_telegram(message)
        elif notification_type == 'email':
            return self.send_email(message, subject)
        elif notification_type == 'webhook':
            return self.send_webhook(message)
        else:
            logger.warning(f"不支持的通知类型: {notification_type}")
            return False
            
    def send_telegram(self, message):
        """
        发送Telegram通知
        
        Args:
            message: 通知内容
            
        Returns:
            bool: 是否成功发送
        """
        telegram_config = self.config.get('telegram', {})
        if not telegram_config.get('enabled', False):
            logger.debug("Telegram通知已禁用")
            return False
            
        bot_token = telegram_config.get('bot_token')
        chat_id = telegram_config.get('chat_id')
        
        if not bot_token or not chat_id:
            logger.warning("缺少Telegram配置(bot_token或chat_id)")
            return False
            
        try:
            url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
            payload = {
                'chat_id': chat_id,
                'text': message,
                'parse_mode': 'Markdown'
            }
            
            response = requests.post(url, data=payload)
            
            if response.status_code == 200:
                logger.info("Telegram通知发送成功")
                return True
            else:
                logger.error(f"Telegram通知发送失败: {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"发送Telegram通知时出错: {e}")
            return False
            
    def send_email(self, message, subject=None):
        """
        发送邮件通知
        
        Args:
            message: 通知内容
            subject: 邮件主题
            
        Returns:
            bool: 是否成功发送
        """
        email_config = self.config.get('email', {})
        if not email_config.get('enabled', False):
            logger.debug("邮件通知已禁用")
            return False
            
        smtp_server = email_config.get('smtp_server')
        smtp_port = email_config.get('smtp_port', 587)
        sender_email = email_config.get('sender_email')
        receiver_email = email_config.get('receiver_email')
        password = email_config.get('password')
        
        if not all([smtp_server, sender_email, receiver_email, password]):
            logger.warning("缺少邮件配置(smtp_server, sender_email, receiver_email或password)")
            return False
            
        try:
            # 创建邮件
            msg = MIMEMultipart()
            msg['From'] = sender_email
            msg['To'] = receiver_email
            msg['Subject'] = subject or f"加密货币交易通知 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            
            # 添加邮件内容
            msg.attach(MIMEText(message, 'plain'))
            
            # 连接SMTP服务器并发送
            with smtplib.SMTP(smtp_server, smtp_port) as server:
                server.starttls()  # 启用TLS加密
                server.login(sender_email, password)
                server.send_message(msg)
                
            logger.info(f"邮件通知已发送至{receiver_email}")
            return True
            
        except Exception as e:
            logger.error(f"发送邮件通知时出错: {e}")
            return False
            
    def send_webhook(self, message):
        """
        发送Webhook通知
        
        Args:
            message: 通知内容
            
        Returns:
            bool: 是否成功发送
        """
        webhook_config = self.config.get('webhook', {})
        if not webhook_config.get('enabled', False):
            logger.debug("Webhook通知已禁用")
            return False
            
        webhook_url = webhook_config.get('url')
        if not webhook_url:
            logger.warning("缺少Webhook URL配置")
            return False
            
        try:
            headers = webhook_config.get('headers', {'Content-Type': 'application/json'})
            payload = webhook_config.get('payload_template', {}).copy()
            
            # 将消息添加到payload
            if isinstance(payload, dict):
                payload['text'] = message
                payload['timestamp'] = datetime.now().isoformat()
            else:
                payload = {'text': message, 'timestamp': datetime.now().isoformat()}
                
            response = requests.post(webhook_url, headers=headers, data=json.dumps(payload))
            
            if response.status_code in [200, 201, 202, 204]:
                logger.info("Webhook通知发送成功")
                return True
            else:
                logger.error(f"Webhook通知发送失败: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"发送Webhook通知时出错: {e}")
            return False
            
    def send_trade_notification(self, trade_info):
        """
        发送交易通知
        
        Args:
            trade_info: 交易信息字典
            
        Returns:
            bool: 是否成功发送
        """
        if not self.enabled:
            return False
            
        # 格式化交易信息
        action = trade_info.get('action', 'UNKNOWN')
        symbol = trade_info.get('symbol', 'UNKNOWN')
        quantity = trade_info.get('quantity', 0)
        price = trade_info.get('price', 0)
        
        message = f"🚨 交易执行通知 🚨\n\n"
        message += f"操作: {action.upper()}\n"
        message += f"交易对: {symbol}\n"
        message += f"数量: {quantity}\n"
        message += f"价格: ${price:.2f}\n"
        message += f"总价值: ${quantity * price:.2f}\n"
        message += f"执行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        
        if 'message' in trade_info:
            message += f"\n备注: {trade_info['message']}"
            
        return self.send_notification(
            message=message,
            subject=f"交易通知: {action.upper()} {symbol}",
        )
        
    def send_alert_notification(self, alert_info):
        """
        发送警报通知
        
        Args:
            alert_info: 警报信息字典
            
        Returns:
            bool: 是否成功发送
        """
        if not self.enabled:
            return False
            
        alert_level = alert_info.get('level', 'INFO').upper()
        alert_type = alert_info.get('type', 'GENERAL').upper()
        alert_message = alert_info.get('message', '')
        
        message = f"⚠️ {alert_level} 警报: {alert_type} ⚠️\n\n"
        message += f"{alert_message}\n"
        message += f"\n时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        
        return self.send_notification(
            message=message,
            subject=f"{alert_level} 警报: {alert_type}"
        )
        
    def send_analysis_notification(self, analysis_summary):
        """
        发送分析结果通知
        
        Args:
            analysis_summary: 分析摘要字典
            
        Returns:
            bool: 是否成功发送
        """
        if not self.enabled:
            return False
            
        symbol = analysis_summary.get('symbol', 'UNKNOWN')
        
        message = f"📊 加密货币分析结果: {symbol} 📊\n\n"
        
        # 价格信息
        price = analysis_summary.get('price', 0)
        change = analysis_summary.get('change_24h_percent', 0)
        message += f"当前价格: ${price:.2f} ({change:+.2f}%)\n\n"
        
        # 信号结果
        message += "分析结果:\n"
        
        if 'technical_analysis' in analysis_summary:
            tech = analysis_summary['technical_analysis']
            message += f"- 技术分析: {tech['signal'].upper()} (信心: {tech['confidence']})\n"
            
        if 'onchain_analysis' in analysis_summary:
            onchain = analysis_summary['onchain_analysis']
            message += f"- 链上分析: {onchain['signal'].upper()} (信心: {onchain['confidence']})\n"
            
        if 'sentiment_analysis' in analysis_summary:
            sentiment = analysis_summary['sentiment_analysis']
            message += f"- 情绪分析: {sentiment['signal'].upper()} (信心: {sentiment['confidence']})\n"
            
        if 'valuation_analysis' in analysis_summary:
            valuation = analysis_summary['valuation_analysis']
            message += f"- 估值分析: {valuation['signal'].upper()} (信心: {valuation['confidence']})\n"
            
        # 辩论结论
        if 'debate_conclusion' in analysis_summary:
            debate = analysis_summary['debate_conclusion']
            message += f"\n最终结论: {debate['signal'].upper()} (信心: {debate['confidence']:.2f})\n"
            message += f"看多信心: {debate['bull_confidence']:.2f}, 看空信心: {debate['bear_confidence']:.2f}\n"
            
        # 风险分析
        if 'risk_analysis' in analysis_summary:
            risk = analysis_summary['risk_analysis']
            message += f"\n风险评分: {risk['risk_score']}/10\n"
            message += f"建议操作: {risk['trading_action'].upper()}\n"
            
        # 交易决策(如果有)
        if 'portfolio_decision' in analysis_summary:
            decision = analysis_summary['portfolio_decision']
            message += f"\n交易决策: {decision['action'].upper()} {decision['quantity']} (信心: {decision['confidence']:.2f})\n"
            
        # 时间戳
        message += f"\n分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        
        return self.send_notification(
            message=message,
            subject=f"{symbol} 分析报告: {analysis_summary.get('debate_conclusion', {}).get('signal', 'NEUTRAL').upper()}"
        )


# 示例用法
if __name__ == "__main__":
    # 配置示例
    notification_config = {
        "enabled": True,
        "telegram": {
            "enabled": True,
            "bot_token": "YOUR_BOT_TOKEN",
            "chat_id": "YOUR_CHAT_ID"
        },
        "email": {
            "enabled": False,
            "smtp_server": "smtp.gmail.com",
            "smtp_port": 587,
            "sender_email": "your_email@gmail.com",
            "receiver_email": "your_email@gmail.com",
            "password": "YOUR_PASSWORD"
        },
        "webhook": {
            "enabled": False,
            "url": "https://your-webhook-url.com",
            "headers": {"Content-Type": "application/json"},
            "payload_template": {"username": "CryptoBot"}
        }
    }
    
    # 创建通知管理器
    notifier = NotificationManager(notification_config)
    
    # 发送测试通知
    notifier.send_notification("这是一条测试通知")
    
    # 发送交易通知
    trade_info = {
        "action": "buy",
        "symbol": "BTC/USDT",
        "quantity": 0.1,
        "price": 50000,
        "message": "基于技术分析和链上数据的突破信号"
    }
    notifier.send_trade_notification(trade_info)