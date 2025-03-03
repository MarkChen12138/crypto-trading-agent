import os
import time
import requests
import json
from dataclasses import dataclass
import backoff
import logging
from dotenv import load_dotenv

# 设置日志
logger = logging.getLogger('api_calls')

@dataclass
class ChatMessage:
    content: str

@dataclass
class ChatChoice:
    message: ChatMessage

@dataclass
class ChatCompletion:
    choices: list[ChatChoice]

# 加载环境变量
load_dotenv()

# 获取OpenRouter API密钥
api_key = os.getenv("OPENROUTER_API_KEY")
default_model = os.getenv("OPENROUTER_MODEL", "anthropic/claude-3-haiku")

# 检查API密钥是否存在
if not api_key:
    logger.warning("未找到OPENROUTER_API_KEY环境变量，将无法使用AI情感分析功能")


@backoff.on_exception(
    backoff.expo,
    (requests.exceptions.Timeout, requests.exceptions.RequestException),
    max_tries=5,
    max_time=300,
    giveup=lambda e: "rate limit" not in str(e).lower()
)
def generate_completion(messages, model=None):
    """使用OpenRouter API生成文本补全，包含重试逻辑"""
    if not api_key:
        logger.error("未找到OpenRouter API密钥，无法执行API调用")
        return None
        
    model = model or default_model
    
    logger.info(f"使用模型: {model}")
    logger.debug(f"消息内容: {messages}")
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    data = {
        "model": model,
        "messages": messages,
        "max_tokens": 1024
    }
    
    try:
        logger.info("正在调用OpenRouter API...")
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            data=json.dumps(data),
            timeout=30
        )
        
        response.raise_for_status()
        response_data = response.json()
        logger.info("API调用成功")
        
        return response_data.get("choices", [{}])[0].get("message", {}).get("content", "")
        
    except Exception as e:
        if "rate limit" in str(e).lower():
            logger.warning(f"触发API请求限制，等待重试... 错误: {str(e)}")
            time.sleep(5)
            raise e
        logger.error(f"API调用失败: {str(e)}")
        return None

def get_chat_completion(messages, model=None, max_retries=3, initial_retry_delay=1):
    """获取聊天完成结果，包含重试逻辑"""
    try:
        if not api_key:
            return None
            
        for attempt in range(max_retries):
            try:
                result = generate_completion(messages, model)
                if result is None:
                    logger.warning(f"尝试 {attempt + 1}/{max_retries}: API返回空值")
                    if attempt < max_retries - 1:
                        retry_delay = initial_retry_delay * (2 ** attempt)
                        logger.info(f"等待 {retry_delay} 秒后重试...")
                        time.sleep(retry_delay)
                        continue
                    return None
                    
                logger.debug(f"API响应: {result[:500]}...")
                return result
                
            except Exception as e:
                logger.error(f"尝试 {attempt + 1}/{max_retries} 失败: {str(e)}")
                if attempt < max_retries - 1:
                    retry_delay = initial_retry_delay * (2 ** attempt)
                    logger.info(f"等待 {retry_delay} 秒后重试...")
                    time.sleep(retry_delay)
                else:
                    logger.error(f"最终错误: {str(e)}")
                    return None
                    
    except Exception as e:
        logger.error(f"get_chat_completion 发生错误: {str(e)}")
        return None
