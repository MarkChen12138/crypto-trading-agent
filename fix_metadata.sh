#!/bin/bash

# 用于修复所有代理文件中的metadata访问问题的脚本

# 代理文件列表
AGENT_FILES=(
  "src/agents/technicals.py"
  "src/agents/onchain_analysis.py"
  "src/agents/sentiment.py"
  "src/agents/valuation.py"
  "src/agents/researcher_bull.py"
  "src/agents/researcher_bear.py"
  "src/agents/debate_room.py"
  "src/agents/risk_manager.py"
  "src/agents/portfolio_manager.py"
  "src/agents/execution.py"
)

# 修复内容 - 将直接访问metadata替换为安全访问
for file in "${AGENT_FILES[@]}"; do
  echo "修复文件: $file"
  
  # 检查文件是否存在
  if [ ! -f "$file" ]; then
    echo "文件不存在: $file"
    continue
  fi
  
  # 使用sed替换内容 (Mac OS X需要使用-i ''，Linux使用-i)
  # 替换 show_reasoning = state["metadata"]["show_reasoning"] 行
  sed -i '' 's/show_reasoning = state\["metadata"\]\["show_reasoning"\]/if "metadata" in state and "show_reasoning" in state.get("metadata", {}):\n        show_reasoning = state["metadata"]["show_reasoning"]\n    else:\n        show_reasoning = False  # 默认不显示推理过程/g' "$file"
  
  echo "完成修复: $file"
done

echo "所有文件修复完成！"