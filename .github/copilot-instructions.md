# Dive into Deep Learning 项目 - GitHub Copilot 指令

## 项目概述
深度学习教程实践项目，包含多个子章节的独立实现。

## 全局编程规范

### 项目结构标准
```
Chpt_X_ProjectName/
├── src/           # 源代码
│   ├── data.py, model.py, trainer.py, utils.py, main.py
│   └── visualization.py # 可视化（可选）
├── configs/config.yaml   # 配置文件
├── data/raw/            # 原始数据
├── outputs/             # 输出结果
│   ├── logs/, models/, visualizations/
├── README.md, requirements.txt
├── test_modules.py      # 测试文件
└── .instruction.md      # 局部指令（可选）
```

### 编码规范
- **语言**: Python 3.8+, PyTorch, PEP 8
- **命名**: 类名PascalCase，函数snake_case，常量UPPER_SNAKE_CASE
- **文档**: 公共函数必须有docstring，类型提示强制要求
- **导入**: 标准库 → 第三方库 → 本地模块

### 核心规范
- **日志**: 使用logging模块，格式`training_YYYYMMDD_HHMMSS.log`
- **输出**: 文件命名含时间戳，分别保存到logs/models/visualizations/
- **配置**: YAML管理超参数，支持命令行覆盖，参数验证
- **测试**: 内嵌测试(`if __name__`) + 独立脚本
- **质量**: 函数≤50行，参数≤7个，嵌套≤3层，单一职责
- **模块化**: 按照单一职责原则分解复杂功能，核心模块代码≤200行，公共函数放在utils.py
- **异常**: 自定义异常类，完整错误上下文，资源清理
- **性能**: 监控内存/GPU/时间，缓存策略，性能日志

### 深度学习规范
- GPU优先，内存优化，随机种子，基础指标+可视化

### Copilot及AI助手规范
- 不要加入用户指令以外的功能。如有必要请先询问用户
- 生成代码时应考虑可读性和简洁性！保持代码易于理解和维护
- 终端命令不支持 "&&"

### 版本控制
- Git管理，提交格式`[类型] 描述`，使用.gitignore忽略模型/数据文件