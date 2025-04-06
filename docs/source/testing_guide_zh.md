# vLLM在线权重切换测试指南

本指南提供了测试vLLM在线权重切换功能的说明。它包括基于CPU的验证测试和基于GPU的功能测试。

## 前提条件

- Python 3.8或更高版本
- Git
- 对于GPU测试：CUDA 11.8或更高版本以及兼容的GPU

## 安装

我们提供了一个安装脚本，用于设置测试所需的依赖项。该脚本创建Python虚拟环境并安装所需的包。

```bash
# 如果尚未克隆仓库，请先克隆
git clone https://github.com/yumc-afk/vllm.git
cd vllm

# 使安装脚本可执行
chmod +x scripts/install_test_dependencies.sh

# 对于仅CPU测试
./scripts/install_test_dependencies.sh --cpu

# 对于GPU测试
./scripts/install_test_dependencies.sh --gpu

# 激活虚拟环境
source venv/bin/activate
```

## 基于CPU的验证测试

这些测试在不需要GPU硬件的情况下验证实现逻辑。它们使用模拟对象和静态分析来验证实现的正确性。

### 运行简化的模拟测试

```bash
# 运行简化的模拟测试
python simplified_mock_tests.py
```

此脚本使用模拟对象测试权重切换实现的控制流。它验证：
- 方法调用正确地通过类层次结构传播
- 参数正确地通过调用链传递
- 在权重更新期间请求被正确阻塞
- 错误处理正常工作

### 运行代码路径分析

```bash
# 运行代码路径分析
python code_path_analysis.py
```

此脚本分析权重切换实现中的所有可能执行路径，以确保正确性并识别潜在问题。

### 运行静态验证

```bash
# 运行静态验证
python simplified_validation.py
```

此脚本对实现进行静态验证，检查API一致性、类型正确性、线程安全性和内存管理。

## 基于GPU的功能测试

这些测试需要GPU硬件，并验证实现的实际功能。

### 运行单元测试

```bash
# 运行权重切换单元测试
pytest tests/async_engine/test_weight_switching.py -v
```

这将运行权重切换功能的单元测试，验证它在实际GPU操作中是否正常工作。

### 运行DeepSeek V3示例

```bash
# 运行DeepSeek V3权重切换测试
python examples/deepseek_v3_weight_switching_test.py
```

此示例演示了使用mini DeepSeek V3模型的权重切换功能。它展示了如何：
1. 使用模型初始化AsyncEngine
2. 使用初始模型生成文本
3. 更新模型权重
4. 使用更新后的模型生成文本

## 故障排除

### 常见问题

1. **ImportError: No module named 'vllm'**
   - 确保您已使用`pip install -e .`安装了vLLM
   - 确保您已使用`source venv/bin/activate`激活了虚拟环境

2. **CUDA内存不足**
   - 减小模型大小或批处理大小
   - 通过关闭其他应用程序释放GPU内存

3. **测试失败**
   - 查看错误消息了解详情
   - 确保您使用的是带有权重切换实现的正确分支
   - 验证所有依赖项是否正确安装

### 获取帮助

如果您遇到本指南未涵盖的问题，请：
1. 查看vLLM文档
2. 在GitHub仓库上开一个issue
3. 联系维护者寻求帮助

## 结论

通过遵循本指南，您应该能够使用基于CPU的验证测试和基于GPU的功能测试来测试vLLM中的在线权重切换功能。基于CPU的测试提供了一种在不需要GPU硬件的情况下验证实现逻辑的方法，而基于GPU的测试则使用真实硬件验证实际功能。
