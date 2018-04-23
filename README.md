跑通了整个流程，30k次迭代后，miou在68.47，与论文及第三方代码中的结果仍然有较大的差距。

# TODOs
- [ ] 调整weight decay，目前为5e-4，第三方代码中使用1e-4。从loss结果看，l2\_reqularize\_loss显著大于交叉熵了
- [ ] 确认dataset部分，和第三方代码比对，包括训练集和验证集的处理
- [ ] 确认主干部分网络结构的正确，特别是multi\_grid部分，是否按原文3.2.1与rates=2有叠加的效果
- [ ] 若仍然有较大的差距，clone第三方代码进行测试
- [ ] 实现接近原文及第三方代码上的效果
- [ ] 实现predict.py的预测功能
- [ ] 实现deeplibv3+，参照第三方代码
- [ ] 在自动驾驶数据集上进行测试
- [ ] 换mobilenetv2进行测试
