跑通了整个流程，30k次迭代后，miou在68.47，与论文及第三方代码中的结果仍然有较大的差距。下载测试了第三方代码，依次替换了主干网络、aspp以及loss等部分，发现到目前位置影响最大的是batch size。

# TODOs
- [x] 调整weight decay，目前为5e-4，第三方代码中使用1e-4。从loss结果看，l2\_reqularize\_loss显著大于交叉熵了。调整之后，miou开始会大一些，但最终也只到68.35（注意当时miou计算是错误的，实际计算的是从第1次evaluate到当前所有结果总的miou，因此从这个结果来看，使用5e-4的weight decay，最终的miou应该是更大的）
- [x] 基于上一点的观察，用自己实现的代码，重新将weight decay调整到5e-4再跑一次50k次迭代、batch size为10的进行测试
- [ ] 5e-4的miou为73+，但是看到cross entropy loss比较低，修改batchnorm decay为0.9997再跑一下
- [x] 实现predict.py的预测功能
- [ ] 在自动驾驶数据集cityscapes上进行测试
- [ ] 实现deeplibv3+，参照第三方代码
- [ ] 换mobilenetv2进行测试
