## mnist.pkl.gz 数据集

```python
f = gzip.open('mnist.pkl.gz', 'rb')
train_data, val_data, test_data = pickle.load(f, encoding = 'bytes')
```

### How to use

```
>>> import network as nn
>>> net = nn.Network([784, 30, 10])
>>> net.train(train, 30, 10, 3.0, test)

Epoch 0: 8949 / 10000
Epoch 1: 9129 / 10000
Epoch 2: 9229 / 10000
Epoch 3: 9265 / 10000
Epoch 4: 9319 / 10000
Epoch 5: 9343 / 10000
Epoch 6: 9411 / 10000
Epoch 7: 9393 / 10000
Epoch 8: 9391 / 10000
Epoch 9: 9411 / 10000
Epoch 10: 9427 / 10000
Epoch 11: 9426 / 10000
Epoch 12: 9445 / 10000
Epoch 13: 9416 / 10000
Epoch 14: 9440 / 10000
Epoch 15: 9447 / 10000
Epoch 16: 9459 / 10000
Epoch 17: 9438 / 10000
Epoch 18: 9462 / 10000
Epoch 19: 9499 / 10000
Epoch 20: 9465 / 10000
Epoch 21: 9471 / 10000
Epoch 22: 9485 / 10000
Epoch 23: 9483 / 10000
Epoch 24: 9462 / 10000
Epoch 25: 9483 / 10000
Epoch 26: 9490 / 10000
Epoch 27: 9471 / 10000
Epoch 28: 9498 / 10000
Epoch 29: 9477 / 10000
```

训练的准确率为 94%， 但是训练的速度太慢了。那还有那些需要改进的地方那？

### Todo

- [ ] 权重矩阵和偏置向量的初始化方法？
- [ ] 激活函数的选择？
- [ ] 优化策略, Momentum, Props, dropout
- [ ] 如何提高运行的速度
- [ ] 如何提高准确率
- [ ] 如何验证程序是正确的

### NN 的优化算法有

优化的基础 EWMA (Exponentially Weighted Moving Average)

1. Momentum, 因为 mini-batch 的梯度并不一定正确只是一个估计，使用 EWMA 的方法来做估计，修正 Gradient, 提高收敛速度。 引用 β 超参.
2. AdaGrad (Adaptive Learning Rate) 根据 Gradient 的信息来调整学习因子。
3. AdaDelta, RMSProp 都是根据 Gradient 的信息来调用学习因子，但是是使用 EWMA 的方法来调整。
4. Adam, 是将 RMSProp 与 Momentum 接合起来。
