## TensorFlow

01. Linear Regression

learning rate 影响很大。

02. Adam 效率果然很高


默认安装的 TensorFlow 执行并没利用 CPU 的 AVX2, FMA 。

```
I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA

```

执行效率会查一些，

epochs 20, batch 100 : 1m 15s

等我重新编译下 TensorFlow 试一下。
