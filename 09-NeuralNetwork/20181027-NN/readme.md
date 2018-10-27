## NN 实现

1. 参数初始化

sizes = [784, 30, 30, 30, 10]

self.w, self.b

2. 激活函数

f(x) = sigmoid(z)
f'(x) = f(x)(1 - f(x))

3. FP

a1 = x
zl = wl a(l-1) + bl
al = f(zl)

4. BP


 
