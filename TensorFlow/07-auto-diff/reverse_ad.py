import math


class Var:
    def __init__(self, value):
        self.value = value
        self.children = []
        self.grad_value = None

    def grad(self):
        '''
        通过递归来求得导数。
        '''
        if self.grad_value is None:
            self.grad_value = sum(weight * var.grad() for
                                  weight, var in self.children)
        return self.grad_value

    def __add__(self, other):
        """
        重载 + ,  保存中间导数。
        """
        z = Var(self.value + other.value)
        self.children.append((1.0, z))
        other.children.append((1.0, z))
        return z

    def __mul__(self, other):
        """
        重载 *
        """
        z = Var(self.value * other.value)
        # 乘法求导表达式，需要记住其系数  weigths 和结果 z
        self.children.append((other.value, z))
        other.children.append((self.value, z))
        return z


def sin(x):
    z = Var(math.sin(x.value))
    x.children.append((math.cos(x.value), z))
    return z


def main():
    x = Var(0.5)
    y = Var(4.2)
    z = x * y + sin(x)

    # Reverse
    z.grad_value = 1

    dx = x.grad()
    dy = y.grad()

    # 在实现计算的时候就已经记下了导数的求法, 真是赞.

    print(z.value, 0.5 * 4.2 + math.sin(0.5))
    print('dx = ', dx, 4.2 + math.cos(0.5))
    print('dy = ', dy, 0.5)


if __name__ == "__main__":
    main()
