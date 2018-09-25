import unittest
import network as nn
import mnist_loader as loader

class TestStringMethods(unittest.TestCase):
    '''
        测试一些字符串方法
    '''

    def test_upper(self):
        self.assertEqual('foo'.upper(), 'FOO')

    def test_isupper(self):
        self.assertTrue('FOO'.isupper())
        self.assertFalse('foo'.isupper())

    def test_split(self):
        s = 'hello world'
        self.assertEqual(s.split(), ['hello', 'world'])
        with self.assertRaises(TypeError):
            s.split(2)

class TestNN(unittest.TestCase):
    '''
        Test NueralNetwork base methods
    '''
    def test_init(self):
        sizes = [3, 4, 4, 4, 2]
        model = nn.NueralNetwork(sizes)
        self.assertEqual(model.L, 5)
        self.assertEqual(len(model.biases), model.L - 1)
        self.assertEqual(len(model.weights), model.L - 1)

        for i in range(model.L - 1):
            with self.subTest():
                self.assertEqual(model.biases[i].shape, (sizes[i + 1], 1))
                self.assertEqual(model.weights[i].shape, (sizes[i + 1], sizes[i]))
    def test_train(self):
        train_data, val_data, test_data = loader.load_data_wrapper()
        sizes = [784, 30, 30, 10]
        model = nn.NueralNetwork(sizes)
        model.train(train_data, 10, 0.1, 25, test_data)

if __name__ == '__main__':
    unittest.main()
