import unittest
import network as nn
import mnist_loader as loader

class TestNN(unittest.TestCase):
    '''
        Test NueralNetwork base methods
    '''
    def test_init(self):
        sizes = [3, 4, 4, 4, 2]
        model = nn.Network(sizes)
        self.assertEqual(model.L, 5)
        self.assertEqual(len(model.bias), model.L - 1)
        self.assertEqual(len(model.weights), model.L - 1)

        for i in range(model.L - 1):
            with self.subTest():
                self.assertEqual(model.bias[i].shape, (sizes[i + 1], 1))
                self.assertEqual(model.weights[i].shape, (sizes[i + 1], sizes[i]))
    def test_train(self):
        train_data, val_data, test_data = loader.load_data_wrapper()
        sizes = [784, 30, 30, 10]
        model = nn.Network(sizes)
        model.train(train_data, 30, 3.0, 25, test_data)

if __name__ == '__main__':
    unittest.main()
