import numpy as np

from layers import (
    FullyConnectedLayer, ReLULayer,
    ConvolutionalLayer, MaxPoolingLayer, Flattener,
    softmax_with_cross_entropy, l2_regularization
    )


class ConvNet:
 
    def __init__(self, input_shape, n_output_classes, conv1_channels, conv2_channels):
        
        self.layers = [
            ConvolutionalLayer(input_shape[2], conv1_channels, 3, 1),
            ReLULayer(),
            MaxPoolingLayer(4, 4),
            ConvolutionalLayer(conv1_channels, conv2_channels, 3, 1),
            ReLULayer(),
            MaxPoolingLayer(4, 4),
            Flattener(),
            FullyConnectedLayer(2 * 2 * conv2_channels, 10)
        ]

    def __clear_grads(self) -> None:
        for param in self.params().values():
            param.grad.fill(0.0)

    def __forward(self, X: np.array) -> np.array:
        layer_output = X

        for layer in self.layers:
            layer_output = layer.forward(layer_output)
        
        return layer_output

    def __backward(self, d_out: np.array) -> None:
        for layer in reversed(self.layers):
            d_out = layer.backward(d_out)

    def compute_loss_and_gradients(self, X, y):
        
        assert X.shape[0] == y.shape[0], f'{X.shape[0]} != {y.shape[0]}'

        self.__clear_grads()
        forward_out = self.__forward(X)
        loss, d_out = softmax_with_cross_entropy(forward_out, y)
        self.__backward(d_out)

        return loss

    def predict(self, X: np.array):
        forward_out = self.__forward(X)
        pred = np.argmax(forward_out, axis=1)

        return pred

    def params(self):
        result = {
            "W1": self.layers[0].params()['W'],
            "B1": self.layers[0].params()['B'],
            "W2": self.layers[3].params()['W'],
            "B2": self.layers[3].params()['B'],
            "W3": self.layers[7].params()['W'],
            "B3": self.layers[7].params()['B'],
        }
        return result
