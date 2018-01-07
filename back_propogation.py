import numpy as np


class NeuralNetwork(object):
    """
    Here we will try to program a Neural Network that can have arbirary number
    of layers and we will also write a program to train and verify the Network.
    """
    def __init__(self, inputs, outputs, nodes=[3], learning_rate=0.05):
        # First initialize the weight matrices.
        # Nodes is an array that contains number of nodes in each layer of the
        # neural net. This should be a non-zero positive array of integers.
        nodes.append(outputs)
        nodes.insert(0, inputs)
        self.weights = []
        for i, val in enumerate(nodes[:-1]):
            self.weights.append(2*np.random.random((nodes[i], nodes[i+1]))-1)


    def __sigmoid(self, x):
       return 1/(1+np.exp(-x))

    def __sigmoid_derivative(self, x):
        return x*(1-x)

    def train(self, training_set_inputs, training_set_outputs, number_of_training_iterations):
        for itr in xrange(number_of_training_iterations):
            for inputs, target in zip(training_set_inputs, training_set_outputs):
                # Forward Pass
                layer_outputs = self.layer_outputs(inputs)
                error = np.linalg.norm(layer_outputs[-1]-target)

                # Backward Pass
                delta = self.deltas(layer_outputs, target)

    def deltas(self, layer_outputs, target):
        delta = [0]*len(self.weights)
        delta[-1] = [np.multiply((layer_outputs[-1]-target),self.__sigmoid_derivative(np.dot(layer_outputs[-2], self.weights[-1])))]
        for i in range(len(self.weights)-2, 0, -1):
            k = np.multiply(np.dot(delta[i+1],np.transpose(self.weights[i+1])), self.__sigmoid_derivative(np.dot(self.weights[i-1]



    def layer_outputs(self, inputs):
        outputs = [inputs]
        for w in self.weights:
            outputs.append(self.__sigmoid(np.dot(outputs[-1], w)))
        return outputs


if __name__ == '__main__':
    nn = NeuralNetwork(3, 3, nodes=[4,4])
    nn.train(np.array([[1,1,1],[1,0,1]]), np.array([[0.5,0.4,0.3],[.3,.2,.1]]), 1)

