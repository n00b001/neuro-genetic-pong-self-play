import numpy as np
from scipy.stats import truncnorm


@np.vectorize
# alternative activation function
def ReLU(x):
    return np.maximum(0.0, x)


@np.vectorize
# derivation of relu
def ReLU_derivation(x):
    if x <= 0:
        return 0
    else:
        return 1


@np.vectorize
def sigmoid(x):
    return 1 / (1 + np.e ** -x)


activation_function = sigmoid


def truncated_normal(mean=0, sd=1, low=0, upp=10):
    return truncnorm((low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)


class NeuralNetwork:

    def __init__(self,
                 nodes: list,
                 learning_rate=0.1,
                 bias=None,
                 weights=None
                 ):

        self.nodes = nodes
        self.learning_rate = learning_rate
        self.bias = 1 if bias else 0
        if weights is None:
            self.weights = self.create_random_weight_matrices()
        else:
            self.weights = self.load_weights(weights)

    def load_weights(self, weights):
        index = 0

        _weights = []
        for i in range(len(self.nodes) - 1):
            input_node_amount = self.nodes[i]
            output_node_amount = self.nodes[i + 1]

            number_of_numbers_number_to_take = (input_node_amount + self.bias) * output_node_amount
            matrix_weights = np.array(weights[index:index + number_of_numbers_number_to_take])
            # matrix_weights = matrix_weights.reshape(input_node_amount + self.bias, output_node_amount)
            matrix_weights = matrix_weights.reshape(output_node_amount, input_node_amount + self.bias)
            _weights.append(matrix_weights)
            index += number_of_numbers_number_to_take
        if index != len(weights):
            print(f"Not all weights loaded!\n Loaded: {index} weights")
        # return np.array(_weights)
        return _weights

    def create_random_weight_matrices(self):
        """ A method to initialize the weight matrices of the neural
        network with optional bias nodes"""

        bias_node = 1 if self.bias else 0
        weights = []

        for i in range(len(self.nodes) - 1):
            rad = 1 / np.sqrt(self.nodes[i] + bias_node)
            X = truncated_normal(mean=0, sd=1, low=-rad, upp=rad)
            weights.append(X.rvs((self.nodes[i + 1], self.nodes[i] + bias_node)))
        return weights

    def train(self, input_vector, target_vector):
        # input_vector and target_vector can be tuple, list or ndarray
        if self.bias:
            # adding bias node to the end of the inpuy_vector
            input_vector = np.concatenate((input_vector, [self.bias]))

        input_vector = np.array(input_vector, ndmin=2).T
        target_vector = np.array(target_vector, ndmin=2).T

        output_vector_hidden = input_vector
        for w in self.weights[:-1]:
            output_vector1 = np.dot(w, output_vector_hidden)
            output_vector_hidden = activation_function(output_vector1)

            if self.bias:
                output_vector_hidden = np.concatenate((output_vector_hidden, [[self.bias]]))
        output_vector2 = np.dot(self.weights[-1], output_vector_hidden)
        output_vector_network = activation_function(output_vector2)

        output_errors = target_vector - output_vector_network
        # update the weights:
        tmp = output_errors * output_vector_network * (1.0 - output_vector_network)
        tmp = self.learning_rate * np.dot(tmp, output_vector_hidden.T)
        self.weights[-1] += tmp

        # calculate hidden errors:
        hidden_errors = np.dot(self.weights[-1].T, output_errors)
        # update the weights:
        tmp = hidden_errors * output_vector_hidden * (1.0 - output_vector_hidden)
        if self.bias:
            x = np.dot(tmp, input_vector.T)[:-1, :]  # ???? last element cut off, ???
        else:
            x = np.dot(tmp, input_vector.T)
        for w in self.weights[:-1]:
            w += self.learning_rate * x

    def run(self, input_vector):
        # input_vector can be tuple, list or ndarray

        if self.bias:
            # adding bias node to the end of the inpuy_vector
            input_vector = np.concatenate((input_vector, [self.bias]))
        input_vector = np.array(input_vector, ndmin=2).T

        for w in self.weights[:-1]:
            input_vector = np.dot(w, input_vector)
            input_vector = activation_function(input_vector)

            if self.bias:
                input_vector = np.concatenate((input_vector, [[self.bias]]))

        input_vector = np.dot(self.weights[-1], input_vector)
        input_vector = activation_function(input_vector)

        inx = np.argmax(np.squeeze(input_vector))
        if inx == 0:
            return [1, 0]
        elif inx == 1:
            return [0, 1]
        else:
            raise Exception("Shouldn't happen")


def main():
    class1 = [(3, 4), (4.2, 5.3), (4, 3), (6, 5), (4, 6), (3.7, 5.8),
              (3.2, 4.6), (5.2, 5.9), (5, 4), (7, 4), (3, 7), (4.3, 4.3)]
    class2 = [(-3, -4), (-2, -3.5), (-1, -6), (-3, -4.3), (-4, -5.6),
              (-3.2, -4.8), (-2.3, -4.3), (-2.7, -2.6), (-1.5, -3.6),
              (-3.6, -5.6), (-4.5, -4.6), (-3.7, -5.8)]

    labeled_data = []
    for el in class1:
        labeled_data.append([el, [1, 0]])
    for el in class2:
        labeled_data.append([el, [0, 1]])

    np.random.shuffle(labeled_data)
    print(labeled_data[:10])

    data, labels = zip(*labeled_data)
    labels = np.array(labels)
    data = np.array(data)

    simple_network = NeuralNetwork(
        nodes=[2, 10, 2],
        weights=np.random.random(size=(1000,))
    )

    for _ in range(20):
        for i in range(len(data)):
            simple_network.train(data[i], labels[i])
    for i in range(len(data)):
        print(f"label: {labels[i]}")
        output = simple_network.run(data[i])
        print(f"output: {output}")


if __name__ == '__main__':
    main()
