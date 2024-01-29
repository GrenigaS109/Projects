import numpy as np
import random
import time
import os


def sigmoid(z):
    return 1.0 / (1.0 + np.e**(-z))


def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))


def sigmoid_vec(z):
    s_v = np.vectorize(sigmoid)
    return s_v(z)


def sigmoid_prime_vec(z):
    s_p_v = np.vectorize(sigmoid_prime)
    return s_p_v(z)


def desired_output(label):
    if label == 0:
        return np.array([0.0, 0.0])
    elif label == 1:
        return np.array([0.0, 1.0])
    elif label == 2:
        return np.array([1.0, 0.0])


def read_nand_example_data(file_path):
    x = np.loadtxt(file_path, delimiter=',')
    data = x[:,:2]
    labels = x[:,-1:]
    return list(zip(data, labels))

class NeuronLayer:
    def __init__(self, shape, is_input=False, is_output=False):
        # initializarea variabilelor membre
        self._is_input = is_input
        self._is_output = is_output
        self._shape = shape

        self._biases = np.random.uniform(-0.5, 0.5, shape)
        self._num_neurons = self._biases.size

        self._nabla_b = np.zeros(self._shape)
        self._activations = np.zeros(shape)
        self._zs = np.zeros(self._shape)
        self._deltas = np.zeros(self._shape)
        self._weights = None
        self._nabla_w = None
        self._input_layer = None
        self._output_layer = None

    def _init_weights(self):
        """
        Initializarea ponderilor. Acestea nu se initializeaza in functia __init__ pentru ca
        in functia __init__ nu este cunoscut stratul de intrare al stratului curent
        """
        if self._input_layer is not None:
            # init weights with random uniform distribution centered on 0
            self._weights = np.random.uniform(-0.5, 0.5, (self._num_neurons,
                                                          self._input_layer.get_num_neurons()))
            self._nabla_w = np.zeros(self._weights.shape)

    def get_activations(self):
        return self._activations

    def get_biases(self):
        return self._biases

    def set_activations(self, activations):
        self._activations = activations

    def get_shape(self):
        return self._shape

    def get_num_neurons(self):
        return self._num_neurons

    def get_weights(self):
        return self._weights

    def get_deltas(self):
        return self._deltas

    def feedforward(self):
        """
        Calculeaza iesirea stratului de neuroni, pe baza intrarilor acestuia,
        folosind formula: a^l = sigma(z),
        unde z = (W . a^l-1 ) + b
        """
        if self._is_input is False:
            prev_activations = np.reshape(self._input_layer.get_activations(),
                                          self._input_layer.get_num_neurons())
            biases = np.reshape(self._biases, self._num_neurons)
            self._zs = np.dot(self._weights, prev_activations) + biases
            self._activations = sigmoid_vec(self._zs)

    def backpropagate(self, label):
        """
        Metoda propagarii inapoi a erorii. Scopul aceste metode este acela de a
        determina erorile neuronilor din stratul curent, pe baza erorilor din
        stratul urmator. In cazul in care se calculeaza eroarea pe ultimul strat,
        aceasta depinde de iesirea reala a retelei si de iesirea dorita.
        nabla_b si nabla_w reprezinta doua variabile acumulatoare pentru erorile din
        stratul de neuroni.
        Delta^l = (Teta^l+1 . Delta^l+1) .* sigma'(z^l)
        Nabla_b += Delta^l
        Nabla_w += Delta^T . a^l
        """
        if self._is_output is False:
            next_weights = np.transpose(self._output_layer.get_weights())
            next_deltas = np.reshape(self._output_layer.get_deltas(),
                                     self._output_layer.get_num_neurons())
            self._deltas = np.multiply(np.dot(next_weights, next_deltas),
                                       np.reshape(sigmoid_prime_vec(self._zs),
                                                  np.dot(next_weights, next_deltas).shape))
        else:
            self._deltas = (self._activations - desired_output(label)) \
                           * sigmoid_prime_vec(self._zs)

        if self._is_input is False:
            # update nabla_b
            self._nabla_b = np.add(self._nabla_b, self._deltas.reshape(self._nabla_b.shape))

            # update nabla_w
            self._nabla_w += np.dot(np.atleast_2d(self._deltas).T,
                                    np.atleast_2d(self._input_layer.get_activations()))

    def reset_nabla_b(self):
        self._nabla_b = np.zeros(self._deltas.shape)

    def reset_nabla_w(self):
        if self._is_input is False:
            self._nabla_w = np.zeros(self._weights.shape)

    def update_weights(self, eta, lambda_, num_samples, mini_batch_size):
        """
        Actualizarea ponderilor stratului.
        Teta -> 1 - (eta * lambda) / m * Teta - (eta / n) * nabla_w
        eta - rata de invatare
        lambda - factorul de regularizare
        m - numarul de exemple de antrenare
        n - dimensiunea setului curent folosit in SGD
        """
        if self._is_input is False:
            self._weights = (1 - eta * lambda_ / num_samples) * self._weights - \
                            (eta / mini_batch_size) * self._nabla_w

    def update_bias(self, eta, mini_batch_size):
        """
        Actualizarea bias-ului
        b -> (eta/n) * nabla_b
        eta -rata de invatare
        n - dimensiunea setului curent folosit in SGD
        """
        self._biases -= (eta / mini_batch_size) * self._nabla_b.reshape(self._biases.shape)

    def set_input(self, input_layer):
        self._input_layer = input_layer
        self._init_weights()

    def set_output(self, output_layer):
        self._output_layer = output_layer


class NeuralNetwork:
    def __init__(self):
        self._layers = []
        self._training_data = None
        self._test_data = None

    def connect_layers(self):
        """
        Conecteaza straturile: cheama functiile set_intput si set_output
        ale straturilor adiacente
        """
        for i in range(0, len(self._layers)):
            if i < len(self._layers) - 1:
                self._layers[i].set_output(self._layers[i + 1])
            if i > 0:
                self._layers[i].set_input(self._layers[i - 1])

    def add_layer(self, layer):
        """
        Adauga un strat de neuroni in retea
        """
        self._layers.append(layer)

    def feedforward(self, network_input):
        """
        Calcularea iesirii retelei
        """
        self._layers[0].set_activations(network_input)
        for i in range(0, len(self._layers)):
            self._layers[i].feedforward()

    def get_activations(self):
        return self._layers[-1].get_activations()

    """
    Urmatoarele doua functii incarca datele de antrenare si datele de test
    pentru reteaua noastra
    """
    def load_nand_data(self, data_path):
        self._training_data = read_nand_example_data(data_path)

    def load_nand_testdata(self, data_path):
        self._test_data = read_nand_example_data(data_path)

    def SGD(self, eta, lambda_, epochs, mini_batch_size):
        """
        Stochastic Gradient Descend - metoda folosita pentru a antrena
        reteaua neurala.
        eta - rata de invatare
        lambda_ - regularizarea
        epochs - numarul de stagii in care vrem sa antrenam reteaua
        mini_batch_size - nr de exemple de antrenare pe care sa aplicam
        algoritmul la fiecare pas
        """
        for i in range(0, epochs):
            random.shuffle(self._training_data)
            start = time.time()
            for j in range(0, len(self._training_data) // mini_batch_size):
                self.update_mini_batch(self._training_data[j * mini_batch_size: (j + 1) * mini_batch_size],
                                       eta,
                                       lambda_,
                                       mini_batch_size)
            end = time.time()
            if self._test_data is None:
                print("Epoch {0} finished.".format(i))
            else:
                print("Epoch {0} : {1} \ {2} \ {3:.02f} s".
                      format(i, self.accuracy(self._test_data),
                             len(self._test_data), end - start))

    def update_mini_batch(self, batch, eta, lambda_, mini_batch_size):
        """
        Functia care antreneaza reteaua, folosind subseturi din setul de
        antrenare.
        Functionare:
            - aplica propagarea in fata pentru a determina iesirile
            - aplica propagarea in spate pentru a determina erorile
            - ajusteaza ponderile
        batch - subsetul de antrenare curent
        eta - rata de invatare
        lambda_ - factorul de regularizare
        mini_batch_size - marimea subsetului de antrenare curent
        """
        for i in range(0, len(self._layers)):
            self._layers[i].reset_nabla_b()
            self._layers[i].reset_nabla_w()
        for i in range(0, len(batch)):
            self.feedforward(batch[i][0])
            self._backpropagate(batch[i][1])
            self._update_parameters(eta, lambda_, mini_batch_size)

    def accuracy(self, source):
        """
        Calculeaza acuratetea retelei. In cazul exemplului cu circuitul de
        insumare SI-NU, am considerat ca reteaua returneaza un rezultat corect
        atunci cand ambele iesiri prezinta diferente de maximum 10% fata de
        iesirile dorite
        source - datele pe care se evalueaza algoritmul
        """
        acc = 0
        for test_data in source:
            self.feedforward(test_data[0])
            activations_output = self._layers[-1].get_activations().reshape(self._layers[-1].get_activations().size)
            d_output = desired_output(test_data[1])

            if abs(activations_output[0] - d_output[0]) < 0.1 and \
               abs(activations_output[1] - d_output[1]) < 0.1:
                acc += 1
        return acc

    def _backpropagate(self, label):
        """
        Propagarea in spate a erorii. Se apeleaza functia cu acelasi
        nume din fiecare strat de neuroni.
        label - iesirea corecta a retelei
        """
        for layer in self._layers[::-1]:
            layer.backpropagate(label)

    def _update_parameters(self, eta, lambda_, mini_batch_size):
        """
        Ajustarea parametrilor retelei
        eta - rata de invatare
        lambda_ - factorul de regularizare
        mini_batch_size: dimensiunea subsetului curent de antrenare
        """
        for i in range(0, len(self._layers)):
            self._layers[i].update_weights(eta,
                                           lambda_,
                                           len(self._training_data),
                                           mini_batch_size)
            self._layers[i].update_bias(eta, mini_batch_size)

def main():
    # Cream 3 straturi de neuroni
    l1 = NeuronLayer((2,), True, False)
    l2 = NeuronLayer((3,))
    l3 = NeuronLayer((2,), False, True)

    # Cream obiectul de tip retea neuronala
    network = NeuralNetwork()

    # Conectam cele 3 straturi la retea
    network.add_layer(l1)
    network.add_layer(l2)
    network.add_layer(l3)
    network.connect_layers()

    # Incarcam datele de antrenare si datele de test
    network.load_nand_data("nand_sum.txt")
    network.load_nand_testdata("nand_sum_test.txt")

    # Antrenam reteaua neurala
    network.SGD(0.5, 0.1, 10, 10)

    # Calculam iesirea retelei neuronale
    network.feedforward([0, 0])
    print("Pentru bitii b1 = 0 si b2 = 0, avem rezultatul: s = {0:.2f} si "
          "carry bit = {1:.2f}".format(network.get_activations()[1],
                                       network.get_activations()[0]))
    network.feedforward([0, 1])
    print("Pentru bitii b1 = 0 si b2 = 1, avem rezultatul: s = {0:.2f} si "
          "carry bit = {1:.2f}".format(network.get_activations()[1],
                                       network.get_activations()[0]))
    network.feedforward([1, 0])
    print("Pentru bitii b1 = 1 si b2 = 0, avem rezultatul: s = {0:.2f} si "
          "carry bit = {1:.2f}".format(network.get_activations()[1],
                                       network.get_activations()[0]))

    network.feedforward([1, 1])
    print("Pentru bitii b1 = 1 si b2 = 1, avem rezultatul: s = {0:.2f} si "
          "carry bit = {1:.2f}".format(network.get_activations()[1],
                                       network.get_activations()[0]))

if __name__ == "__main__":
    main()
