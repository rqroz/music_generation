from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense, Activation

"""
    Model Architecture:
        The model will be constructed using four different type of layers,
        described bellow:

        - LSTM Layers: Recurrent Neural Network layer that takes a sequence as
                       an input and can return either sequences or a matrix.

        - Droupout Layers: Regularisation technique that consists of setting a
                           fraction of input units to 0 at each update during
                           the training to prevent overfitting. The fraction is
                           determined by the parameter used with the layer.

        - Dense Layers: Also known as Fully Connected Layers, describes a fully
                        connected Neural Network layer where each input node is
                        connected to each output node.

        - The Activation Layer: Determines what activation function the neural
                                network will use to calculate the output of a
                                node.
"""

def create_network_model(network_input, n_vocab):
    """
        * Layers

        - input_shape: always required for the first layer, it is used to inform
                       the network the shape of the dataset it will be training.

        - units: positive integer representing the  dimensionality of the output
                 space (number of nodes for the layer).

        - rate: float between 0 and 1 meaning the fraction of the input units to
                drop during the training.

        - activation: name of activation function to use, or alternatively, a
                      Theano or TensorFlow operation.

        Note: To assure that the output of the network will map directly to the
              categorical classes, the amount of nodes in the last layer should
              always match the number of different outputs in the system.
    """
    layers = [
        LSTM(units=256, input_shape=(network_input.shape[1], network_input.shape[2]), return_sequences=True),
        Dropout(rate=0.3),
        LSTM(units=512, return_sequences=True),
        Dropout(rate=0.3),
        LSTM(units=256),
        Dense(units=256),
        Dropout(rate=0.3),
        Dense(units=n_vocab),
        Activation(activation='softmax'),
    ]

    """
        * Model
        - loss: function used to calculate the loss for each iteration of the
                training.
        - optimizer: function used to produce slightly better/faster results
                     by updating the model parameters such as Weights and Bias
                     values.

        Notes:
            - RMSProp:
                Adaptive learning rate method proposed by Geoff Hinton which
                divides the learning rate by an exponentially decaying average
                of squared gradients. It is usually a good choice for recurrent
                neural networks.

        Related Links:
            https://rdipietro.github.io/friendly-intro-to-cross-entropy-loss/
            https://www.quora.com/Why-is-it-said-that-RMSprop-optimizer-is-recommended-in-training-recurrent-neural-networks-What-is-the-explanation-behind-it
    """
    model = Sequential()
    for l in layers:
        model.add(l)

    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    return model
