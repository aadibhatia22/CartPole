
import numpy as np
class NeuralNetwork():


    """
    ARCHITECTURE IS FOR EXAMPLE
    [4,8,4,2]
    4 is input size
    8,4 are number of layers
    2 is output size
    """
    def __init__(self, learning_rate:float, discount_factor:float, architecture = np.ndarray):

        """
         Input of
          [4,8,4,2]
          is 3 layers since
          w1 makes 4 -> 8
          w2 makes 8 -> 4
          w3 makes 4 -> 2 (output)

          you will have 3 weights but the final layer will have 2 neurons WHICH WILL OUT PUT a [2,] WHICH IS YOUR OUTPUT
          """
        self.number_of_layers = architecture.shape[0] - 1
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.bias =  np.empty(self.number_of_layers+1, dtype=object)
        self.architecture = architecture
        self.weight = np.empty(self.number_of_layers+1, dtype=object)
        rng = np.random.default_rng()
        """
        MAKING THE NEURONS AND BIASES
        LETS SAY I GET AN INPUT OF [4,8,4,2]
        THEN W1[] and B1[] will be multiplied to their respective neurons to get a combined shape of [8,1] so 
        Ex.
         W1 = [8,4]
         hidden layer form W2 = [8,4] (8 neurons with 4 parameters) * input(hidden layer from before) [4,1]  + biases to each element gives a hidden layer of [8,1]
         Then 
         hidden layer form W2 = [4,8] * [8,1] +biases to each element 
         Then put into w3 and you will get output shape of 2
         
         This works since if you have 8 neurons your hidden layer will be [8,1] and to accommodate that the next 
         layer that has 4 neurons just has 8 "parameters" per neuron so matrix multiplication can still happen
        """
        for i in range(self.architecture.shape[0]):
           if i == 0:
                self.weight[i] = None
                self.bias[i] = None
           else:
               """
               lets say i have [4,8,5,2]
               
               input is 4,
               its technically i = 0 in bias and weight so bias and weight are none
               
               
               it goes to the first layer of neurons that have 
               [8 neurons, 4 params in each neuron] *[4,1 input]
                ^^^^^^^^ is stored in weight[1][#neuron number] and its bias is bias[1][#neuron number]
               -> [8,1]
               goes into second layer fo neurons that are
               [5 neurons with 8] * [8,1] 
               ^^^^^^^^ is stored in weight[2][#neuron number] and its bias is bias[1][#neuron number]
               -> [5,1]
               
               then last later
               [2 neurons with 5]  * [5,1]
                ^^^^^^^^ is stored in weight[3][#neuron number] and its bias is bias[3][#neuron number]
                                             ^ last element in the array

               ->[2,1] WHICH IS OUTPUT
               
               """
               self.weight[i] = rng.random((architecture[i],architecture[i-1]))
               self.bias[i] = rng.random((architecture[i],1))

    def print_network(self, show_values=True, precision=3):
        print("\n" + "=" * 50)
        print("NEURAL NETWORK STRUCTURE")
        print("=" * 50)

        for i in range(len(self.architecture)):
            # Input layer
            if i == 0:
                print(f"\nLayer {i} (INPUT)")
                print(f" Neurons: {self.architecture[i]}")
                continue

            # Hidden / Output layers
            layer_type = "OUTPUT" if i == len(self.architecture) - 1 else "HIDDEN"

            print(f"\nLayer {i} ({layer_type})")
            print(f" Neurons: {self.architecture[i]}")

            # Weights
            W = self.weight[i]
            print(f" Weights shape: {W.shape}")

            if show_values:
                print(" Weights:")
                print(np.round(W, precision))

            # Bias
            b = self.bias[i]
            print(f" Bias shape: {b.shape}")

            if show_values:
                print(" Bias:")
                print(np.round(b, precision))

            print("-" * 50)

        print("=" * 50 + "\n")




