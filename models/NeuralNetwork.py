
import numpy as np
from IPython.core.pylabtools import retina_figure
import math
from state_action_pair import StateActionPair
from agent_action import AgentAction
class NeuralNetwork():

    def toggleTraining(self):
        self.isTraining = not self.isTraining;



    """
    ARCHITECTURE IS FOR EXAMPLE
    [4,8,4,2]
    4 is input size
    8,4 are number of layers
    2 is output size
    """
    def __init__(self, learning_rate:float, discount_factor:float, architecture: np.ndarray, random_number1:float =-1.0, random_number2:float=1.0):
        # preparing to store state action pairs
        self.state_action_pairs: list[StateActionPair] = []

        self.isTraining = True;


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
        self.random_number1 = random_number1
        self.random_number2 = random_number2
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
               ^^^^^^^^ is stored in weight[2][#neuron number] and its bias is bias[2][#neuron number]
               -> [5,1]
               
               then last later
               [2 neurons with 5]  * [5,1]
                ^^^^^^^^ is stored in weight[3][#neuron number] and its bias is bias[3][#neuron number]
                                             ^ last element in the array

               ->[2,1] WHICH IS OUTPUT
               
               """
               self.weight[i] = rng.uniform(self.random_number1, self.random_number2,(architecture[i],architecture[i-1]))
               self.bias[i] = rng.uniform(self.random_number1, self.random_number2, (architecture[i],1))

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

    def sigmoid(self, input):
        return 1 / (1 + np.exp(-input))

    def print_prediction_steps(self, input: np.array, precision=3):
        if np.shape(input) != (self.architecture[0],):
            print("INVALID INPUT SHAPE")
            print(f" Expected: {(self.architecture[0],)}")
            print(f" Got:      {np.shape(input)}")
            return -1

        input = input.reshape((self.architecture[0], 1))

        print("\n" + "=" * 60)
        print("FORWARD PASS")
        print("=" * 60)
        print(f"Architecture: {self.architecture}")
        print(f"Starting input shape: {input.shape}")
        print("Starting input:")
        print(np.round(input, precision))

        for i in range(1, len(self.architecture)):
            layer_type = "OUTPUT" if i == len(self.architecture) - 1 else "HIDDEN"
            W = self.weight[i]
            b = self.bias[i]

            print("\n" + "-" * 60)
            print(f"Layer {i} ({layer_type})")
            print(f"Calculating: weight[{i}] @ input + bias[{i}]")
            print(f"weight[{i}] shape: {W.shape}")
            print(f"input shape:     {input.shape}")
            print(f"bias[{i}] shape:   {b.shape}")

            z = W @ input + b

            print(f"Result before activation shape: {z.shape}")
            print("Result before activation:")
            print(np.round(z, precision))

            if i != len(self.architecture) - 1:
                input = np.maximum(0, z)
                print("Activation: ReLU, max(0, value)")
            else:
                input = self.sigmoid(z)
                print("Activation: Sigmoid, 1 / (1 + exp(-value))")

            print("Result after activation:")
            print(np.round(input, precision))

        print("\n" + "=" * 60)
        print("FINAL OUTPUT")
        print("=" * 60)
        print(np.round(input, precision))
        print("=" * 60 + "\n")

        return input

    def make_prediction(self, input:np.array):
        if np.shape(input) != (self.architecture[0],):
            return -1;
        input = input.reshape((self.architecture[0], 1))
        for i in range(len(self.architecture)):
            if i == 0:
                pass
            else:
                # @ does matrix multiplication
                input = self.weight[i]@input + self.bias[i]
                if(i!=len(self.architecture)-1):
                    #RELU
                    input = np.where(input <0, 0, input)
                elif(i==len(self.architecture)-1):
                    #SIGMOID
                    return self.sigmoid(input)
        return -1;

    #will use the make_prediction method, but then store state, action, and probability of that action for back prop
    #if this is the last observation from an episode it will run the backprop with the current stored-state action pair

    #assuming that make_prediction was run before
    """
    make_prediction and then returns the chance of going left
    after in the sim a random action is chosen
    then we use this function to log the state, the action taken, probability of that action
    
    0 -> LEFT
    1-> RIGHT
    
    if it is the final action it runs back propagation and does the training
    
    """



    """storing state action pairs, takes in the agent's actions, and the input that produced that input"""
    def record_action(self, input:np.ndarray, actionTaken: AgentAction, final_observation: bool = False, reward:float = None):
        probability_of_left = self.make_prediction(input, final_observation)

        # LEARN MORE ABOUT d_log_pt
        if(actionTaken == AgentAction.LEFT):
            self.state_action_pairs.append(StateActionPair(input, actionTaken, AgentAction.LEFT - probability_of_left))
        if(actionTaken == AgentAction.RIGHT):
            self.state_action_pairs.append(StateActionPair(input, actionTaken, AgentAction.Right - probability_of_left))

        if(final_observation):
            self.backpropagation(reward)


    def backpropagation(self, reward:float):
        #loss -> gradient L to W1 and L and Bias then do W - learning_rate*bias
        return -1;




    #### HELPER METHODS

    def relu_derivative(self, input:np.array):
        #1 if x is positive 0 otherwise
        return 1*(input>0)

    def first_layer_delta():
        return -1;