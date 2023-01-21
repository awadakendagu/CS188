import nn

class PerceptronModel(object):
    def __init__(self, dimensions):
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimensions` is the dimensionality of the data.
        For example, dimensions=2 would mean that the perceptron must classify
        2D points.
        """
        self.weight = nn.Parameter(1, dimensions)
        #weight为1*dimension的Parameter

    def get_weights(self):
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.weight

    def run(self, x):
        """
        Calculates the score assigned by the perceptron to a data point x.

        Inputs:
            x: a node with shape (1 x dimensions)
        Returns: a node containing a single number (the score)
        """
        return nn.DotProduct(self.weight, x)
    	#点乘

    def get_prediction(self, x):
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        prod = nn.as_scalar(self.run(x))
        if prod >=0:
            return 1
        else:
            return -1
        #将点乘转化成一个float

    def train(self, dataset):
        """
        Train the perceptron until convergence.
        """
        batch_size = 1
        #每次学习一组数据
        change_flag = True
        while change_flag:
            change_flag = False
            for x, y in dataset.iterate_once(batch_size):
                #更新
                result = self.get_prediction(x)
                if result != nn.as_scalar(y):
                    self.weight.update(nn.Constant(nn.as_scalar(y)*x.data), 1)
                    change_flag = True

class RegressionModel(object):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """
    def __init__(self):
        # Initialize your model parameters here
        self.batch_size = 50
        self.alpha = 0.01
        self.weight1 = nn.Parameter(1, 128)
        self.bias1 = nn.Parameter(1, 128)
        self.weight2 = nn.Parameter(128, 64)
        self.bias2 = nn.Parameter(1, 64)
        self.weight3 = nn.Parameter(64, 1)
        self.bias3 = nn.Parameter(1, 1)
        self.params = [self.weight1, self.bias1, self.weight2, self.bias2, self.weight3, self.bias3]


    def run(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        first_output = nn.ReLU(nn.AddBias(nn.Linear(x, self.weight1), self.bias1))
        second_output = nn.ReLU(nn.AddBias(nn.Linear(first_output, self.weight2), self.bias2))
        output= nn.AddBias(nn.Linear(second_output, self.weight3), self.bias3)
        return output

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a loss node
        """
        y_hat = self.run(x)
        return nn.SquareLoss(y_hat, y)

    def train(self, dataset):
        """
        Trains the model.
        """
        loss = 1
        while loss >=0.015:
            for x, y in dataset.iterate_once(self.batch_size):
                loss = self.get_loss(x, y)
                grads = nn.gradients(loss, self.params)
                loss = nn.as_scalar(loss)
                for i in range(len(self.params)):
                    self.params[i].update(grads[i], -self.alpha)


class DigitClassificationModel(object):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Initialize your model parameters here
        self.batch_size = 100
        self.alpha = 0.1
        self.weight1 = nn.Parameter(784, 256)
        self.bias1 = nn.Parameter(1, 256)
        self.weight2 = nn.Parameter(256, 128)
        self.bias2 = nn.Parameter(1, 128)
        self.weight3 = nn.Parameter(128, 64)
        self.bias3 = nn.Parameter(1, 64)
        self.weight4 = nn.Parameter(64, 10)
        self.bias4 = nn.Parameter(1, 10)
        self.params = [self.weight1, self.bias1, self.weight2, self.bias2, self.weight3, self.bias3, self.weight4, self.bias4]

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Your model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.

        Inputs:
            x: a node with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
        """
        first_output = nn.ReLU(nn.AddBias(nn.Linear(x, self.weight1), self.bias1))
        second_output = nn.ReLU(nn.AddBias(nn.Linear(first_output, self.weight2), self.bias2))
        third_output = nn.ReLU(nn.AddBias(nn.Linear(second_output, self.weight3), self.bias3))
        output = nn.AddBias(nn.Linear(third_output, self.weight4), self.bias4)
        return output

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss node
        """
        y_hat = self.run(x)
        return nn.SoftmaxLoss(y_hat, y)

    def train(self, dataset):
        """
        Trains the model.
        """
        valid_acc = 0
        while valid_acc < 0.98:
            for x, y in dataset.iterate_once(self.batch_size):
                loss = self.get_loss(x, y)
                grads = nn.gradients(loss, self.params)
                loss = nn.as_scalar(loss)
                for i in range(len(self.params)):
                    self.params[i].update(grads[i], -self.alpha)
            valid_acc = dataset.get_validation_accuracy()

class LanguageIDModel(object):
    """
    A model for language identification at a single-word granularity.

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Our dataset contains words from five different languages, and the
        # combined alphabets of the five languages contain a total of 47 unique
        # characters.
        # You can refer to self.num_chars or len(self.languages) in your code
        self.num_chars = 47
        self.languages = ["English", "Spanish", "Finnish", "Dutch", "Polish"]

        # Initialize your model parameters here
        self.batch_size = 100
        self.alpha = 0.1
        self.initial_weight = nn.Parameter(self.num_chars, 256)
        self.initial_bias = nn.Parameter(1, 256)
        self.single_weight = nn.Parameter(self.num_chars, 256)
        self.front_weight = nn.Parameter(256, 256)
        self.bias = nn.Parameter(1, 256)
        self.output_weight = nn.Parameter(256, len(self.languages))
        self.output_bias = nn.Parameter(1, len(self.languages))
        self.params = [self.initial_weight, self.initial_bias, self.single_weight, self.front_weight, self.bias, self.output_weight, self.output_bias]

    def run(self, xs):
        """
        Runs the model for a batch of examples.

        Although words have different lengths, our data processing guarantees
        that within a single batch, all words will be of the same length (L).

        Here `xs` will be a list of length L. Each element of `xs` will be a
        node with shape (batch_size x self.num_chars), where every row in the
        array is a one-hot vector encoding of a character. For example, if we
        have a batch of 8 three-letter words where the last word is "cat", then
        xs[1] will be a node that contains a 1 at position (7, 0). Here the
        index 7 reflects the fact that "cat" is the last word in the batch, and
        the index 0 reflects the fact that the letter "a" is the inital (0th)
        letter of our combined alphabet for this task.

        Your model should use a Recurrent Neural Network to summarize the list
        `xs` into a single node of shape (batch_size x hidden_size), for your
        choice of hidden_size. It should then calculate a node of shape
        (batch_size x 5) containing scores, where higher scores correspond to
        greater probability of the word originating from a particular language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
        Returns:
            A node with shape (batch_size x 5) containing predicted scores
                (also called logits)
        """
        temp = nn.ReLU(nn.AddBias(nn.Linear(xs[0], self.initial_weight), self.initial_bias))
        for char in xs[1:]:
            temp = nn.ReLU(nn.AddBias(nn.Add(nn.Linear(char, self.single_weight), nn.Linear(temp, self.front_weight)), self.bias))
        output = nn.AddBias(nn.Linear(temp, self.output_weight), self.output_bias)
        return output

    def get_loss(self, xs, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 5). Each row is a one-hot vector encoding the correct
        language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
            y: a node with shape (batch_size x 5)
        Returns: a loss node
        """
        y_hat = self.run(xs)
        return nn.SoftmaxLoss(y_hat, y)

    def train(self, dataset):
        """
        Trains the model.
        """
        valid_acc = 0
        while valid_acc < 0.85:
            for x, y in dataset.iterate_once(self.batch_size):
                loss = self.get_loss(x, y)
                grads = nn.gradients(loss, self.params)
                loss = nn.as_scalar(loss)
                for i in range(len(self.params)):
                    self.params[i].update(grads[i], -self.alpha)
            valid_acc = dataset.get_validation_accuracy()
