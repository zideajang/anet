class Layer:
    def __init__(self) -> None:
        self.input = None
        self.output = None

    def forward(self,input):
        pass

    def backward(self,output_grad,learning_rate):
        pass

    