import numpy as np

class LogicGate:
    def __init__(self):
        self.w1 = None
        self.w2 = None
        self.th = None
        self.out = None
        self.x1 = self.x2 = None


    def print_output(self, gate):
        if gate == 'AND':
            print(f'{self.x1} AND {self.x2} is {self.out}')
        elif gate == 'OR':    
            print(f'{self.x1} OR {self.x2} is {self.out}')
        elif gate == 'NAND':    
            print(f'{self.x1} NAND {self.x2} is {self.out}')
        elif gate == 'NOR':    
            print(f'{self.x1} NOR {self.x2} is {self.out}')
    

    def or_gate(self, x1, x2):
        self.w1 = 0.5
        self.w2 = 0.5
        self.th = 0.0

        self.x1 = x1
        self.x2 = x2

        # use numpy instead
        x = np.array([x1, x2])
        w = np.array([self.w1, self.w2])

        # if x1*self.w1 + x2*self.w2 > self.th:
        if np.sum(x*w) > self.th:
            self.out = 1
            return 1
        else:
            self.out = 0
            return 0   
             
    def Nor_gate(self, x1, x2):
        self.w1 = -0.5
        self.w2 = -0.5
        self.th = 0.0

        self.x1 = x1
        self.x2 = x2

        # use numpy instead
        x = np.array([x1, x2])
        w = np.array([self.w1, self.w2])

        # if x1*self.w1 + x2*self.w2 > self.th:
        if np.sum(x*w) > self.th:
            self.out = 1
            return 1
        else:
            self.out = 0
            return 0      


    def and_gate(self, x1, x2):
        self.w1 = 0.5
        self.w2 = 0.5
        self.th = 0.99

        self.x1 = x1
        self.x2 = x2

        x = np.array([x1, x2])
        w = np.array([self.w1, self.w2])
        
        # if x1*self.w1 + x2*self.w2 > self.th:
        if np.sum(x*w) > self.th:
            self.out = 1
            return 1
        else:
            self.out = 0
            return 0
        
    def Nand_gate(self, x1, x2):
        self.w1 = -0.5
        self.w2 = -0.5
        self.th = -0.99

        self.x1 = x1
        self.x2 = x2

        x = np.array([x1, x2])
        w = np.array([self.w1, self.w2])
        
        # if x1*self.w1 + x2*self.w2 > self.th:
        if np.sum(x*w) > self.th:
            self.out = 1
            return 1
        else:
            self.out = 0
            return 0
        
        

# testing code
if __name__ == '__main__':
    logic_gate = LogicGate()

    logic_gate.and_gate(1, 0)
    logic_gate.print_output('AND')

    logic_gate.or_gate(1, 0)
    logic_gate.print_output('OR')

    logic_gate.Nand_gate(1, 0)
    logic_gate.print_output('NAND')

    logic_gate.Nor_gate(1, 0)
    logic_gate.print_output('NOR')