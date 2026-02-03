class Optimizer:
    def __init__(self):
        self.target=None
        self.hooks=[]
    def setup(self,target):