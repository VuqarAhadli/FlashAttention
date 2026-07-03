import math

class flashAttetionmodule:
    def __init__(self,values,scores,block_size):                                                                                                  

        self.values = values
        self.scores = scores
        self.block_size = block_size

        self.max_in_block = None #m
        self.sum_weighted = None #n
        self.denominator = None #l

        self.alpha = None

        self.prev_max = None
        self.prev_sum = None
        self.prev_denominator = None



    def calculate(self):
        for i in range(0,len(self.scores),self.block_size):

            block_values = self.values[i:i+self.block_size]
            block_scores = self.scores[i:i+self.block_size]
            
            self.max_in_block = max(block_scores)

            if self.prev_max is None:
                self.prev_max = self.max_in_block

            new_max = max(self.prev_max,self.max_in_block)

            self.alpha = math.exp(self.prev_max - new_max)

            weights = [math.exp(_-new_max) for _ in block_scores ]
            
            self.denominator = sum(weights)
            self.sum_weighted = sum([weights[i] * block_values[i] for i in range(len(block_values))])

            if self.prev_sum is not None:
                self.sum_weighted += self.alpha * self.prev_sum
                self.denominator += self.alpha * self.prev_denominator


            self.prev_sum = self.sum_weighted
            self.prev_denominator = self.denominator
            self.prev_max = new_max

            self.alpha = None

        return self.sum_weighted / self.denominator
      
if __name__=="__main__":
  attn = flashAttetionmodule([10,20,30,40,50,60],[2,1,3,0,-1,4],2)
  
  print(f"{attn.calculate():.4f}")

"""
Algorithm
m - max value in a block
n - weighted sum
l - denominator
alpha - if maximum value in the previous block is bigger the weighted sum and denominator are normalised with a factor alpha e^(prev_max - new_max)

The previous numerator and denominator are rescaled by alpha before
adding the current block's contributions. This keeps all exponentials
numerically stable while producing the same result as a full softmax.

"""
