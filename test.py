from neuronalNetwork.neuronalNetwork import NeuronalNetwork
import numpy as np

testNN = NeuronalNetwork([4, 2, 2],
                         np.array([
                                 [0,0, 0, 0, 0.15, 0.25, 0, 0],
                                 [0,0, 0, 0, 0.2, 0.3, 0, 0],
                                 [0,0, 0, 0, 0.35, 0.35, 0, 0],
                                 [0,0, 0, 0, 0, 0, 0.6, 0.6],
                                 [0,0, 0, 0, 0, 0, 0.4, 0.5],
                                 [0,0, 0, 0, 0, 0, 0.45, 0.55],
                                 [0,0, 0, 0, 0, 0, 0, 0],
                                 [0,0, 0, 0, 0, 0, 0, 0]]))
trainingData = [
    {"input": [1, 0.1, 0.5, 0.5],
     "output": [1, 0.99]},
    {
    "input": [0.1, 0.05, 1, 1],
     "output": [0.99, 0.01]}]

#testNN.train(trainingData)

print("Network 2")

testNN2 = NeuronalNetwork([2,3,1],
                           np.array([
                                [0, 0, 0.8,0.4, 0.3,0],
                                [0, 0, 0.2, 0.9,0.5,0],
                                [0, 0, 0, 0,0,0.3],
                                [0, 0, 0, 0,0,0.5],
                                [0, 0, 0, 0,0,0.9],
                                [0, 0, 0, 0,0,0]]),learnRate = 0.9)
                           
trainingData2 = [
    {"input": [1, 1],
     "output": [0]},
    {"input": [0, 1],
     "output": [1]},
    {"input": [1, 0],
     "output": [1]},
    {"input": [0, 0],
     "output": [0]},
 
]
    
testData2 = [
      {"input": [1, 1],
     "output": [0]},
    {"input": [0, 1],
     "output": [1]},
    {"input": [1, 0],
     "output": [1]},
    {"input": [0, 0],
     "output": [0]},
   
 
]
  
error = 1

x=0
while(x <500000):
    error = testNN2.train(trainingData2);
    x+=1
    print(error)


testNN2.test(trainingData2);





