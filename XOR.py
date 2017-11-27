from networkComparison import NetworkComparer

trainingData = [
    {"input": [1, 1],
     "output": [0]},
    {"input": [0, 1],
     "output": [1]},
    {"input": [1, 0],
     "output": [1]},
    {"input": [0, 0],
     "output": [0]}]

testData = [
    {"input": [1, 1],
     "output": [0]},
    {"input": [0, 1],
     "output": [1]},
    {"input": [1, 0],
     "output": [1]},
    {"input": [0, 0],
     "output": [0]}]

networkData = [
    {"fnc_activate_type": "log",
     "learn_rate": [0.9],
     "max_iterations": 10000,
     "layers": [2, 3, 1]},
]

networkComparer = NetworkComparer(
    networkData=networkData,
    trainingData=trainingData,
    testData=testData
)
networkComparer.compareNetworks()

