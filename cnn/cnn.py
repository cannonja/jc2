from mr.datasets.mnist import MnistDataset
from mr.learn.scaffold import Scaffold
from mr.learn.convolve import ConvolveLayer
from mr.learn.unsupervised.lca import Lca
from mr.learn.supervised.perceptron import Perceptron

## Load images
nload = 7
train, test, vp = MnistDataset(nload).split(nload * 1 // 7)

## Set up model
print ("Building model")
model = Scaffold()
c = ConvolveLayer(layer = Lca(15), visualParams = vp, convSize = 7,
            convStride = 3)
model.layer(c)
model.layer(Perceptron())

## Train and test model
print ("Training model")
model.fit(*train)
print (model.layers[0].nOutputs)
print (model.layers[0].nOutputsConvolved)
path = 'visualize.png'
path2 = 'visualize1.png'
model.visualize(vp, path)
model.visualize(vp, path2, inputs = test[0][0])
print ("Testing model")
print (model.score(*test))
