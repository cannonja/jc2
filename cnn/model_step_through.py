#from mr.datasets.mnist import MnistDataset
#from mr.learn.scaffold import Scaffold
#from mr.learn.convolve import ConvolveLayer
#from mr.learn.convolve import PoolLayer
#from mr.learn.unsupervised.lca import Lca
#from mr.learn.supervised.perceptron import Perceptron
import numpy as np
import pandas as pd
from collections import OrderedDict




def _partial_fit(x, convSize, convStride, convH, convW, imH, imW, imC):    
    
    buf = np.zeros(convSize * convSize * imC)
    db = []
    count = 1
    for iy in range(convH):
        for ix in range(convW):
            mx = ix * convStride
            my = iy * convStride
            rlen = convSize * imC
            for r in range(convSize):
                xStart = imC * (imW * (my + r) + mx)
                bStart = r * rlen                
                buf[bStart:bStart + rlen] = x[xStart:xStart + rlen]
                if (r + 1) % 3 == 0:
                    #print(buf, iy, ix, mx, my)
                    row = OrderedDict([('buf', buf.copy()), ('patch_num', count), ('convRow', iy), ('convCol', ix), ('mx', mx), ('my', my)])
                    db.append(row)
                    count += 1

            #self._layer._partial_fit(buf, None)
    return pd.DataFrame(db, columns = row.keys())
            
x = np.array([[3,3,2,1,0], [0,0,1,3,1], [3,1,2,2,3], [2,0,0,2,2], [2,0,0,0,1]])           
convSize = 3
convStride = 1
convH = 3
convW = 3
imH = 5
imW = 5
imC = 1     

df = _partial_fit(x.flatten(), convSize, convStride, convH, convW, imH, imW, imC)   
print(df)

















### Load images
#nload = 10
##train, test, vp = MnistDataset(nload).split(nload * 1 // 7)
#train, test, vp = MnistDataset(nload).split(5)

## Set up model
#print ("Building model")
#model = Scaffold()
#c = ConvolveLayer(layer = Lca(15), visualParams = vp, convSize = 7,
#            convStride = 3)
#c._init(len(train[0][0]), None)
#model.layer(c)






'''
#p = PoolLayer(visualParams = c.visualParams)
#model.layer(p)
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
'''
