import numpy as np
import mr
from mr.unsupervised import Lca

net = Lca(50)
net.fit(X)
print (net.score(X))

