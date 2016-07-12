import unittest
import socket
import sys
import os

machine = socket.gethostname()
if (machine == 'Jack-PC'):
    #Big laptop
    sys.path.append('C:\\Users\\Jack2\\Desktop\\Git_Repos\\jc2\\Rozell\\' + \
                     'Walt_code\\mr')
    sys.path.append('C:\\Users\\Jack2\\Desktop\\Git_Repos\\jc2\\Rozell\\' + \
                     'Walt_code')
elif (machine == 'Tab'):
    #Little laptop
    sys.path.append('C:\\Users\\Jack\\Desktop\\Git_Repos\\jc2\\Rozell\\' + \
                     'Walt_code\\mr')
    sys.path.append('C:\\Users\\Jack\\Desktop\\Git_Repos\\jc2\\Rozell\\' + \
                     'Walt_code')
else:
    #PSU machines (linux lab)
    base1 = os.path.expanduser('~/dev/jc2/Rozell/Walt_code')
    sys.path.append(os.path.join(base1, 'mr'))
    sys.path.append(os.path.join(base1))

import pyximport
pyximport.install()
import modelBase



class test_get_params(unittest.TestCase):

    MODELBASE = mr.unsupervised.Lca

    def test_fail(self):
        model = self.MODELBASE()
        self.failUnless(False)
        

if __name__ == '__main__':
    unittest.main()




#class test_set_params(unittest.TestCase):


        
    
    
    

