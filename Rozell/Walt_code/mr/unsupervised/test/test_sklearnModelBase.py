import unittest
import socket
import sys
import os

machine = socket.gethostname()
if (machine == 'Jack-PC'):
    #Big laptop
    sys.path.append('C:\\Users\\Jack2\\Desktop\\Git_Repos\\jc2\\Rozell\\' + \
                     'Walt_code\\mr')
elif (machine == 'Tab'):
    #Little laptop
    sys.path.append('C:\\Users\\Jack2\\Desktop\\Git_Repos\\jc2\\Rozell\\' + \
                     'Walt_code\\mr')
else:
    #PSU machines (linux lab)
    base1 = os.path.expanduser('~/dev/jc2/Rozell/Walt_code')
    sys.path.append(os.path.join(base1, 'mr'))

import pyximport
pyximport.install()    
import modelBase



class test_get_params(unittest.TestCase):
    
    def test_fail(self):    
        self.failUnless(False)
        

if __name__ == '__main__':
    unittest.main()
   
    


#class test_set_params(unittest.TestCase):


        
    
    
    

