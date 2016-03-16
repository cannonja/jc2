import numpy as np
import math

###To do######
'''
-Implement methods to:
    -Save image, reconstruction, and components
    -Calculate and plot error function E(t)
        -Implement Rozell's E(t) and Walt's E(t)
'''

class r_network:
    'Class to represent Rozell LCA network'

    def __init__(self, D):
        self.dictionary = D
        self.s = None
        self.b = None
        self.a = None

    #Takes the stimulus signal, sets s, then computes b according to Rozell
    def set_stimulus(self, signal):
        self.s = np.asarray(signal, dtype=float)
        self.b = np.dot(np.transpose(self.dictionary), self.s)
        #print("set_stimulus\ns = ", self.s.shape, sum(self.s), "\nb = ",
        #      self.b.shape, sum(self.b))

    #Takes u, lamb, and t_type then returns a according to Rozell
    #u is the internal state variable, lamb is lambda (threshold),
    #and t_type is the soft or hard thresholding option ('S' or 'H')
    def thresh(self, u, lamb, t_type):
        if (u < lamb):
            return 0
        if (t_type == 'H'):
            return u
        if (t_type == 'S'):
            return u - lamb

    def return_sparse(self, lamb, tau, delta, u_stop, t_type):
        u = np.zeros(self.b.shape)
        self.a = u.copy()  #Initialize a by setting equal to u
        inhibit = np.dot(np.transpose(self.dictionary), self.dictionary)\
                        - np.eye(self.dictionary.shape[1])
        udot = (1/tau) * (self.b - u - np.dot(inhibit, self.a))

        '''
        print("b", self.b)
        print("u", u)
        print("a", self.a)
        #print("b - u", (self.b - u))
        #print("inhibit * a", np.dot(inhibit, self.a))
        print("udot", udot)
        '''
        
        
        
        loop_flag = True
        
        #Generate vector self.a
        #print(self.b)
        #debug = []
        num_iter = 0
        while (loop_flag):
            #print("loop start....\n")
            u = u + (udot * delta)
            #print("u", u)
            #print("b - u", (self.b - u))

            #Update a vector
            for i in range(len(self.a)):
                self.a[i] = self.thresh(u[i], lamb, t_type)
            #print("a", self.a)
            #print("inhibit * a", np.dot(inhibit, self.a))            
            
            udot = (1/tau) * (self.b - u - np.dot(inhibit, self.a))            
            #print("udot", udot)

            udot_length = math.sqrt(np.dot(np.transpose(udot),udot))
            num_iter += 1
            if (udot_length < u_stop):
                loop_flag = False
                print(num_iter)
            
            
            #print("\nloop end.....\n")
            #debug.append({ 'a': self.a.copy(), 'u': u.copy(), 'udot': ... })
        '''
        df = pandas.DataFrame(debug)
        print df.to_string()
        '''
 
        return self.a
            

        
        

    

    
        

