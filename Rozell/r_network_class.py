import numpy as np
import math
import pandas

###To do######
'''
-Implement methods to:
    -Save image, reconstruction, and components
    -Calculate and plot error function E(t)
        -Implement Rozell's E(t) and Walt's E(t)
    -
'''

class r_network:
    'Class to represent Rozell LCA network'

    def __init__(self, D):
        self.dictionary = D
        self.s = None
        self.b = None
        self.a = None
        self.lamb = None
        self.tau = None
        self.delta = None
        self.u_stop = None
        self.t_type = None

    def scale(self, num):
        self.s *= num
        self.dictionary *= num
        self.b = np.dot(np.transpose(self.dictionary), self.s)
        


    #Takes the stimulus signal, sets s, then computes b according to Rozell
    def set_stimulus(self, signal):
        self.s = np.asarray(signal, dtype=float)
        self.b = np.dot(np.transpose(self.dictionary), self.s)


    def set_lambda(self, lamb):
        self.lamb = lamb

    def set_tau(self, tau):    
        self.tau = tau

    def set_delta(self, delta):
        self.delta = delta

    def set_ustop(self, u_stop):
        self.u_stop = u_stop

    def set_ttype(self, t_type):
        self.t_type = t_type

    def set_parameters(self, lamb, tau, delta, u_stop, t_type):
        self.set_lambda(lamb)
        self.set_tau(tau)
        self.set_delta(delta)
        self.set_ustop(u_stop)
        self.set_ttype(t_type)


    #Takes u then returns a according to Rozell
    #u is the internal state variable, lamb is lambda (threshold),
    #and t_type is the soft or hard thresholding option ('S' or 'H')
    def thresh(self, u):
        if (u < self.lamb):
            return 0
        if (self.t_type == 'H'):
            return u
        if (self.t_type == 'S'):
            return u - self.lamb

    def return_sparse(self):
        u = np.zeros(self.b.shape)
        self.a = u.copy()  #Initialize a by setting equal to u
        self.scale(1/255)
        inhibit = np.dot(np.transpose(self.dictionary), self.dictionary)\
                        - (np.eye(self.dictionary.shape[1]) / 255)
        udot = (1/self.tau) * (self.b - u - np.dot(inhibit, self.a))
        loop_flag = True
        
        #Generate vector self.a
        #debug = []
        len_u = len(u)
        while (loop_flag):
            u = u + (udot * self.delta)
            #Update a vector
            for i in range(len(self.a)):
                self.a[i] = self.thresh(u[i])           
            
            udot = (1/self.tau) * (self.b - u - np.dot(inhibit, self.a))            
            udot_length = math.sqrt(np.dot(udot,udot))
            if ((udot_length / len_u) < (self.u_stop / len_u)):
                loop_flag = False

            #debug.append({ 'a': self.a.copy(), 'u': u.copy(), 'udot': ... })

        self.scale(255)
        '''
        df = pandas.DataFrame(debug)
        print df.to_string()
        '''
 
        return self.a

    #Returns 1-D numpy array containing E(t), the left-hand operand of E(t),
    #and the right-hand operand of E(t).
    #That is the operands of the right-hand side of the error equation
    def return_error(self):
        stim = self.s
        recon = np.dot(self.dictionary, self.a)
        resid = stim - recon
        a = 0.5 * math.sqrt(np.dot(resid, resid))
        b = None
        sparsity = len(self.a[self.a > 0]) / len(self.a)
        norm1 = sum(abs(self.a))
        cost = 0

        if (self.t_type == 'S'):
            cost = norm1
        elif (norm1 > self.lamb):
            cost = self.lamb / 2

        b = self.lamb * cost        
        error = np.array([[self.lamb, (a + b), a, b, sparsity]])

        return error

    #This method returns two arrays:
    #coefficients consists of the active coefficients
    #rfileds consists of the respective dictionary elements
    def get_rfields(self):
        indices = np.where(self.a > 0)
        rfields = self.dictionary[:, indices]
        coefficients = self.a[indices]

        return (coefficients, rfields)

        
            

        
        

    

    
        

