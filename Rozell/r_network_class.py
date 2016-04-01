import numpy as np
import math
import pandas


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

    def generate_sparse(self):
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
        #error = np.array([[self.lamb, (a + b), a, b, sparsity]])
        error = np.array([[(a + b), a, b, sparsity]])

        return error

    #This method returns two arrays:
    #coefficients consists of the active coefficients
    #rfileds consists of the respective dictionary elements
    def get_rfields(self):
        indices = np.where(self.a > 0)
        rfields = self.dictionary[:, indices]
        coefficients = self.a[indices]

        return (coefficients, rfields)

    #This method takes a single lambda (as a list) or an array
    #of lambdas, then returns an error table where each row represents
    #different error measures at a particular lambda
    def error_table(self, lambdas):
        df = pandas.DataFrame() #DataFrame used for error table





        display = []  #List to hold rows of image data for grid (rfields scaled)
        display2 = [] #Unscaled rfields

        #For each value of lambda, set lambda and run Rozell on the given image
        for j in lambdas:
            display_row = []   #List to hold one row of image data (for display)
            display_row2 = []  #For display2
            self.set_lambda(j)
            self.generate_sparse()  #Calculate sparse code        
            ##Add row of error data to error table
            row = pandas.DataFrame(self.return_error())
            df = df.append(row)
            ##Add list of dictionary elements scaled by coefficients to list
            ##Also adding recostruction to list
            indices = np.flatnonzero(self.a)
            coeff = self.a[indices]
            rfields = self.dictionary[:, indices]
            reconstruction = np.dot(rfields, coeff).reshape((28,28))
            display_row.append(reconstruction)
            display_row2.append(reconstruction)
            for k in range(len(coeff)):
                display_row.append((coeff[k] * rfields[:, k]).reshape((28,28)))
                display_row2.append(rfields[:, k].reshape((28,28)))
            display.append(display_row)
            display2.append(display_row2)
        

        
            

        
        

    

    
        

