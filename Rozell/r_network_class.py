import numpy as np

class r_network:
    'Class to represent Rozell LCA network'

    def __init__(self, D):
        self.dictionary = D
        self.s = None
        self.b = None
        self.a = None

    #Takes the stimulus signal, sets s, then computes b according to Rozell
    def set_stimulus(signal):
        self.s = np.asarray(signal)
        self.b = np.asarray(np.dot(np.transpose(D), self.s))

    #Takes u, lamb, and t_type then returns a according to Rozell
    #u is the internal state variable, lamb is lambda (threshold),
    #and t_type is the soft or hard thresholding option ('S' or 'H')
    def thresh(u, lamb, t_type):
        if (u < lamb):
            return 0
        if (t_type == 'H'):
            return u
        if (t_type == 'S'):
            return u - lamb

    def return_sparse(lamb, tau, delta, u_stop, t_type):
        u = np.zeros(self.b.shape)
        self.a = self.thresh(u, lamb, t_type)
        inhibit = np.dot(np.transpose(dictionary), dictionary) - np.eye(dictionary.shape[1]))
        loop_flag = True
        
        while (loop_flag):
            u_dot = 1/tau * (self.b - u - np.dot(inhibit, self.a))
            u = u + u * delta
            self.a = self.thresh(u, lamb, t_type)
            if (u_dot < u_stop):
                loop_flag = False

        return a
            

        
        

    

    
        

