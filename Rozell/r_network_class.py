import numpy as np

class r_network:
    'Class to represent Rozell LCA network'

    def __init__(self, D):
        self.dictionary = D
        self.s = None
        self.b = None
        self.a = None

    #Takes the stimulus signal, sets s, then computes b according to Rozell
    def set_stimulus(self, signal):
        self.s = signal
        self.b = np.dot(np.transpose(self.dictionary), self.s)

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
        self.a = np.ones(u.shape)  #Initialize a by setting equal to u
        inhibit = np.dot(np.transpose(self.dictionary), self.dictionary) - np.eye(self.dictionary.shape[1])
        loop_flag = True

        
        #Compute initial a vector with u equal to zero vector
        print(self.a)
        for i in range(len(self.a)):
            self.a[i] = self.thresh(u[i], lamb, t_type)
        
        #Generate a vector
        print(self.b)
        while (loop_flag):
            print(u)
            print(self.a)
            u_dot = (1/tau) * (self.b - u - np.dot(inhibit, self.a))
            print(u_dot)
            u = u + (u_dot * delta)
            #Update a vector
            for i in range(len(self.a)):
                self.a[i] = self.thresh(u[i], lamb, t_type)
            if ((u_dot < u_stop).all()):
                loop_flag = False

        return self.a
            

        
        

    

    
        

