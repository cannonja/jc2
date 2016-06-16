import numpy as np
import math
import pandas
from PIL import Image
import matplotlib.pyplot as plt
import pdb

class r_network:
    'Class to represent Rozell LCA network'

    def __init__(self, D):
        self.dictionary = D.astype(float)
        self.trained = self.dictionary.copy()
        self.im_dim = None
        self.s = None
        self.b = None
        self.a = None
        self.lamb = None
        self.tau = None
        self.delta = None
        self.u_stop = None
        self.t_type = None


    #Takes the stimulus signal, sets s, then computes b according to Rozell
    def set_stimulus(self, signal, train = False):
        self.s = np.asarray(signal, dtype=float)

        if (train):
            self.b = np.dot(np.transpose(self.trained), self.s)
        else:
            self.b = np.dot(np.transpose(self.dictionary), self.s)

    #Takes a tuple and sets the dimensions of the images in question
    def set_dim(self, dims):
        self.im_dim = dims


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

    def generate_sparse(self, train = False):
        u = np.zeros(self.b.shape)
        self.a = u.copy()  #Initialize a by setting equal to u

        #Use trained dictionary if using to train network
        if (train):
            inhibit = np.dot(np.transpose(self.trained), self.trained)\
                            - np.eye(self.trained.shape[1])
        else:
            inhibit = np.dot(np.transpose(self.dictionary), self.dictionary)\
                            - np.eye(self.dictionary.shape[1])

        udot = (1/self.tau) * (self.b - u - np.dot(inhibit, self.a))
        loop_flag = True

        #Generate vector self.a
        len_u = len(u)
        iterations = 0
        ulen = []
        while (loop_flag):
            iterations += 1
            u = u + (udot * self.delta)
            #Update a vector
            for i in range(len(self.a)):
                self.a[i] = self.thresh(u[i])
            udot = (1/self.tau) * (self.b - u - np.dot(inhibit, self.a))
            udot_length = math.sqrt(np.dot(udot,udot))
            #print (udot_length / len_u)
            ulen.append(udot_length / len_u)
            if udot_length / len_u < self.u_stop and iterations > 60: #or iterations > 5100:
                loop_flag = False
                print (iterations)

        '''
        plt.figure()
        plt.plot(range(iterations), ulen)
        plt.show()
        '''


        return (self.a)   #, iterations, ulen)


    #This method updates the copy of the dictionary stored in the "trained"
    #data member, then returns the residual
    def update_trained(self, alpha):
        stim = self.s
        recon = np.dot(self.a, np.transpose(self.trained))
        resid = stim - recon

        wdot = resid * ((self.a * alpha)[:, np.newaxis])
        wdot = np.asmatrix(resid).T * (np.asmatrix(self.a) * alpha)
        #print("resid {}, a {}".format(resid.__class__, self.a.__class__))
        #self.trained = (self.trained + np.transpose(wdot)).copy()
        self.trained += wdot
        #Clamp to [0,1]
        self.trained = np.minimum(1., np.maximum(0, self.trained))
        return resid



    #Returns 1-D numpy array containing E(t), the left-hand operand of E(t),
    #and the right-hand operand of E(t).
    #That is the operands of the right-hand side of the error equation
    def return_error(self):
        stim = self.s
        recon = np.dot(self.dictionary, self.a)
        resid = stim - recon
        a = 0.5 * math.sqrt(np.dot(resid, resid))
        b = None
        #sparsity = len(self.a[self.a > 0]) / len(self.a)
        sparsity = np.count_nonzero(self.a)
        norm1 = sum(abs(self.a))
        cost = 0

        if (self.t_type == 'S'):
            cost = norm1
        elif (norm1 > self.lamb):
            cost = self.lamb / 2.

        b = self.lamb * cost
        #error = np.array([[self.lamb, (a + b), a, b, sparsity]])
        error = np.array([[(a + b), a, b, sparsity]])

        return error

    #This method returns two arrays:
    #coefficients consists of the active coefficients
    #rfields consists of the respective dictionary elements
    def get_rfields(self):
        indices = np.where(self.a > 0)
        rfields = self.dictionary[:, indices]
        coefficients = self.a[indices]

        return (coefficients, rfields)



    #This method takes a single lambda (as a list) or an array
    #of lambdas, then returns an error table (as a pandas dataframe)
    #and the data for two images (as lists).

    #In the error table, each row represents different error measures at
    #the given lambda.  The images are intended to be displayed as grids
    #where each row corresponds to the lambdas in the error table.

    #The first image in each row is the reconstruction and the remaining images
    #in the row are the chosen receptive fields that make up the reconstruction.
    #One image has the coefficients applied to the receptive fields and the other does not.

    #The data for each image grid is returned as a list, where each element in the list
    #is a list of 28x28 numpy arrays representing the reconstruction and receptive fields
    #for the respective lambda (rows of the image grid without the stimulus).
    #The stimulus will be added when the final grid image is built.

    #It's important to understand that this method uses the values set by the current
    #network previously!!!!
    def reconstruct(self, lambdas):
        df = pandas.DataFrame() #DataFrame used for error table
        display = []  #List to hold rows of image data for grid (rfields scaled)
        display2 = [] #Unscaled rfields

        #For each value of lambda, set lambda and run Rozell on the given image
        for j in lambdas:
            display_row = []   #List to hold one row of image data (for display)
            display_row2 = []  #For display2
            self.set_lambda(j)
            catch = self.generate_sparse() #Calculate sparse code
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

        return (df, display, display2)




    ##This method takes a number representing the number of image rows and
    ##the data to fill the grid with.
    ##It returns a 2-D numpy array of image data for a single image (grid)
    def fill_grid(self, num_rows, grid_data):
        ##Get max number of components for display grid dimensions
        biggest = 0
        for j in grid_data:
            if (len(j) > biggest):
                biggest = len(j)
        ##Allocate pixels for display grid
        grid = np.full((28 * num_rows, 28 * (biggest + 1)), 0)

        ##Fill display grid with image data
        ##Iterate over rows -> for each row, add columns
        for j in range(num_rows):
            rows = slice(j*28, (j+1)*28)
            #Original image
            grid[rows, :28] = self.s.reshape((28,28))
            #Reconstruction and components
            for k in range(len(grid_data[j])):
                cols = slice((k+1)*28, (k+2)*28)
                grid[rows, cols] = grid_data[j][k]

        return grid


    #This method takes the number of rows and columns for the resulting image grid
    #It takes a file path to save the dictionary and a boolean (train) to
    #determine whether the original dictionary or trained dictionary data is used
    def save_dictionary(self, num_rows, num_cols, path, train = False):
        ## Initialize grid
        line_pix = 2        #Pixels allocated for grid line thinkness
        line_color = 0      #Color of grid lines
        r_data = (self.im_dim[0] + line_pix) * num_rows
        c_data = (self.im_dim[1] + line_pix) * num_cols
        if (len(self.im_dim) == 2):
            grid = np.full((r_data, c_data), 255.)
        else:
            grid = np.full((r_data, c_data, 3), 255.)

        k = 0
        for i in range(num_rows):
            rows = slice(i * self.im_dim[0], (i + 1) * self.im_dim[0] + line_pix)
            for j in range(num_cols):
                cols = slice(j * self.im_dim[1], (j + 1) * self.im_dim[1] + line_pix)
                if (train):
                    if (len(self.im_dim) == 2):
                        im_data = self.trained[:, k].reshape(self.im_dim)
                        grid_data = np.hstack(im_data, np.full((r_data, line_pix)), line_color)
                        grid[rows, cols] = np.vstack(grid_data, np.full((line_pix, c_data), line_color))
                    else:
                        im_data = self.trained[:, k].reshape(self.im_dim)
                        grid_data = np.hstack(im_data, np.full((r_data, line_pix, 3)), line_color)
                        grid[rows, cols, :] = np.vstack(grid_data, np.full((line_pix, c_data, 3), line_color))
                else:
                     if (len(self.im_dim) == 2):
                        im_data = self.dictionary[:, k].reshape(self.im_dim)
                        grid_data = np.hstack(im_data, np.full((r_data, line_pix)), line_color)
                        grid[rows, cols] = np.vstack(grid_data, np.full((line_pix, c_data), line_color))
                    else:
                        im_data = self.dicitonary[:, k].reshape(self.im_dim)
                        grid_data = np.hstack(im_data, np.full((r_data, line_pix, 3)), line_color)
                        grid[rows, cols, :] = np.vstack(grid_data, np.full((line_pix, c_data, 3), line_color))
               k += 1

        grid *= 255.
        if (len(self.im_dim) == 2):
            im_grid = Image.fromarray(grid, mode='L')
        else:
            im_grid = Image.fromarray(grid, mode='RGB')
        #im_grid = im_grid.convert('L')
        im_grid.save(path, 'PNG')

