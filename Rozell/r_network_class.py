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



    def generate_sparse(self, plot_udot = False, train = False):
        u = np.zeros(self.b.shape)
        self.a = u.copy()  #Initialize a by setting equal to u

        #Use trained dictionary if using to train network
        if (train):
            inhibit = np.dot(np.transpose(self.trained), self.trained)\
                            - np.eye(self.trained.shape[1])
        else:
            inhibit = np.dot(np.transpose(self.dictionary), self.dictionary)\
                            - np.eye(self.dictionary.shape[1])

        #Generate vector self.a
        udot = (1/self.tau) * (self.b - u - np.dot(inhibit, self.a))
        loop_flag = True
        u_length = len(u)
        iterations = 0
        ulen = []  #List to hold u_length over the iterations

        while (loop_flag):
            iterations += 1
            u += np.multiply(udot, self.delta)

            #Update self.a vector
            for i in range(u_length):
                self.a[i] = self.thresh(u[i])


            udot = (1/self.tau) * (self.b - u - np.dot(inhibit, self.a))
            udot_length = math.sqrt(np.dot(udot,udot))
            if plot_udot:
                ulen.append(udot_length / u_length)
            if udot_length / u_length < self.u_stop and iterations > 60 or iterations > 5100:
                loop_flag = False

        if plot_udot:
            plt.figure()
            plt.plot(range(iterations), ulen)
            plt.title('Length of udot vector')
            plt.show()



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





    #data member, then returns the residual
    def update_trained(self, alpha, clamp = True):
        stim = self.s
        recon = np.dot(self.a, np.transpose(self.trained))
        resid = stim - recon
        wdot = np.multiply(resid, ((self.a * alpha)[:, np.newaxis]))
        self.trained += np.transpose(wdot)

        #Clamp to [0,1]
        if clamp:
            self.trained = np.minimum(1., np.maximum(0., self.trained))

        return resid



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
    def reconstruct(self, lambdas, plt_udot=False):
        df = pandas.DataFrame() #DataFrame used for error table
        display = []  #List to hold rows of image data for grid (rfields scaled)
        display2 = [] #Unscaled rfields

        #For each value of lambda, set lambda and run Rozell on the given image
        for j in lambdas:
            display_row = []   #List to hold one row of image data (for display)
            display_row2 = []  #For display2
            self.set_lambda(j)
            self.generate_sparse(plot_udot=plt_udot) #Calculate sparse code
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


    #This method takes a 2 or 3-D numpy array and adds grid lines to it
    #It's intended to take dictionary data and add the lines such that the patches are visible
    #It returns the modified data
    def add_gridlines(self, data, num_rows, num_cols, line_color, line_pix):

        if len(self.im_dim) == 3:
            new1 = np.array(data[:self.im_dim[0], :, :])
            line = np.full((line_pix, data.shape[1], 3), line_color)
            for i in range(1, num_rows):
                rows = slice(i * self.im_dim[0], (i + 1) * self.im_dim[0])
                new1 = np.vstack((new1, line))
                new1 = np.vstack((new1, data[rows, :, :]))

            new2 = np.array(new1[:, :self.im_dim[1], :])
            line = np.full((new1.shape[0], line_pix, 3), line_color)
            for i in range(1, num_cols):
                cols = slice(i * self.im_dim[0], (i + 1) * self.im_dim[0])
                new2 = np.hstack((new2, line))
                new2 = np.hstack((new2, new1[:, cols, :]))

        else:
            new1 = np.array(data[:self.im_dim[0], :])
            line = np.full((line_pix, data.shape[1]), line_color)
            for i in range(1, num_rows):
                rows = slice(i * self.im_dim[0], (i + 1) * self.im_dim[0])
                new1 = np.vstack((new1, line))
                new1 = np.vstack((new1, data[rows, :]))

            new2 = np.array(new1[:, :self.im_dim[1]])
            line = np.full((new1.shape[0], line_pix), line_color)
            for i in range(1, num_cols):
                cols = slice(i * self.im_dim[0], (i + 1) * self.im_dim[0])
                new2 = np.hstack((new2, line))
                new2 = np.hstack((new2, new1[:, cols]))

        return new2


    #This method takes the number of rows and columns for the resulting image grid
    #It takes a file path to save the dictionary and a boolean (train) to
    #determine whether the original dictionary or trained dictionary data is used
    def save_dictionary(self, num_rows, num_cols, path, line_color = 0, line_pix = 1, train = False):
        ## Initialize grid
        line_color /= 255.
        if (len(self.im_dim) == 2):
            grid = np.full((self.im_dim[0] * num_rows, self.im_dim[1] * num_cols), 255.)
        else:
            grid = np.full((self.im_dim[0] * num_rows, self.im_dim[1] * num_cols, 3), 255.)

        k = 0
        for i in range(num_rows):
            rows = slice(i * self.im_dim[0], (i + 1) * self.im_dim[0])
            for j in range(num_cols):
                cols = slice(j * self.im_dim[1], (j + 1) * self.im_dim[1])
                if (train):
                    if (len(self.im_dim) == 2):
                        grid[rows, cols] = self.trained[:, k].reshape(self.im_dim)
                    else:
                        grid[rows, cols, :] = self.trained[:, k].reshape(self.im_dim)
                else:
                    if (len(self.im_dim) == 2):
                        grid[rows, cols] = self.dictionary[:, k].reshape(self.im_dim)
                    else:
                        grid[rows, cols, :] = self.dictionary[:, k].reshape(self.im_dim)
                k += 1

        dict_data = grid.copy() * 255.
        grid = self.add_gridlines(grid, num_rows, num_cols, line_color, line_pix)

        ## Save grid
        plt.figure()
        if len(self.im_dim) == 2:
            plt.imshow(grid, cmap=plt.get_cmap('gray'))
        else:
            plt.imshow(grid)
        plt.savefig(path)

        return dict_data

