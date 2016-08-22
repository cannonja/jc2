import skvideo.io

vdata = skvideo.io.vread('Neovision2-Training-Tower-001.mpg')

print (vdata.shape)