import skvideo.io

video = 'Neovision2-Training-Tower-001.mpg'

vdata = skvideo.io.vread(video)

print (type(vdata[0]), vdata[0].shape)
