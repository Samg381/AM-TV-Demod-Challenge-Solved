from scipy.io import wavfile
import numpy as np
from PIL import Image
import sys

# Allow printing numpy arras w/o truncation
np.set_printoptions(threshold=sys.maxsize)

fs, data = wavfile.read("AM_IQ_Image_Fs48khz.wav")
print("Shape:", data.shape, "Dtype:", data.dtype)

# assuming .wav contains two channels: real, imaginary (I, Q)
I = data[:,0]
Q = data[:,1]

#debug
print(f'Max I: {max(I)} | Max Q: {max(Q)}')

# Well, turns out only I is populated with data. so this is a 'real' signal.
# in any case, converting to complex will still work (just know there is no phase)

# Float to complex IQ
complex_samples = I + 1j*Q

# create our long array of samples (note this is one continuous stream)
envelope = abs(complex_samples)

# Debug
# print(envelope)
# print(len(envelope))

# we were told img is 800x800px (640k px)
# think of the envelope as a stream of these pixels
# so, select the first 640k pixels in the stream, and reshape to 800x800
img = envelope[:800*800].reshape(800,800)

# creating the 'image'
img -= img.min() # normalize all samples to 0 (make sure lowest pixel value is 0.0)
img /= img.max() # divide all values by maximum sample (makes sure hightst value is 1.0)

img = (img*255).astype(np.uint8) # multiply 0.0 - 1.0 floats by 255 to gain pixel values

# Create an image from the array
Image.fromarray(img, 'L').save("decoded_image.png")
