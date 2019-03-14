import numpy as np
import matplotlib.pyplot as plt
import cv2

##Input any image !

img = plt.imread("kMeansImg6.jpg", 'jpg')

## Uncomment to print the type and shape of input image.
##print(type(img))
##print(img.shape)

originalShape = img.shape
plt.imshow(img)
plt.show()


testImg = img.reshape(-1, 3)

testImg.shape


##KMeansAlgo (k= integer)
## "k" equals the number of colors in which image wll be segmented !

k = int(input())


def distance(x1, x2):
    return np.sqrt(sum(np.square(x1-x2)))


np.random.seed(3)   ## Fixing the random generation of variables.

initialPoints = 255*np.random.random((k, 3))
initialPoints = list(initialPoints)


##Uncomment to print the randomly generated points.
##print(initialPoints)

pts = {}
for ix in range(k):
    pts[ix] = []
print(pts)

colors = {
    'points': pts,
    'initialPoints': initialPoints,
    'centres': []
}

# Kmeans logic in which finding the centre_mean-most 'k' points.


def kmeans(testImg, colors, pts):
    colors['centres'] = []
    for ix in range(len(testImg)):
        l = []
        for jx in range(len(colors['initialPoints'])):
            l.append(distance(testImg[ix], colors['initialPoints'][jx]))
        colors['points'][np.argmin(l)].append(testImg[ix])
    
    for kx in range(k):
        colors['centres'].append(np.mean(colors['points'][kx],axis =0))
    colors['initialPoints'] = colors['centres']
    colors['points'] = pts
    return colors['centres']


km = []
for jx in range(10):
    km.append(kmeans(testImg, colors,pts))


## Converting values to feasible pixel values ! 
for ix in range(k):
    for jx in range(3):
        colors['centres'][ix][jx] = int(colors['centres'][ix][jx])


centresArray = np.array(colors['centres'])

imagePixeled = []
for ix in range(len(testImg)):
    d = []
    for jx in range(k):
        d.append(distance(testImg[ix], centresArray[jx]))
    imagePixeled.append(centresArray[np.argmin(d)])


imageImshow = np.array(imagePixeled)/255 # dividing by 255 for imshow() !

imageImshow = imageImshow.reshape(originalShape) # Reshaping for final output

#### Finding the DOMINANT COLORS !

i = 1

plt.figure(0,figsize=(4,2))
color = []

for ix in colors['centres']:
    plt.subplot(1,k,i)
    i+=1
    color.append(ix)
    
    a = np.zeros((100,100,3), dtype = 'uint8')
    a[:,:,:]= ix
    plt.imshow(a)
plt.show()


### Printing the color Segmented image !


plt.imshow(imageImshow)
plt.show()
