##############################################################################

#  SHADOW REMOVAL WITH SUBREGION MATCHING AND ILLUMINATION TRANSFER

#  DONE BY:
#  BHARATH KUMAR 
#  CHOCKALINGAM
#  DC VIVEK 
#  HARINATH GOBI

##############################################################################

import cv2
import numpy as np
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import sys
from skimage import segmentation
import torch.nn.init
import scipy

use_cuda = torch.cuda.is_available()

############################# SEGMENTATION VARIABLES #########################

nChannel = 100
maxIter = 100
minLabels = 3
lr = 0.1
nConv = 2
num_superpixels = 10000
compactness = 100
visualize = 1
input_file = 'shadow.jpg'

##############################################################################

img = cv2.imread(input_file)
shadow_removed_img = cv2.imread(input_file)
gray = cv2.imread(input_file, 0)
blur = cv2.bilateralFilter(img,9,75,75)

#############################    HSI CONVERSION    ###########################

blur = np.divide(blur, 255.0)

hsi = np.zeros((blur.shape[0],blur.shape[1],blur.shape[2]),dtype=np.float)
ratio_map = np.zeros((blur.shape[0],blur.shape[1]),dtype=np.uint8)

for i in range(blur.shape[0]):
    for j in range(blur.shape[1]):
        hsi[i][j][2] = (blur[i][j][0]+blur[i][j][1]+blur[i][j][2])/3
        hsi[i][j][0] = math.acos(((blur[i][j][2]-blur[i][j][1])*(blur[i][j][2]-blur[i][j][0]))/(2*math.sqrt((blur[i][j][2]-blur[i][j][1])*(blur[i][j][2]-blur[i][j][1])+(blur[i][j][2]-blur[i][j][0])*(blur[i][j][1]-blur[i][j][0]))))
        hsi[i][j][1] = 1 - 3*min(blur[i][j][0],blur[i][j][1],blur[i][j][2])/hsi[i][j][2]
        ratio_map[i][j] = hsi[i][j][0]/(hsi[i][j][2]+0.01)                    

###############################################################################
 
#########################    SHADOW DETECTION   ###############################

hist = np.histogram(ratio_map.ravel(),256,[0,256])
ret,th = cv2.threshold(ratio_map,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
ret,inv_th = cv2.threshold(ratio_map,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
bin_thresh = cv2.medianBlur(th,15)
bin_inv_thresh = cv2.medianBlur(inv_th,15)

###############################################################################

shadow_region = cv2.bitwise_and(img,img,mask = th)
background_region = cv2.bitwise_and(img,img,mask = inv_th)

##############################  SEGMENTATION  #################################

# CNN model
class MyNet(nn.Module):
    def __init__(self,input_dim):
        super(MyNet, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, nChannel, kernel_size=3, stride=1, padding=1 )
        self.bn1 = nn.BatchNorm2d(nChannel)
        self.conv2 = []
        self.bn2 = []
        for i in range(nConv-1):
            self.conv2.append( nn.Conv2d(nChannel, nChannel, kernel_size=3, stride=1, padding=1 ) )
            self.bn2.append( nn.BatchNorm2d(nChannel) )
        self.conv3 = nn.Conv2d(nChannel, nChannel, kernel_size=1, stride=1, padding=0 )
        self.bn3 = nn.BatchNorm2d(nChannel)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu( x )
        x = self.bn1(x)
        for i in range(nConv-1):
            x = self.conv2[i](x)
            x = F.relu( x )
            x = self.bn2[i](x)
        x = self.conv3(x)
        x = self.bn3(x)
        return x

def segment(im):
    data = torch.from_numpy( np.array([im.transpose( (2, 0, 1) ).astype('float32')/255.]) )
    if use_cuda:
        data = data.cuda()
    data = Variable(data)

    # slic
    labels = segmentation.slic(im, compactness=compactness, n_segments=num_superpixels)
    labels = labels.reshape(im.shape[0]*im.shape[1])
    u_labels = np.unique(labels)
    l_inds = []
    for i in range(len(u_labels)):
        l_inds.append( np.where( labels == u_labels[ i ] )[ 0 ] )

    # train
    model = MyNet( data.size(1) )
    if use_cuda:
        model.cuda()
        for i in range(nConv-1):
            model.conv2[i].cuda()
            model.bn2[i].cuda()
    model.train()
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    label_colours = np.random.randint(255,size=(100,3))
    for batch_idx in range(maxIter):
        # forwarding
        optimizer.zero_grad()
        output = model( data )[ 0 ]
        output = output.permute( 1, 2, 0 ).contiguous().view( -1, nChannel )
        ignore, target = torch.max( output, 1 )
        im_target = target.data.cpu().numpy()
        nLabels = len(np.unique(im_target))
        if visualize:
            im_target_rgb = np.array([label_colours[ c % 100 ] for c in im_target])
            im_target_rgb = im_target_rgb.reshape( im.shape ).astype( np.uint8 )
            cv2.imshow( "output", im_target_rgb )
            cv2.waitKey(1)

        # superpixel refinement
        for i in range(len(l_inds)):
            labels_per_sp = im_target[ l_inds[ i ] ]
            u_labels_per_sp = np.unique( labels_per_sp )
            hist = np.zeros( len(u_labels_per_sp) )
            for j in range(len(hist)):
                hist[ j ] = len( np.where( labels_per_sp == u_labels_per_sp[ j ] )[ 0 ] )
            im_target[ l_inds[ i ] ] = u_labels_per_sp[ np.argmax( hist ) ]
        target = torch.from_numpy( im_target )
        if use_cuda:
            target = target.cuda()
        target = Variable( target )
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()

        print (batch_idx, '/', maxIter, ':', nLabels, loss.data[0])
        if nLabels <= minLabels:
            print ("nLabels", nLabels, "reached minLabels", minLabels, ".")
            break  
    return im_target_rgb        

shadow_region = segment(shadow_region)
shadow_region = cv2.bitwise_and(shadow_region,shadow_region,mask = th)
background_region = segment(background_region)
background_region = cv2.bitwise_and(background_region,background_region,mask = inv_th)

###############################################################################

############################  SUBREGION MATCHING  #############################

shadow_region_colors = list(np.unique(shadow_region.reshape(-1, shadow_region.shape[2]), axis=0))
background_region_colors = list(np.unique(background_region.reshape(-1, background_region.shape[2]), axis=0))

shadow_subregions = []
background_subregions = []

for i in range(len(shadow_region_colors)):
    shadow_subregions.append([])
    for x in range(shadow_region.shape[0]):
        for y in range(shadow_region.shape[1]):
            if (shadow_region[x][y] == shadow_region_colors[i]).all():
                shadow_subregions[i].append([x,y])           

for i in range(len(background_region_colors)):
    background_subregions.append([])
    for x in range(background_region.shape[0]):
        for y in range(background_region.shape[1]):
            if (background_region[x][y] == background_region_colors[i]).all():
                background_subregions[i].append([x,y])           

shadow_region_colors.pop(0)
background_region_colors.pop(0)
shadow_region_colors = np.array(shadow_region_colors)
background_region_colors = np.array(background_region_colors)

shadow_subregions.pop(0)
background_subregions.pop(0)

#print(background_region_colors)
#print(shadow_region_colors)
#print(len(shadow_subregions))
#print(len(shadow_subregions[0]),len(shadow_subregions[1]),len(shadow_subregions[2]))

feature_vector = []
kernel_vector = []
scales = 4
orientation = 6

for s in range(scales):
    for o in range(orientation):
        g_kernel = cv2.getGaborKernel((21, 21), (4.0+s*2), (np.pi*o)/6, 10.0, 0.5, 0, ktype=cv2.CV_32F)
        filtered_img = cv2.filter2D(gray, cv2.CV_8UC3, g_kernel)
        kernel_vector.append(g_kernel)
        feature_vector.append(filtered_img)

shadow_subregion_feature_vector = []
background_subregion_feature_vector = []

shadow_region_coord_mean = []
background_region_coord_mean = []

feature_vector_mean = [np.mean(vector) for vector in feature_vector]
feature_vector_std = [np.std(vector) for vector in feature_vector]

iterator = 0

for vector in feature_vector:
    shadow_region_coord_mean.append([])
    shadow_subregion_feature_vector.append([])
    background_region_coord_mean.append([])
    background_subregion_feature_vector.append([])

    for region in shadow_subregions:
        mean = 0
        std = 0
        x_mean = 0
        y_mean = 0
        for x,y in region:
            mean = mean + vector[x][y]
            x_mean = x_mean + x
            y_mean = y_mean + y
        mean = mean/(vector.shape[0]*vector.shape[1])    
        for x,y in region:
            std = std + abs((vector[x][y]-mean)*(vector[x][y]-mean))   
        shadow_subregion_feature_vector[iterator].append([mean,math.sqrt(std/(vector.shape[0]*vector.shape[1]))]) 
        shadow_region_coord_mean[iterator].append([x_mean/(len(region)),y_mean/(len(region))])
        
    for region in background_subregions:
        mean = 0
        std = 0
        x_mean = 0
        y_mean = 0
        for x,y in region:
            mean = mean + vector[x][y]
            x_mean = x_mean + x
            y_mean = y_mean + y
        mean = mean/(vector.shape[0]*vector.shape[1])    
        for x,y in region:
            std = std + abs((vector[x][y]-mean)*(vector[x][y]-mean))    
        background_subregion_feature_vector[iterator].append([mean,math.sqrt(std/(vector.shape[0]*vector.shape[1]))]) 
        background_region_coord_mean[iterator].append([x_mean/(len(region)),y_mean/(len(region))])
    iterator = iterator + 1

#print(len(background_subregions),len(shadow_subregions),len(shadow_subregion_feature_vector),len(shadow_region_coord_mean))        
#print(len(background_region_coord_mean),len(background_subregion_feature_vector),len(background_region_coord_mean[0]),len(background_subregion_feature_vector[0]),len(background_subregion_feature_vector[0][0]),len(background_region_coord_mean[0][0]))
#print(shadow_region_coord_mean[0],shadow_region_coord_mean[1])
#print(shadow_subregion_feature_vector[0],shadow_subregion_feature_vector[1],shadow_subregion_feature_vector[2])

dist_texture = []
dist_space = []
index = 0
dist = 0
text = 0
iterator = 0
_iterator = 0

#for s in range(scales):
#    for o in range(orientation):
#        dist_texture.append([])
#        index = (orientation*s)+o 
#        for shadow_reg in range(len(shadow_subregion_feature_vector[index])):
#            dist_texture[index].append([])
#            for background_reg in range(len(background_subregion_feature_vector[index])):
#                text = text + abs((shadow_subregion_feature_vector[index][shadow_reg][0]-background_subregion_feature_vector[index][shadow_reg][0])/feature_vector_mean[index][0]) + abs((shadow_subregion_feature_vector[index][shadow_reg][1]-background_subregion_feature_vector[index][shadow_reg][1])/feature_vector_mean[index][1])        
#                dist_texture[index][iterator].append(text)    
#            iterator = iterator + 1

#print(feature_vector_mean[0],shadow_subregion_feature_vector[0],background_subregion_feature_vector[0])

for shadow_reg in range(len(shadow_subregion_feature_vector[index])):
    dist_texture.append({})
    _iterator = 0
    for background_reg in range(len(background_subregion_feature_vector[index])):
        text = 0
        for s in range(scales):
            for o in range(orientation):
                index = (orientation*s)+o
                text = text + abs((shadow_subregion_feature_vector[index][shadow_reg][0]-background_subregion_feature_vector[index][shadow_reg][0])/feature_vector_mean[index]) + abs((shadow_subregion_feature_vector[index][shadow_reg][1]-background_subregion_feature_vector[index][shadow_reg][1])/feature_vector_mean[index])        
        dist_texture[iterator][_iterator] = text
        _iterator = _iterator + 1    
    iterator = iterator + 1

iterator = 0
_iterator = 0   

for shadow_reg in range(len(shadow_subregion_feature_vector[index])):
    dist_space.append({})
    _iterator = 0
    for background_reg in range(len(background_subregion_feature_vector[index])):
        dist = math.sqrt((shadow_region_coord_mean[0][shadow_reg][0]-background_region_coord_mean[0][background_reg][0])*(shadow_region_coord_mean[0][shadow_reg][0]-background_region_coord_mean[0][background_reg][0])+(shadow_region_coord_mean[0][shadow_reg][1]-background_region_coord_mean[0][background_reg][1])*(shadow_region_coord_mean[0][shadow_reg][1]-background_region_coord_mean[0][background_reg][1]))        
        dist_space[iterator][_iterator] = dist
        _iterator = _iterator + 1
    iterator = iterator + 1

#print(dist_texture,dist_space)

for i in range(len(dist_texture)):
    dist_texture[i] = sorted(dist_texture[i].items(), key=lambda kv: kv[1])
    dist_space[i] = sorted(dist_space[i].items(), key=lambda kv: kv[1])

match = []

for i in range(len(dist_texture)):
    min = len(dist_texture[0])+len(dist_space[0])
    minimum = 0
    for j in range(len(dist_texture[0])):
        for k in range(len(dist_space[0])):
            if dist_texture[i][j][0] == dist_space[i][k][0]:
                if j+k < min:
                    min = j+k
                    minimum = dist_texture[i][j][0]
    match.append(minimum)                                                          

#print(match)
###############################################################################

############################    SHADOW REMOVAL    #############################

luminance = np.zeros((img.shape[0],img.shape[1]))
#shadow_removed_image = np.zeros((img.shape[0],img.shape[1],img.shape[2]))
#sigma_shadow = [1]*len(shadow_subregion_feature_vector[0])
#sigma_background = [1]*len(background_subregion_feature_vector[0])

for x in range(img.shape[0]):
    for y in range(img.shape[1]):
        luminance[x][y] = img[x][y][2]*0.2126 + img[x][y][1]*0.7152 + img[x][y][0]*0.0722
 
iterator_ = 0

shad_reg_mean = []
back_reg_mean = []

for region in shadow_subregions:
    r = 0
    g = 0
    b = 0
    for x,y in region:
        b = b + img[x][y][0]
        g = g + img[x][y][1]
        r = r + img[x][y][2]
    b = b/len(region)
    g = g/len(region)
    r = r/len(region)
    shad_reg_mean.append([b,g,r])
        
for region in background_subregions:
    r = 0
    g = 0
    b = 0
    for x,y in region:
        b = b + img[x][y][0]
        g = g + img[x][y][1]
        r = r + img[x][y][2]
    b = b/len(region)
    g = g/len(region)
    r = r/len(region)
    back_reg_mean.append([b,g,r])

luminance_shad_std = []
luminance_back_std = []

for region in shadow_subregions:
    std = 0
    mean = 0
    for x,y in region:
        mean = mean + luminance[x][y]
    mean = mean/len(region)
    for x,y in region:
        std = std + abs((luminance[x][y]-mean)*(luminance[x][y]-mean))
    luminance_shad_std.append(math.sqrt(std/len(region)))


for region in background_subregions:
    std = 0
    mean = 0
    for x,y in region:
        mean = mean + luminance[x][y]
    mean = mean/len(region)
    for x,y in region:
        std = std + abs((luminance[x][y]-mean)*(luminance[x][y]-mean))
    luminance_back_std.append(math.sqrt(std/len(region)))

#match = [0,1,2]
_iterator_ = 0
for region in shadow_subregions:
    for x,y in region:
        shadow_removed_img[x][y][0] = back_reg_mean[match[_iterator_]][0] + (luminance_back_std[match[_iterator_]]/luminance_shad_std[_iterator_])*(img[x][y][0]-shad_reg_mean[_iterator_][0])
        shadow_removed_img[x][y][1] = back_reg_mean[match[_iterator_]][1] + (luminance_back_std[match[_iterator_]]/luminance_shad_std[_iterator_])*(img[x][y][1]-shad_reg_mean[_iterator_][1])
        shadow_removed_img[x][y][2] = back_reg_mean[match[_iterator_]][2] + (luminance_back_std[match[_iterator_]]/luminance_shad_std[_iterator_])*(img[x][y][2]-shad_reg_mean[_iterator_][2])
    _iterator_ = _iterator_ + 1    

#print(luminance_back_std,luminance_shad_std,back_reg_mean,shad_reg_mean)
print(match)

###############################################################################

cv2.imshow("original_image",img)
cv2.imshow("detected_shadow",bin_thresh)
cv2.imshow("shadow_region",shadow_region)
cv2.imshow("background_region",background_region)
cv2.imshow("shadow_removed_image",shadow_removed_img)

cv2.waitKey(0)
cv2.destroyAllWindows(0)

