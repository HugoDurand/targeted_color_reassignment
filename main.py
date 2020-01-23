import cv2
from matplotlib import pyplot as plt
import numpy as np

img_1 = cv2.imread('./images/img1.jpg',  cv2.IMREAD_UNCHANGED)

# display the images
figure_size = 15
plt.figure(figsize=(figure_size, figure_size))
plt.subplot(1,3,1),plt.imshow(cv2.cvtColor(img_1, cv2.COLOR_BGR2RGB))
plt.title('Input image product only'), plt.xticks([]), plt.yticks([])

plt.show()

# convert Imagee from BGR plane to RGB plane
img = cv2.cvtColor(img_1, cv2.COLOR_BGR2HSV)

# Reshape the image to pass as input to the K means
vectorized = img.reshape((-1,3))
vectorized = np.float32(vectorized)

# cluster ctiteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

K =2
attempts=10
ret,label,center=cv2.kmeans(vectorized,K,None,criteria,attempts,cv2.KMEANS_PP_CENTERS)
center = np.uint8(center)

# Print the clusters
print(f"The {K} cluters are {[list(x) for x in center]}")

# Look at the cluster output
res = center[label.flatten()]
result_image = res.reshape((img.shape))

# Print the cluster output
figure_size = 15
plt.figure(figsize=(figure_size,figure_size))
plt.subplot(1,2,1),plt.imshow(cv2.cvtColor(img, cv2.COLOR_HSV2RGB))
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(1,2,2),plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_HSV2RGB))
plt.title(f'Color Segmentation with {K} clusters'), plt.xticks([]), plt.yticks([])
plt.show()

# test blue top
delta = np.array([[0,0,0],[70,0,0]],dtype='uint8' )
delta_sub = np.array([[0,0,0],[0,0,0]],dtype='uint8' )

# build altering filters
label.flatten()
filter_mask = np.where(label.flatten()== 1, 1, 0)
add_mask = delta[filter_mask]
sub_mask = delta_sub[filter_mask]
mask_image_sub = sub_mask.reshape((img.shape))
mask_image = add_mask.reshape((img.shape))

# alter the image
dst = cv2.subtract(img,mask_image_sub)
dst_1 = cv2.add(dst,mask_image)

# HSV to RGB to dsiplay in mathplotlib
img_original = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
img_altered = cv2.cvtColor(dst_1, cv2.COLOR_HSV2RGB)

# display the image
figure_size = 15
plt.figure(figsize=(figure_size,figure_size))
plt.subplot(1,2,1),plt.imshow(img_original)
plt.title('Original Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(1,2,2),plt.imshow(img_altered)
plt.title('Output Image'), plt.xticks([]), plt.yticks([])
plt.show()
