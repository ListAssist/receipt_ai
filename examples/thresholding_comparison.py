import cv2
import matplotlib.pyplot as plt

# Load image from disk
img = cv2.imread("images/rechnung.jpeg", cv2.IMREAD_GRAYSCALE)

# simple thresholding
_, simple = cv2.threshold(img, 181, 255, cv2.THRESH_BINARY)
# adaptive gauss thresholding (blockSize=11, C=2)
adaptive_gauss = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 10)
# otsu's method
_, otsu = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)

# create mappings
headlines = ["Grayscale Bild", 'Global Threshold (T=128)',
            'Adaptive Gauss Threshold (b=11, C=10)', 'Otsu Threshold']
images = [img, simple, adaptive_gauss, otsu]

# plot histogramm
plt.hist(img.ravel(), 256, [0, 256])
plt.show()

# plot images using matplotlib
for i in range(len(headlines)):
    plt.subplot(2, 2, i + 1), plt.imshow(images[i], 'gray')
    plt.title(headlines[i])
plt.show()

cv2.getGaussianKernel(11, )