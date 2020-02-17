import cv2
import matplotlib.pyplot as plt
import numpy as np

# Load image from disk
img = cv2.imread("images/rechnung.jpeg", cv2.IMREAD_GRAYSCALE)
histogram, _ = np.histogram(img, bins=256)

# Divide by sum of all numbers so sum = 1 (probabilities)
histogram = histogram / histogram.sum()
thresh = 1

inter_class_variance = -1
# formula to be minimized (between class)
# q1 * (1 - q1) * (u1 - u2)^2
# q = Sum of histogram probability till current threshold
# u = SUM(currentThreshold * histogram probability
for t in range(1, 255):
    print(f"Current best {thresh} loop {t}")

    q1 = histogram[:t].sum()
    q2 = histogram[t:].sum()

    u1 = np.arange(0, t) * histogram[:t]
    u1 = u1.sum() / q1

    u2 = np.arange(t, 256) * histogram[t:]
    u2 = u2.sum() / q2

    variance = q1 * (1 - q1) * (u1 - u2)**2
    if variance > inter_class_variance:
        inter_class_variance = variance
        thresh = t

print(f"Threshold value: {thresh}")
otsu_output = (img > thresh) * 255

plt.imshow(otsu_output, cmap="gray")
plt.show()
