import cv2
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
# plot config
from matplotlib.lines import Line2D
from scipy.interpolate import interp1d
from matplotlib.colors import LinearSegmentedColormap as cmap

fig, ax1 = plt.subplots()
ax1.set_title("Histogram und Varianz Animation")
ax1.set_xlabel("Pixel Werte")
ax1.set_ylabel("Anzahl der Pixel")
ax1.set_axisbelow(True)
ax1.set_xlim(0, 255)

custom_lines = [Line2D([0], [0], color="blue", lw=8),
                Line2D([0], [0], color="red", lw=8),
                Line2D([0], [0], color="black", lw=8),
                Line2D([0], [0], color="green", lw=8)]
plt.legend(custom_lines, ["Histogramm", "Inter-Class Varianz", "Threshindex", "Ermittelter Threshold"], loc="upper center", bbox_to_anchor=(0.5, -0.15), borderpad=1, edgecolor="black", fancybox=False, ncol=2)
plt.grid("on")

ax2 = ax1.twinx()
ax2.set_ylabel("Inter-Class Varianz")
ax2.tick_params(axis="y", labelcolor="tab:red")
ax2.set_ylim(0, 1400)
fig.tight_layout()
lineaxis = ax1.twinx()
lineaxis.get_yaxis().set_visible(False)

# Load image from disk
img = cv2.imread("images/rechnung.jpeg", cv2.IMREAD_GRAYSCALE)
histogram, _ = np.histogram(img, bins=256)

ax1.hist(img.ravel(), 256, [0, 256], label="Histogramm")

# Divide by sum of all numbers so sum = 1 (probabilities)
histogram = histogram / histogram.sum()
thresh = 1

inter_class_variance = -1
# formula to be minimized (between class)
# q1 * (1 - q1) * (u1 - u2)^2
# q = Sum of histogram probability till current threshold
# u = SUM(currentThreshold * histogram probability
x_variances = []
y_variances = []

plt.pause(20)

for t in range(1, 255):
    print(f"Current best {thresh} loop {t}")

    q1 = histogram[:t].sum()
    q2 = histogram[t:].sum()

    u1 = np.arange(0, t) * histogram[:t]
    u1 = u1.sum() / q1

    u2 = np.arange(t, 256) * histogram[t:]
    u2 = u2.sum() / q2

    variance = q1 * (1 - q1) * (u1 - u2)**2

    x_variances.append(t)
    y_variances.append(variance)
    if t > 3:
        f = interp1d(x_variances, y_variances, kind="cubic")
        ax2.plot(x_variances, y_variances, color="tab:red", label="kong")

    lineaxis.cla()
    lineaxis.axvline(t, color="black", label="Current Loop Index")

    plt.pause(0.005)
    if variance > inter_class_variance:
        inter_class_variance = variance
        thresh = t

# plotting
lineaxis.cla()
lineaxis.axvline(thresh, color="green")

plt.pause(10)


print(f"Threshold value: {thresh}")
_, thresholded = cv2.threshold(img, thresh, 255, cv2.THRESH_BINARY)

plt.clf()
plt.imshow(thresholded, cmap="gray")
plt.show()
