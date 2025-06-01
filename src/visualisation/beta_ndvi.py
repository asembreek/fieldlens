import numpy as np
import pandas
from scipy.stats import beta
import ee
import matplotlib.pyplot as plt

class Beta:
    def __init__(self, data, region):
        hist = data.reduceRegion(
            ee.Reducer.fixedHistogram(-1, 1, 500), region, scale=10
        ).getInfo()
        band_name = list(hist.keys())[0]
        hist_data = hist[band_name]  # [bin_center, count]

        self.a = np.array(hist_data)
        # a[:, 0] -> bin center NDVI from -1 to 1
        # a[:, 1] -> count of pixels in the bin

        self.x = self.a[:, 0]  #
        self.y = self.a[:, 1] / np.sum(self.a[:, 1])
        # raw count of pixels falling in each bin / total num pixels -> 0 <= y <= 1

        self.counts = self.a[:, 1].astype(int)
        self.u = (self.x + 1) / 2
        self.samples = np.repeat(self.u, self.counts)

        self.params = None
        self.u_fit = np.linspace(0, 1, 500)
        self.x_fit = 2 * self.u_fit - 1
        self.pdf_fit = None


    def draw_raw(self):
        plt.grid()
        plt.plot(self.x, self.y, ".")
        plt.show()


    def fit_beta(self):
        self.params = beta.fit(self.samples, floc = 0, fscale = 1)
        self.pdf_fit = beta.pdf(self.u_fit, *self.params)
        print(f"Fitted Beta parameters: α={self.params[0]:.3f}, β={self.params[1]:.3f}")


    def draw_fit(self):
        bin_width = self.x[1] - self.x[0]
        pdf_x = self.pdf_fit / 2
        plt.plot(self.x_fit, pdf_x * bin_width)
        plt.grid()
        plt.show()

    def draw_raw_fit(self):
        if self.pdf_fit is None:
            self.fit_beta()
        bin_width = self.x[1] - self.x[0]
        pdf_x = self.pdf_fit / 2
        plt.plot(self.x, self.y, ".", label="Raw Data")
        plt.plot(self.x_fit, pdf_x * bin_width, linestyle="--", label="Fitted Beta")
        plt.xlabel("NDVI")
        plt.ylabel("Frequency / PDF")
        plt.title("NDVI Histogram and Fitted Beta Distribution")
        plt.legend()
        plt.grid()
        plt.show()
