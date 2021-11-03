import matplotlib
import numpy as np
import matplotlib.image as mpimg
from matplotlib import pyplot as plt

# displays alongside : 1) a picture of the experimental data from pearce original article
# 2) A four lines matplotlib plot. The figure is supposed to look as close
# as possible to the picture, as the four lines values were interpreted using pearce's figure
# the interpreted data is used during a grid-search to compute the Mean Square Error (MSE)
# the function allows to compare original and interpreted data
def get_pearce_experimental_data():

    fig, axs = plt.subplots(1, 2, figsize=(15,4))

    axs[0].set(title="Original visual results")
    axs[0].imshow(mpimg.imread("../images/results_pearce.jpg"))
    axs[0].set_xticks([])
    axs[0].set_yticks([])
    axs[0].set_frame_on(False)
    axs[0].plot(aspect="auto")

    hippocampal_1 = [120, 71, 67, 52, 41, 48, 23, 28, 23, 18, 17]
    control_1 = [120, 112, 78, 86, 77, 68, 60, 58, 38, 25, 11]
    hippocampal_4 = [78, 54, 63, 60, 44, 43, 40, 32, 24, 28, 12]
    control_4 = [58, 40, 29, 16, 13, 18, 17, 10, 12, 10, 7]

    experimental_data_pearce = {"hip1":hippocampal_1, "cont1":control_1, "hip4":hippocampal_4, "cont4":control_4}

    axs[1].plot(hippocampal_1, marker="o", color='black', label="HPC lesion - trial 1")
    axs[1].plot(control_1, marker="o", markerfacecolor='none', color='black', label="Control - trial 1")
    axs[1].plot(hippocampal_4, marker="o", linestyle='--', color='black', label="HPC lesion - trial 4")
    axs[1].plot(control_4, marker="o", linestyle='--', markerfacecolor='none', color='black', label="Control - trial 4")

    axs[1].set(title="Redefined explicit results")
    axs[1].set(ylabel="Escape latency (s)")
    axs[1].set(xlabel="Session")
    axs[1].spines['right'].set_visible(False)
    axs[1].spines['top'].set_visible(False)
    handles, labels = axs[1].get_legend_handles_labels()
    axs[1].legend(handles=[handles[0],handles[1],handles[2],handles[3]], labels=[labels[0],labels[1],labels[2],labels[3]])
    axs[1].plot(aspect="auto")
    axs[1].xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(1))
    plt.xticks(np.arange(11), np.arange(1, 11+1))

    plt.show()
    plt.close()

    return experimental_data_pearce

# displays alongside : 1) a picture of the experimental data from rodrigo original article
# 2) A 10 column matplotlib histogram. The histogram is supposed to look as close
# as possible to the picture, as the 10 columns values were interpreted using rodrigo's figure
# the interpreted data is used during a grid-search to compute the Mean Square Error (MSE)
# the function allows to compare original and interpreted data
def get_rodrigo_experimental_data():

    fig, axs = plt.subplots(2, 2, figsize=(15,8))

    axs[0,0].set_title("Original visual results")
    axs[0,0].imshow(mpimg.imread("../images/results_rodrigo_proximal.jpg"))
    axs[0,0].set_xticks([])
    axs[0,0].set_yticks([])
    axs[0,0].set_frame_on(False)
    axs[0,0].plot(aspect="auto")

    axs[1,0].imshow(mpimg.imread("../images/results_rodrigo_distal.jpg"))
    axs[1,0].set_xticks([])
    axs[1,0].set_yticks([])
    axs[1,0].set_frame_on(False)
    axs[1,0].plot(aspect="auto")

    proximal = [0.28, 0.22, 0.18, 0.13, 0.16]
    distal = [0.28, 0.11, 0.025, 0., 0.]

    experimental_data_rodrigo = {"dist":distal, "prox":proximal}

    axs[0,1].bar(["0°", "45°", "90°", "135°", "180°"], proximal, color='gray', edgecolor="black", )
    axs[0,1].set_title("Redefined explicit results")
    axs[0,1].set_ylabel("Proportion of steps searching in the B octant")
    axs[0,1].set_xlabel("Tests")

    axs[1,1].bar(["0°", "45°", "90°", "135°", "180°"], distal, color='gray', edgecolor="black")
    axs[1,1].set_ylabel("Proportion of steps searching in the F octant")
    axs[1,1].set_xlabel("Tests")

    plt.show()
    plt.close()

    return experimental_data_rodrigo
