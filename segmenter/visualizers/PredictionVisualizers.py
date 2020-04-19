from visualizers import Visualizer
from matplotlib import pyplot as plt

def Mask(Visualizer):

    def visualize(self):
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)

        fig.set_size_inches(10, 5)

        ax1.imshow(image, cmap='gray')
        ax1.axis('off')
        ax1.set_title('Original Image')

        ax2.imshow(mask, cmap='gray')
        ax2.axis('off')
        ax2.set_title('Original Mask')

        ax3.imshow(highlighted_image)
        ax3.axis('off')
        ax3.set_title('Predictions')

        ax4.imshow(highlighted_mask, cmap='gray')
        ax4.set_title("Predicted Mask (IOU {:.2f})".format(iou))
        ax4.axis('off')

        plt.subplots_adjust(wspace=0, hspace=0)

        plt.savefig(os.path.join(threshold_dir, "{}.png".format(name)), dpi=100, bbox_inches = 'tight', pad_inches = 0.5)
        plt.close()