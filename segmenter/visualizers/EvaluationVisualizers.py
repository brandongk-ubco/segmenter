from visualizers import Visualizer
from matplotlib import pyplot as plt

class AUCVisualizer(Visualizer):


    def visualize(self, method, tpr, fpr, auc):
        plt.plot(fpr, tpr)
        plt.title('Receiver Operating Characteristic Curve for {} (AUC = {})'.format(method, round(auc, 3)))
        import pdb
        pdb.set_trace()
        plt.ylim([0, 1])
        plt.xlim([0, max(fpr)])
        plt.xlabel('False Positive Rate (1 - Specificity)')
        plt.ylabel('True Positive Rate (Sensitivity)')
        return plt 