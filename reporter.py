import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

class Reporter:

    def __init__(self):
        pass

    def _define_attr(self, metrics):
        (self.reporter_config, self.dataset_config, self.model_config, self.predictions, self.true_labels,
        self.confusion_matrix, self.precision, self.recall, self.crossentropy, self.fps) = metrics

        self.confusion_matrix_plot_size = self.reporter_config["confusion_matrix_plot_size"]
        if self.confusion_matrix_plot_size:
            self.confusion_matrix_plot_size = tuple(self.confusion_matrix_plot_size)

        self.confidence_plot_size = self.reporter_config["confidence_plot_size"]
        if self.confidence_plot_size:
            self.confidence_plot_size = tuple(self.confidence_plot_size)

        self.dataset_label = self.dataset_config["label_mapping"]
        self.model_path = self.model_config["model_path"]

    def plot_confusion_matrix(self):
        self.confusion_matrix = np.divide(self.confusion_matrix, np.tile(self.confusion_matrix.sum(axis=1),(4,1)).T)
        sns.set(rc={"figure.figsize":self.confusion_matrix_plot_size})
        sns.heatmap(self.confusion_matrix, annot=True, 
                            xticklabels=self.dataset_label,
                            yticklabels=False,
                            cmap="YlGnBu", linewidths=0.5, fmt=".3f", vmin=0, vmax=1)
        plt.show()

    def plot_confidence_distribution(self):  ## testing
        num_class = len(self.dataset_config["label_mapping"])
        plt.figure(figsize=self.confidence_plot_size)
        for i in range(num_class):
            data, x, y = [], [], []
            for index, j in enumerate(self.predictions):
                j = j[0]
                if np.argmax(j) != np.argmax(self.true_labels[index]):
                    data.append(j[i])
            data = np.array(data)
            plt.subplot(num_class, 1, i+1)
            plt.title(self.dataset_config["label_mapping"][i])
            sns.histplot(data, kde=True, bins=10, binrange=[0,1])
        plt.show()

    def print_metrics(self):
        print(f"Performance of {self.model_path}:\n",
                f"fps: {round(self.fps)}\n",
                f"crossentropy: {round(self.crossentropy,3)}\n",
                f"precision: {[round(i,2) for i in self.precision]}\n",
                f"recall: {[round(i,2) for i in self.recall]}\n",
            )

    def run(self, metrics):
        self._define_attr(metrics)
        self.print_metrics()
        if self.confusion_matrix_plot_size:
            self.plot_confusion_matrix()
        if self.confidence_plot_size:
            self.plot_confidence_distribution()
