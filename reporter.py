import os
import numpy as np
import seaborn as sns
import onnxruntime
import matplotlib.pyplot as plt
from datetime import datetime

class Reporter:

    def __init__(self):
        self.datetime_now = str(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        self.datetime_now = self.datetime_now.replace(" ","_")
        self.result_log = f"./reports/text/run_on_{self.datetime_now}.txt"

    def _define_attr(self, metrics):
        (self.reporter_config, self.dataset_config, self.model_config, self.predictions, self.true_labels,
        self.confusion_matrix, self.precision, self.recall, self.crossentropy, self.fps) = metrics

        self.predictions = [i[0] for i in self.predictions]
        self.true_labels = [i[0] for i in self.true_labels]

        self.confusion_matrix_plot_size = self.reporter_config["confusion_matrix_plot_size"]
        if self.confusion_matrix_plot_size:
            self.confusion_matrix_plot_size = tuple(self.confusion_matrix_plot_size)

        self.confidence_plot_size = self.reporter_config["confidence_plot_size"]
        if self.confidence_plot_size:
            self.confidence_plot_size = tuple(self.confidence_plot_size)

    def plot_confusion_matrix(self):
        self.confusion_matrix = np.divide(self.confusion_matrix, np.tile(self.confusion_matrix.sum(axis=1),(4,1)).T)
        sns.set(rc={"figure.figsize":self.confusion_matrix_plot_size})
        sns_plot = sns.heatmap(self.confusion_matrix, annot=True, 
                            xticklabels=self.dataset_config["label_mapping"],
                            yticklabels=False,
                            cmap="YlGnBu", linewidths=0.5, fmt=".3f", vmin=0, vmax=1)
        save_name = f"./reports/figure/{self.datetime_now}_{os.path.splitext(self.model_config['model_path'].split('/')[-1])[0]}.png"
        if not os.path.exists(save_name):
            plt.savefig(save_name)  
            print(f"confusion matrix is saved to {save_name}") 
        else:
            print("cannot save as figure name already exist")     
        plt.show()

    def plot_confidence_distribution(self):  ## testing
        num_class = len(self.dataset_config["label_mapping"])
        plt.figure(figsize=self.confidence_plot_size)
        for i in range(num_class):
            data, x, y = [], [], []
            for index, j in enumerate(self.predictions):
                if np.argmax(j) != np.argmax(self.true_labels[index]):
                    data.append(j[i])
            data = np.array(data)
            plt.subplot(num_class, 1, i+1)
            plt.title(self.dataset_config["label_mapping"][i])
            sns.histplot(data, kde=True, bins=10, binrange=[0,1])
        plt.show()

    def print_metrics(self):
        message = [f"Performance of {self.model_config['model_path']}:\n",
                    f" fps on {onnxruntime.get_device()}: {round(self.fps,1)}\n",
                    f" crossentropy:{round(self.crossentropy,3)}\n",
                    f" precision:{[round(i,2) for i in self.precision]}\n",
                    f" recall:{[round(i,2) for i in self.recall]}\n",
                    f" label:{self.dataset_config['label_mapping']}\n",
                    ]
        with open(self.result_log,"a") as log:
            log.writelines(message)
            log.close()
        for i in message:
            print(i)

    def run(self, metrics):
        self._define_attr(metrics)
        self.print_metrics()
        if self.confusion_matrix_plot_size:
            self.plot_confusion_matrix()
        if self.confidence_plot_size:
            self.plot_confidence_distribution()
