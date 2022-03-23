import os
import json

import torch
from torchvision import transforms, datasets
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from prettytable import PrettyTable


class ConfusionMatrix(object):

    def __init__(self, num_classes: int, labels: list):
        self.matrix = np.zeros((num_classes, num_classes))
        self.num_classes = num_classes
        self.labels = labels
        self.matrix_prob = np.zeros((num_classes, num_classes))
        self.Precision = []
        self.Recall = []
        self.Specificity = []

    def update(self, preds, labels):
        for p, t in zip(preds, labels):
            self.matrix[p, t] += 1
        for num in range(self.num_classes):
            self.matrix_prob[0:, num] = self.matrix[0:, num] / np.sum(self.matrix[0:, num])

    def summary(self):
        # calculate accuracy
        sum_TP = 0
        for i in range(self.num_classes):
            sum_TP += self.matrix[i, i]
        acc = sum_TP / np.sum(self.matrix)
        print("the model accuracy is ", acc)

        # precision, recall, specificity
        table = PrettyTable()
        table_m = PrettyTable()
        table_mp = PrettyTable()
        table.field_names = ["", "Precision", "Recall", "Specificity"]
        table_m.field_names = [""] + self.labels
        table_mp.field_names = [""] + self.labels
        for i in range(self.num_classes):
            TP = self.matrix[i, i]
            FP = np.sum(self.matrix[i, :]) - TP
            FN = np.sum(self.matrix[:, i]) - TP
            TN = np.sum(self.matrix) - TP - FP - FN
            Precision = round(TP / (TP + FP), 3) if TP + FP != 0 else 0.
            Recall = round(TP / (TP + FN), 3) if TP + FN != 0 else 0.
            Specificity = round(TN / (TN + FP), 3) if TN + FP != 0 else 0.
            table.add_row([self.labels[i], Precision, Recall, Specificity])
            self.Precision.append(Precision)
            self.Recall.append(Recall)
            self.Specificity.append(Specificity)
        for row in range(self.num_classes):
            table_m.add_row([self.labels[row]] + self.matrix[row][:].tolist())
            table_mp.add_row([self.labels[row]] + self.matrix_prob[row][:].tolist())
        print(table)
        print(table_m)
        print(table_mp)

    def plot(self):
        matrix = self.matrix
        print(matrix)
        plt.imshow(matrix, cmap=plt.cm.Blues)
        plt.xticks(range(self.num_classes), self.labels, rotation=45)
        plt.yticks(range(self.num_classes), self.labels)
        plt.colorbar()
        plt.xlabel('True Labels')
        plt.ylabel('Predicted Labels')
        plt.title('Confusion matrix')

        thresh = matrix.max() / 2
        for x in range(self.num_classes):
            for y in range(self.num_classes):
                info = int(matrix[y, x])
                plt.text(x, y, info,
                         verticalalignment='center',
                         horizontalalignment='center',
                         color="white" if info > thresh else "black")
        plt.tight_layout()
        plt.show()

    def get_epoch_matrix_dict(self):
        """
        返回的是两个字典
        """
        confusion_matrix_dict = {}
        confusion_matrix_pb_dict = {}
        for i in range(self.num_classes):
            confusion_matrix_dict = dict(confusion_matrix_dict, **{self.labels[i]: self.matrix[i][:].tolist()})
            confusion_matrix_pb_dict = dict(confusion_matrix_pb_dict,
                                            **{self.labels[i]: self.matrix_prob[i][:].tolist()})
        return confusion_matrix_dict, confusion_matrix_pb_dict
