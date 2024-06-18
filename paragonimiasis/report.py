import pandas as pd
import numpy as np

class Report:
    _table = {}
    _columns = []

    def __init__(self, columns, classifier):
        self._columns = []
        for column in columns:
            if column != classifier and column != 'Unnamed: 0':
                self._columns.append(column)

        for column in self._columns:
            self._table[column] = []

        self._table['accuracy'] = []
        self._table['mcc'] = []
        self._table['tp'] = []
        self._table['fn'] = []
        self._table['fp'] = []
        self._table['tn'] = []

    def record(self, subset, report):
        for column in self._columns:
            if column in subset:
                self._table[column].append(1)
            else:
                self._table[column].append(0)

        self._table['accuracy'].append(report['accuracy'])
        self._table['mcc'].append(report['mcc'])

        cm = report['confusion_matrix']
        self._table['tp'].append(cm[0,0])
        self._table['fn'].append(cm[0,1])
        self._table['fp'].append(cm[1,0])
        self._table['tn'].append(cm[1,1])

    def as_dataframe(self):
        return pd.DataFrame(self._table)
