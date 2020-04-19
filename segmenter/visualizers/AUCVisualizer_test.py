import json
import os
import unittest
from numpy.testing import *
from matplotlib import pyplot as plt
from auc import *


class TestEvaluate(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.results = results_from_path("./test_fixtures/roc")

    def test_loads(self):
        self.assertGreater(len(self.results), 0)

    def test_auc(self):
        tpr, fpr, auc = compile_results(self.results["Vote"])
        plot = plot_results("Vote", tpr, fpr, auc)
        plot.show()


if __name__ == "__main__":
    unittest.main()
