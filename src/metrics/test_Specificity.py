import unittest
from .Specificity import Specificity
import numpy as np
from tensorflow.keras import backend as K
from numpy.testing import *

class SpecificityCase(unittest.TestCase):
    

    def testNone(self):
        gt = np.zeros([1, 100, 100, 1])
        pr = np.zeros([1, 100, 100, 1])
        assert_almost_equal(Specificity()(gt, pr), 1)

    def testAll(self):
        gt = np.zeros([1, 100, 100, 1])
        pr = np.ones([1, 100, 100, 1])
        assert_almost_equal(Specificity()(gt, pr, ), 0)

    def testQuarter(self):
        gt = np.zeros([1, 100, 100, 1])
        pr = np.zeros([1, 100, 100, 1])
        pr[0, 50:100, 50:100, 0] = 1.
        assert_almost_equal(Specificity()(gt, pr), 1 - np.sum(pr) / np.size(gt))

    def testHalf(self):
        gt = np.zeros([1, 100, 100, 1])
        pr = np.zeros([1, 100, 100, 1])
        pr[0, 75:100, 75:100, 0] = 1.
        assert_almost_equal(Specificity()(gt, pr),  1 - np.sum(pr) / np.size(gt))

    def testFew(self):
        gt = np.zeros([1, 100, 100, 1])
        pr = np.zeros([1, 100, 100, 1])
        pr[0, 99:100, 99:100, 0] = 1.
        assert_almost_equal(Specificity()(gt, pr),  1 - np.sum(pr) / np.size(gt))

    def testAllMatch(self):
        gt = np.ones([1, 100, 100, 1])
        pr = np.ones([1, 100, 100, 1])
        assert_almost_equal(Specificity()(gt, pr),  1)

    def testQuarterPositiveNoPredictions(self):
        gt = np.zeros([1, 100, 100, 1])
        gt[0, 50:100, 50:100, 0] = 1.
        pr = np.zeros([1, 100, 100, 1])
        assert_almost_equal(Specificity()(gt, pr), 1)

    def testQuarterPositive(self):
        gt = np.zeros([1, 100, 100, 1])
        gt[0, 50:, 50:, 0] = 1.
        pr = np.zeros([1, 100, 100, 1])
        pr[0, :50, :50, 0] = 1.
        assert_almost_equal(Specificity()(gt, pr), 1 - np.sum(pr) / (np.size(gt) - np.sum(gt)))

if __name__ == "__main__":
    unittest.main()