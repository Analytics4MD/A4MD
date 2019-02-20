import pytest
import numpy as np


class BaseTest:

    @pytest.fixture
    def ref_xyz(self):
        array = np.asarray([[0.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0]])
        return array

    @pytest.fixture
    def xyz(self):
        array = np.asarray([[1.0, 1.0, 1.0],
                  [1.0, 1.0, 1.0],
                  [1.0, 1.0, 1.0],
                  [1.0, 1.0, 1.0],
                  [1.0, 1.0, 1.0],
                  [1.0, 1.0, 1.0],
                  [1.0, 1.0, 1.0],
                  [1.0, 1.0, 1.0],
                  [1.0, 1.0, 1.0],
                  [1.0, 1.0, 1.0]])
        return array
