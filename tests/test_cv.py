from base_test import BaseTest
import pytest


class TestCollectiveVariables(BaseTest):
    """
    Test class for testing collective variables 
    """

    def test_rmsd_direct(self, ref_xyz, xyz):
        import a4md.analysis.trajectory_helper as th
        import numpy as np

        rmsd = th.get_rmsd(xyz,ref_xyz, method='direct')
        ermsd = [0.0, 3.0] 
        ermsd = np.sqrt(ermsd)
        assert np.isclose(rmsd,ermsd).all()

    def test_rmsd_window(self, ref_xyz, xyz):
        import a4md.analysis.trajectory_helper as th
        import numpy as np

        rmsd = th.get_rmsd(xyz,ref_xyz, method='window')
        ermsd = [0.0, 3.0] 
        ermsd = np.sqrt(ermsd)
        assert np.isclose(rmsd,ermsd).all()
