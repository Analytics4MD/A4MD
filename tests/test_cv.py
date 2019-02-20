from base_test import BaseTest
import pytest


class TestCollectiveVariables(BaseTest):
    """
    Test class for testing collective variables 
    """

    def test_rmsd_direct(self, ref_xyz, xyz):
        import freud.msd as msd
        import numpy as np

        direct_msd = msd.MSD(mode='direct')
        positions = []
        positions.append(ref_xyz)
        positions.append(xyz)
        positions = np.asarray(positions)

        msd_val = direct_msd.compute(positions)
        rmsd = np.sqrt(msd_val.msd)
        ermsd = [0.0, 3.0] 
        ermsd = np.sqrt(ermsd)
        assert np.isclose(rmsd,ermsd).all()

    def test_rmsd_window(self, ref_xyz, xyz):
        import freud.msd as msd
        import numpy as np

        direct_msd = msd.MSD(mode='window')
        positions = []
        positions.append(ref_xyz)
        positions.append(xyz)
        positions = np.asarray(positions)
        msd_val = direct_msd.compute(positions)
        print('msd',msd_val.msd)
        rmsd = np.sqrt(msd_val.msd)
        ermsd = [0.0, 3.0] 
        ermsd = np.sqrt(ermsd)
        assert np.isclose(rmsd,ermsd).all()
