import unittest
import numpy as np

from digiforest_analysis_ros.forest_analysis import TreeManager


class TestForestAnalysis(unittest.TestCase):
    def test_angle_coverage_dicontinuity_once_negative(self):
        # Jumping over discontinuity once in negative direction
        tm = TreeManager()
        cluster = {"info": {"axis": {"center": np.array([0, 0, 0])}}}
        poses = np.array([[0.5, 0.5], [0.5, -0.5], [-0.5, -0.5]])

        angle_from, angle_to, _, _ = tm.calculate_coverage(cluster, poses)
        self.assertAlmostEqual(angle_from, 1.25 * np.pi)
        self.assertAlmostEqual(angle_to, 0.25 * np.pi)

    def test_angle_coverage_dicontinuity_once_negative2(self):
        # Jumping over discontinuity once in negative direction with a larger coverage
        tm = TreeManager()
        cluster = {"info": {"axis": {"center": np.array([0, 0, 0])}}}
        poses = np.array([[0.5, 0.5], [0.5, -0.5], [-0.5, -0.5], [-0.5, 0.5]])

        angle_from, angle_to, _, _ = tm.calculate_coverage(cluster, poses)
        self.assertAlmostEqual(angle_from, 0.75 * np.pi)
        self.assertAlmostEqual(angle_to, 0.25 * np.pi)

    def test_angle_coverage_dicontinuity_once_positive(self):
        # Jumping over discontinuity once in positive direction
        tm = TreeManager()
        cluster = {"info": {"axis": {"center": np.array([0, 0, 0])}}}
        poses = np.array([[0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]])

        angle_from, angle_to, _, _ = tm.calculate_coverage(cluster, poses)
        self.assertAlmostEqual(angle_from, 1.75 * np.pi)
        self.assertAlmostEqual(angle_to, 0.75 * np.pi)

    def test_angle_coverage_dicontinuity_twice(self):
        # Jumping over discontinuity twice
        tm = TreeManager()
        cluster = {"info": {"axis": {"center": np.array([0, 0, 0])}}}
        poses = np.array([[0.5, -0.5], [0.5, 0.5], [0.5, -0.5]])

        angle_from, angle_to, _, _ = tm.calculate_coverage(cluster, poses)
        self.assertAlmostEqual(angle_from, 1.75 * np.pi)
        self.assertAlmostEqual(angle_to, 0.25 * np.pi)

    def test_angle_coverage_360(self):
        # 360 degree coverage
        tm = TreeManager()
        cluster = {"info": {"axis": {"center": np.array([0, 0, 0])}}}
        poses = np.array([[0.5, -0.5], [0.5, 0.5], [-0.5, -0.5], [0.5, -0.5]])

        angle_from, angle_to, _, _ = tm.calculate_coverage(cluster, poses)
        self.assertAlmostEqual(angle_from, 0)
        self.assertAlmostEqual(angle_to, 2 * np.pi)

    def test_distance_coverage_same(self):
        # Distance coverage, all same distances
        tm = TreeManager()
        cluster = {"info": {"axis": {"center": np.array([0, 0, 0])}}}
        poses = np.array([[0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]])

        _, _, min_distance, max_distance = tm.calculate_coverage(cluster, poses)
        self.assertAlmostEqual(min_distance, np.sqrt(0.5))
        self.assertAlmostEqual(max_distance, np.sqrt(0.5))

    def test_distance_coverage_different(self):
        # Distance coverage, different distances
        tm = TreeManager()
        cluster = {"info": {"axis": {"center": np.array([0, 0, 0])}}}
        poses = np.array([[1, -1], [0.5, 0.5], [-0.75, 0.75]])

        _, _, min_distance, max_distance = tm.calculate_coverage(cluster, poses)
        self.assertAlmostEqual(min_distance, np.sqrt(0.5))
        self.assertAlmostEqual(max_distance, np.sqrt(2.0))

    def test_angle_coverage_check_psoitive(self):
        # Check how two arc segments of length 90 deg are merged
        tm = TreeManager(reco_min_angle_coverage=np.pi)

        result = tm.check_angle_coverage([(0, 0.5 * np.pi), (np.pi, 1.5 * np.pi)])
        self.assertEqual(result, True)

    def test_angle_coverage_check_negative(self):
        # Check how two arc segments of length 90 deg are merged
        tm = TreeManager(reco_min_angle_coverage=np.pi)

        result = tm.check_angle_coverage([(0, 0.5 * np.pi), (np.pi, 1.49 * np.pi)])
        self.assertEqual(result, False)


if __name__ == "__main__":
    TESTING_METHOD = "empirical"  # "unit" or "empirical

    if TESTING_METHOD == "unit":
        unittest.main()

    elif TESTING_METHOD == "empirical":
        from matplotlib import pyplot as plt
        from matplotlib.patches import Arc
        import matplotlib

        matplotlib.use("TkAgg")

        num_poses = 5
        poses = np.random.rand(num_poses, 2) - 0.5

        cluster = {"info": {"axis": {"center": np.array([0, 0, 0])}}}
        tm = TreeManager()
        min_angle, max_angle, min_distance, max_distance = tm.calculate_coverage(
            cluster, poses
        )
        print(np.rad2deg(max_angle - min_angle))

        plt.figure()
        plt.gca().set_aspect("equal")
        # plt.plot(poses[:, 0], poses[:, 1], c="r")
        plt.scatter(0, 0, c="k")
        for i in range(poses.shape[0] - 1):
            plt.arrow(
                poses[i, 0],
                poses[i, 1],
                poses[i + 1, 0] - poses[i, 0],
                poses[i + 1, 1] - poses[i, 1],
                head_width=0.05,
                length_includes_head=True,
            )
        plt.xlim(-1, 1)
        plt.ylim(-1, 1)
        r = np.sqrt(0.5)
        plt.gca().add_artist(
            Arc((0, 0), 2 * r, 2 * r, 0, np.rad2deg(min_angle), np.rad2deg(max_angle))
        )
        plt.plot(
            [0, r * np.cos(min_angle)],
            [0, r * np.sin(min_angle)],
            c="k",
            linewidth=0.75,
        )
        plt.plot(
            [0, r * np.cos(max_angle)],
            [0, r * np.sin(max_angle)],
            c="k",
            linewidth=0.75,
        )
        # plt.gca().add_artist(Arc((0, 0), 1, 1, 0, -170, 170))
        plt.show()
        print(poses)
