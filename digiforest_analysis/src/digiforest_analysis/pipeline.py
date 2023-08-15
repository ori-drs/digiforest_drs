from open3d.geometry import PointCloud
from digiforest_analysis.tasks import GroundSegmentation
from digiforest_analysis.tasks import TreeSegmentation
from digiforest_analysis.tasks import TreeAnalysis
from digiforest_analysis.tasks import ForestAnalysis
from digiforest_analysis.tasks import TemporalAnalysis


class Pipeline:
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        self._ground_segmentation = GroundSegmentation()
        self._tree_segmentation = TreeSegmentation()
        self._tree_analysis = TreeAnalysis()
        self._forest_analysis = ForestAnalysis()

        # If incremental analysis, then do temporal analysis
        self._temporal_analysis = TemporalAnalysis()

    def process(self, cloud: PointCloud):
        report = {}

        # Extract the ground
        self._ground_cloud, self._forest = self._ground_segmentation.process(
            cloud=cloud
        )

        # Extract the trees from the forest cloud
        self._trees = self._tree_segmentation.process(cloud=self._forest)

        # Get the specific attributes of each tree
        self._trees = self._tree_analysis.process(trees=self._trees)

        # Get general attributes from the full forest
        self._forest_attributes = self._forest_analysis.process(trees=self._trees)

        return report

    @property
    def ground(self):
        return self._ground_cloud

    @property
    def forest(self):
        return self._forest

    @property
    def trees(self):
        return self._trees

    @property
    def forest_attributes(self):
        return self._forest_attributes
