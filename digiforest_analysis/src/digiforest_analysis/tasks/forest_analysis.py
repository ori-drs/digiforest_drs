from digiforest_analysis.tasks import BaseTask


class ForestAnalysis(BaseTask):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _process(self, **kwargs):
        forest = kwargs.get("forest")
        trees = kwargs.get("trees")

        report = {}

        # Get bounding box of forest
        bbox_forest = forest.get_axis_aligned_bounding_box()
        bbox_forest_dims = bbox_forest.get_extent()
        report["area"] = (bbox_forest_dims[0] * bbox_forest_dims[1]).item()

        # Number of trees
        report["num_trees"] = len(trees)

        return report
