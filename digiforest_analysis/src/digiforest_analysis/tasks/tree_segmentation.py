from digiforest_analysis.tasks import BaseTask


class TreeSegmentation(BaseTask):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _process(self, **kwargs):
        cloud = kwargs.get("cloud")
        assert len(cloud.point.normals) > 0

        # Implement your code here
        #

        trees = {}
        return trees
