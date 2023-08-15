from digiforest_analysis.tasks import BaseTask


class ForestAnalysis(BaseTask):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _process(self, **kwargs):
        trees = kwargs.get("trees")

        # Implement your code here
        #

        return trees
