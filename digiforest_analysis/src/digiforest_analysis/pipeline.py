from open3d.geometry import PointCloud
from digiforest_analysis.tasks import GroundSegmentation
from digiforest_analysis.tasks import TreeSegmentation
from digiforest_analysis.tasks import TreeAnalysis
from digiforest_analysis.tasks import ForestAnalysis
from digiforest_analysis.tasks import TemporalAnalysis

from digiforest_analysis.utils import io
from pathlib import Path

import yaml


class Pipeline:
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        self._cloud = None
        self._header = None

        filename = kwargs.get("file", None)
        if filename is not None:
            self.load_cloud(filename)

        out_dir = kwargs.get("out_dir", "/tmp")
        self.setup_output(out_dir)

        # Setup modules
        params = kwargs.get("ground_segmentation", {})
        self._ground_segmentation = GroundSegmentation(**params)

        params = kwargs.get("tree_segmentation", {})
        self._tree_segmentation = TreeSegmentation(**params)

        params = kwargs.get("tree_analysis", {})
        self._tree_analysis = TreeAnalysis(**params)

        params = kwargs.get("forest_analysis", {})
        self._forest_analysis = ForestAnalysis(**params)

        # If incremental analysis, then do temporal analysis
        params = kwargs.get("forest_analysis", {})
        self._temporal_analysis = TemporalAnalysis(**params)

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

    def load_cloud(self, filename):
        # Check validity of input
        filename = Path(filename)
        if not filename.exists():
            raise ValueError(f"Input file [{filename}] does not exist")

        # Get file attributes
        self._cloud_name = filename.name
        self._cloud_format = filename.suffix

        # Read cloud
        print(f"Loading {filename}...")
        self._cloud, self._header = io.load(str(filename), binary=True)
        assert len(self._cloud.point.positions) > 0

        # Compute normals if not available
        if not hasattr(self._cloud.point, "normals"):
            print("No normals found in cloud, computing...", end="")
            self._cloud.estimate_normals()
            print("done")

    def setup_output(self, output_dir):
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)

    def save_cloud(self, cloud, label="default"):
        filename = Path(self._output_dir).joinpath(label + self._cloud_format)
        if self._header is None:
            header_fix = None
        else:
            header_fix = {"VIEWPOINT": self._header["VIEWPOINT"]}
        io.write(cloud, header_fix, str(filename))

    def save_trees(self, trees, label="trees"):
        save_folder = Path(self._output_dir).joinpath(label)
        save_folder.mkdir(parents=True, exist_ok=True)
        if self._header is None:
            header_fix = None
        else:
            header_fix = {"VIEWPOINT": self._header["VIEWPOINT"]}

        for tree in trees:
            # Write clouds
            i = tree["info"]["id"]
            tree_cloud = tree["cloud"]

            # Write cloud
            tree_cloud_filename = Path(
                save_folder, f"tree_cloud_{i:04}{self._cloud_format}"
            )
            io.write(tree_cloud, header_fix, str(tree_cloud_filename))

            # Write tree info
            tree_info_filename = Path(save_folder, f"tree_cloud_{i:04}.yaml")
            with open(str(tree_info_filename), "w") as yaml_file:
                yaml.dump(tree["info"], yaml_file, indent=4)

    def process(self, cloud: PointCloud = None, header: dict = None):
        if cloud is not None and header is not None:
            self._cloud = cloud
            self._header = header

        if self._cloud is None:
            raise ValueError("'cloud or header empty'")

        # Extract the ground
        print("Extracting ground...")
        self._ground_cloud, self._forest_cloud = self._ground_segmentation.process(
            cloud=self._cloud
        )
        self.save_cloud(self._ground_cloud, label="ground_cloud")
        self.save_cloud(self._forest_cloud, label="forest_cloud")

        # Extract the trees from the forest cloud
        print("Segmenting trees...")
        self._trees = self._tree_segmentation.process(cloud=self._forest_cloud)
        self.save_trees(self._trees, label="segmented")

        # Get the specific attributes of each tree
        print("Analyzing trees...")
        self._trees = self._tree_analysis.process(
            trees=self._trees, ground_cloud=self._ground_cloud
        )
        self.save_trees(self._trees, label="filtered")

        # Get general attributes from the full forest
        print("Analyzing forest patch...")
        report = self._forest_analysis.process(
            forest=self._forest_cloud, trees=self._trees
        )

        return report
