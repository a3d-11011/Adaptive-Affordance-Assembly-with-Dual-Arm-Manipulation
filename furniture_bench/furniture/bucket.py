from furniture_bench.utils.pose import get_mat
from furniture_bench.furniture.furniture import Furniture
from furniture_bench.furniture.parts.bucket_body import BucketBody
from furniture_bench.assemble_config import config

class Bucket(Furniture):
    def __init__(self):
        super().__init__()
        self.name="bucket"
        furniture_conf=config["furniture"]["bucket"]
        self.furniture_conf=furniture_conf
        self.parts=[
            BucketBody(furniture_conf["bucket_body"], 0)
        ]
        self.num_parts=len(self.parts)