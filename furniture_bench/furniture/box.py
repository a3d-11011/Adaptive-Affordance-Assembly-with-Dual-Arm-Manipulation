from furniture_bench.utils.pose import get_mat
from furniture_bench.furniture.furniture import Furniture
from furniture_bench.furniture.parts.box_body import BoxBody
from furniture_bench.assemble_config import config

class Box(Furniture):
    def __init__(self):
        super().__init__()
        self.name="box"
        furniture_conf=config["furniture"]["box"]
        self.furniture_conf=furniture_conf
        self.parts=[
            BoxBody(furniture_conf["box_body"], 0)
        ]
        self.num_parts=len(self.parts)