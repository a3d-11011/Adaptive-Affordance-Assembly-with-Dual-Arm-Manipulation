from furniture_bench.furniture.furniture import Furniture
from furniture_bench.furniture.parts.cask_body import CaskBody
from furniture_bench.assemble_config import config


class Cask(Furniture):
    def __init__(self):
        super().__init__()
        self.name="cask"
        furniture_conf=config["furniture"]["cask"]
        self.furniture_conf=furniture_conf
        self.parts=[
            CaskBody(furniture_conf["cask_body"], 0)
        ]
        self.num_parts=len(self.parts)