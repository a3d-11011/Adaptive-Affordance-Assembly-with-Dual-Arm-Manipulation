from furniture_bench.utils.pose import get_mat
from furniture_bench.furniture.furniture import Furniture
from furniture_bench.furniture.parts.desk_leg import DeskLeg
from furniture_bench.furniture.parts.desk_table_top import DeskTableTop
from furniture_bench.assemble_config import config


class Desk_pull(Furniture):
    def __init__(self):
        super().__init__()
        self.name = "desk_pull"
        furniture_conf = config["furniture"]["desk_pull"]
        
        self.furniture_conf = furniture_conf
        self.tag_size = furniture_conf["tag_size"]
        self.parts = [
            DeskTableTop(furniture_conf["desk_top"], 0),
            DeskLeg(furniture_conf["desk_leg1"], 1),
            DeskLeg(furniture_conf["desk_leg2"], 2),
            DeskLeg(furniture_conf["desk_leg3"], 3),
        ]
        self.num_parts = len(self.parts)
        
        self.should_be_assembled = [(0, 1), (0, 2), (0, 3)]


        