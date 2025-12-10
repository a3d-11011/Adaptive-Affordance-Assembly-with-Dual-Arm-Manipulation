from furniture_bench.furniture.cabinet import Cabinet
from furniture_bench.furniture.furniture import Furniture
from furniture_bench.furniture.one_leg import OneLeg
from furniture_bench.furniture.stool import Stool
from furniture_bench.furniture.square_table import SquareTable
from furniture_bench.furniture.oval_table import OvalTable
from furniture_bench.furniture.round_table import RoundTable
from furniture_bench.furniture.drawer import Drawer
from furniture_bench.furniture.chair import Chair
from furniture_bench.furniture.desk import Desk
from furniture_bench.furniture.desk_trapezoid import DeskTrapezoid
from furniture_bench.furniture.desk_rectangle import DeskRectangle
from furniture_bench.furniture.desk_triangle import DeskTriangle
from furniture_bench.furniture.lamp import Lamp
from furniture_bench.furniture.desk_pull import Desk_pull
from furniture_bench.furniture.bucket import Bucket
from furniture_bench.furniture.box import Box


def furniture_factory(furniture: str) -> Furniture:
    if furniture == "square_table":
        return SquareTable()
    elif furniture == "desk":
        return Desk()
    elif furniture == "desk_trapezoid":
        return DeskTrapezoid()
    elif furniture == "desk_rectangle":
        return DeskRectangle()
    elif furniture == "desk_triangle":
        return DeskTriangle()
    elif furniture == "round_table":
        return RoundTable()
    elif furniture == "oval_table":
        return OvalTable()
    elif furniture == "drawer":
        return Drawer()
    elif furniture == "chair":
        return Chair()
    elif furniture == "lamp":
        return Lamp()
    elif furniture == "cabinet":
        return Cabinet()
    elif furniture == "stool":
        return Stool()
    elif furniture == "one_leg":
        return OneLeg()
    elif furniture == "desk_pull":
        return Desk_pull()
    elif furniture == "bucket":
        return Bucket()
    elif furniture == "box":
        return Box()
    else:
        raise ValueError(f"Unknown furniture type: {furniture}")
