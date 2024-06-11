# Alpha 1
# Beta 5
# Gamma 5
# Lambda 1
# DieSize 0.0 0.0 50.0 30.0
# NumInput 3
# Input INPUT0 0 5
# Input INPUT1 0 25
# Input CK0 0 15
# NumOutput 2
# Output OUTPUT0 50 5
# Output OUTPUT1 50 15
# Output OUTPUT2 50 25
# FlipFlop 1 FF1 5.0 10.0 3
# Pin D 0.0 8.0
# Pin Q 5.0 8.0
# Pin CLK 0.0 2.0
# FlipFlop 2 FF2 8.0 10.0 5
# Pin D0 0.0 9.0
# Pin D1 0.0 6.0
# Pin Q0 8.0 9.0
# Pin Q1 8.0 6.0
# Pin CLK 0.0 2.0
# Gate G1 5.0 10.0 2
# Pin IN 0.0 8.0
# Pin OUT 5.0 2.0
# NumInstances 4
# Inst C1 FF1 20.0 0.0
# Inst C2 FF1 20.0 10.0
# Inst C3 FF1 20.0 20.0
# Inst C4 G1 10.0 10.0
# NumNets 4
# Net N1 3
# Pin INPUT0
# Pin C1/D
# Pin C2/D
# Net N2 2
# Pin INPUT1
# Pin C3/D
# Net N3 2
# Pin C1/Q
# Pin OUTPUT0
# Net N4 2
# Pin C2/Q
# Pin OUTPUT1
# Net N5 2
# Pin C3/Q
# Pin OUTPUT2
# Net CK0 3
# Pin CLK0
# Pin C1/CLK
# Pin C4/IN
# Net CK1 3
# Pin C4/OUT
# Pin C2/CLK
# Pin C3/CLK
# BinWidth 10.0
# BinHeight 10.0
# BinMaxUtil 79.0
# PlacementRows 0.0 0.0 2.0 10.0 10
# PlacementRows 0.0 10.0 2.0 10.0 10
# PlacementRows 0.0 20.0 2.0 10.0 10
# DisplacementDelay 0.01
# QpinDelay FF1 1.0
# QpinDelay FF2 2.0
# TimingSlack C1 D 1.0
# TimingSlack C2 D 1.0
# TimingSlack C3 D 1.0
# GatePower FF1 10.0
# GatePower FF2 17.0
from dataclasses import dataclass, field
# from functools import cached_property
from cached_property import cached_property

import networkx as nx

from .plot import *
from .utility import *
from typing import List, Dict

@dataclass
class DieSize:
    xLowerLeft: float
    yLowerLeft: float
    xUpperRight: float
    yUpperRight: float

    def __post_init__(self):
        self.xLowerLeft = float(self.xLowerLeft)
        self.yLowerLeft = float(self.yLowerLeft)
        self.xUpperRight = float(self.xUpperRight)
        self.yUpperRight = float(self.yUpperRight)


@dataclass
class Flip_Flop:
    bits: int
    name: str
    width: float
    height: float
    num_pins: int
    pins: List = field(default_factory=list,repr=False)
    pins_query: dict = field(init=False, repr=False)
    qpin_delay: float = field(default=None)
    power: float = field(init=False)

    def __post_init__(self):
        self.bits = int(self.bits)
        self.width = float(self.width)
        self.height = float(self.height)
        self.num_pins = int(self.num_pins)

    @property
    def area(self):
        return self.width * self.height


@dataclass
class Gate:
    name: str
    width: float
    height: float
    num_pins: int
    pins: List = field(default_factory=list)
    pins_query: dict = field(init=False)

    def __post_init__(self):
        self.width = float(self.width)
        self.height = float(self.height)
        self.num_pins = int(self.num_pins)


@dataclass
class Pin:
    name: str
    x: float = None
    y: float = None
    inst_name: str = field(init=False, default=None)

    def __post_init__(self):
        if self.x:
            self.x = float(self.x)
        if self.y:
            self.y = float(self.y)


@dataclass
class PhysicalPin:
    name: str
    inst: object = field(default=None)
    slack: float = field(default=None, init=False)

    @property
    def pos(self):
        if isinstance(self.inst, Inst):
            return (
                self.inst.x + self.inst.lib.pins_query[self.name].x,
                self.inst.y + self.inst.lib.pins_query[self.name].y,
            )
        else:
            return (self.inst.x, self.inst.y)

    @property
    def rel_pos(self):
        if isinstance(self.inst, Inst):
            return (
                self.inst.lib.pins_query[self.name].x,
                self.inst.lib.pins_query[self.name].y,
            )
        else:
            return (0, 0)

    @property
    def full_name(self):
        if isinstance(self.inst, Inst):
            return self.inst.name + "/" + self.name
        else:
            return self.name

    @property
    def is_ff(self):
        return self.inst.is_ff

    @property
    def is_io(self):
        return self.inst.is_io

    @property
    def is_gt(self):
        return not self.inst.is_io and not self.inst.is_ff

    @property
    def inst_name(self):
        return self.inst.name


@dataclass
class Inst:
    name: str
    lib_name: str
    x: float
    y: float
    lib: object = field(init=False, repr=False)
    pins: List[PhysicalPin] = field(default_factory=list, init=False, repr=False)
    pins_query: dict = field(init=False, repr=False)
    is_io: bool = field(init=False, default=False, repr=False)

    def __post_init__(self):
        self.x = float(self.x)
        self.y = float(self.y)

    @property
    def qpin_delay(self):
        return self.lib.qpin_delay

    @property
    def is_ff(self):
        return isinstance(self.lib, Flip_Flop)

    def assign_pins(self, pins):
        self.pins = pins
        self.pins_query = {pin.name: pin for pin in pins}

    @property
    def pos(self):
        return self.x, self.y

    def moveto(self, xy):
        self.x = xy[0]
        self.y = xy[1]

    @property
    def dpins(self):
        return [pin.full_name for pin in self.pins if pin.name.startswith("d")]

    @property
    def box(self):
        return BoxContainer(self.lib.width, self.lib.height, offset=(self.x, self.y)).box


@dataclass
class Input:
    name: str
    x: float
    y: float
    pins: List[PhysicalPin] = field(init=False)
    is_io: bool = field(init=False, default=True, repr=False)
    is_ff: bool = field(init=False, default=False, repr=False)

    def __post_init__(self):
        self.x = float(self.x)
        self.y = float(self.y)
        self.pins = [PhysicalPin(self.name, self)]


@dataclass
class Output:
    name: str
    x: float
    y: float
    pins: List[PhysicalPin] = field(init=False)
    is_io: bool = field(init=False, default=True, repr=False)
    is_ff: bool = field(init=False, default=False, repr=False)

    def __post_init__(self):
        self.x = float(self.x)
        self.y = float(self.y)
        self.pins = [PhysicalPin(self.name, self)]


@dataclass
class Net:
    name: str
    num_pins: int
    pins: List[PhysicalPin] = field(default_factory=list)
    metadata: str = field(init=False, default=None)

    def __post_init__(self):
        self.num_pins = int(self.num_pins)


@dataclass
class PlacementRows:
    x: float
    y: float
    width: float
    height: float
    num_cols: int

    def __post_init__(self):
        self.x = float(self.x)
        self.y = float(self.y)
        self.width = float(self.width)
        self.height = float(self.height)
        self.num_cols = int(self.num_cols)


@dataclass
class QpinDelay:
    name: str
    delay: float

    def __post_init__(self):
        self.delay = float(self.delay)


@dataclass
class TimingSlack:
    inst_name: str
    pin_name: str
    slack: float

    def __post_init__(self):
        self.slack = float(self.slack)


@dataclass
class GatePower:
    name: str
    power: float

    def __post_init__(self):
        self.power = float(self.power)


@dataclass
class Setting:
    alpha: float = None
    beta: float = None
    gamma: float = None
    lambde: float = None
    die_size: DieSize = None
    num_input: int = None
    inputs: List[Input] = field(default_factory=list)
    num_output: int = None
    outputs: List[Output] = field(default_factory=list)
    flip_flops: List[Flip_Flop] = field(default_factory=list)
    library: Dict = field(init=False)
    gates: List[Gate] = field(default_factory=list)
    num_instances: int = None
    instances: List[Inst] = field(default_factory=list)
    num_nets: int = None
    nets: List[Net] = field(default_factory=list)
    bin_width: float = None
    bin_height: float = None
    bin_max_util: float = None
    placement_rows: List = field(default_factory=list)
    displacement_delay: float = None
    qpin_delay: List = field(default_factory=list)
    timing_slack: List[TimingSlack] = field(default_factory=list)
    gate_power: List = field(default_factory=list)
    G: nx.Graph = field(init=False)
    __ff_templates: dict = field(init=False, repr=False)

    def convert_type(self):
        self.alpha = float(self.alpha)
        self.beta = float(self.beta)
        self.gamma = float(self.gamma)
        self.lambde = float(self.lambde)
        self.num_input = int(self.num_input)
        self.num_output = int(self.num_output)
        io_query = {input.name: input for input in self.inputs}
        io_query.update({output.name: output for output in self.outputs})
        for flip_flop in self.flip_flops:
            flip_flop.pins_query = {pin.name: pin for pin in flip_flop.pins}
        self.library = {flip_flop.name: flip_flop for flip_flop in self.flip_flops}
        for gate in self.gates:
            gate.pins_query = {pin.name: pin for pin in gate.pins}
        lib_query = {flip_flop.name: flip_flop for flip_flop in self.flip_flops}
        lib_query.update({gate.name: gate for gate in self.gates})
        self.num_instances = int(self.num_instances)
        for inst in self.instances:
            inst.lib = lib_query[inst.lib_name]
            inst.assign_pins([PhysicalPin(pin.name, inst) for pin in inst.lib.pins])
        self.__ff_templates = {ff_name: Inst(ff_name, ff_name, 0, 0) for ff_name in lib_query}
        for ff_name in self.__ff_templates:
            self.__ff_templates[ff_name].lib = lib_query[ff_name]
            self.__ff_templates[ff_name].assign_pins(
                [
                    PhysicalPin(pin.name, self.__ff_templates[ff_name])
                    for pin in lib_query[ff_name].pins
                ]
            )

        inst_query = {instance.name: instance for instance in self.instances}
        self.num_nets = int(self.num_nets)
        self.G = nx.DiGraph()
        for net in self.nets:
            pins = []
            for pin in net.pins:
                if "/" in pin.name:
                    inst_name, pin_name = pin.name.split("/")
                    inst = inst_query[inst_name]
                    pins.append(inst.pins_query[pin_name])
                else:
                    pin.inst = io_query[pin.name]
                    pins.append(pin)
            net.pins = pins

        self.bin_width = float(self.bin_width)
        self.bin_height = float(self.bin_height)
        self.bin_max_util = float(self.bin_max_util)
        self.displacement_delay = float(self.displacement_delay)
        for qpin_delay in self.qpin_delay:
            lib_query[qpin_delay.name].qpin_delay = qpin_delay.delay

        for timing_slack in self.timing_slack:
            inst_query[timing_slack.inst_name].pins_query[
                timing_slack.pin_name
            ].slack = timing_slack.slack
            # print(inst_query[timing_slack.inst_name].pins_query[timing_slack.pin_name])
        for gate_power in self.gate_power:
            lib_query[gate_power.name].power = gate_power.power

    def check_integrity(self):
        for ff in self.flip_flops:
            assert ff.qpin_delay is not None, f'library "{ff.name}" qpin_delay is not set'
        # for inst in self.instances:
        #     assert inst.lib_name in self.library

    def get_new_instance(self, lib_name):
        inst = copy.deepcopy(self.__ff_templates[lib_name])
        return inst


def read_file(input_path) -> Setting:
    setting = Setting()

    with open(input_path, "r") as file:
        library_state = 0
        for line in file.readlines():
            line = line.strip()
            line = line.lower()
            if line.startswith("#"):
                continue
            if line.startswith("alpha"):
                setting.alpha = line.split(" ")[1]
            elif line.startswith("beta"):
                setting.beta = line.split(" ")[1]
            elif line.startswith("gamma"):
                setting.gamma = line.split(" ")[1]
            elif line.startswith("lambda"):
                setting.lambde = line.split(" ")[1]
            elif line.startswith("diesize"):
                setting.die_size = DieSize(*line.split(" ")[1:])
            elif line.startswith("numinput"):
                setting.num_input = line.split(" ")[1]
            elif line.startswith("input"):
                setting.inputs.append(Input(*line.split(" ")[1:]))
            elif line.startswith("numoutput"):
                setting.num_output = line.split(" ")[1]
            elif line.startswith("output"):
                setting.outputs.append(Output(*line.split(" ")[1:]))
            elif line.startswith("flipflop") and setting.num_instances is None:
                setting.flip_flops.append(Flip_Flop(*line.split(" ")[1:]))
                library_state = 1
            elif line.startswith("gate") and setting.num_instances is None:
                setting.gates.append(Gate(*line.split(" ")[1:]))
                library_state = 2
            elif line.startswith("pin") and setting.num_instances is None:
                assert library_state == 1 or library_state == 2, library_state
                if library_state == 1:
                    setting.flip_flops[-1].pins.append(Pin(*line.split(" ")[1:]))
                elif library_state == 2:
                    setting.gates[-1].pins.append(Pin(*line.split(" ")[1:]))
            elif line.startswith("numinstances"):
                setting.num_instances = line.split(" ")[1]
            elif line.startswith("inst"):
                setting.instances.append(Inst(*line.split(" ")[1:]))
            elif line.startswith("numnets"):
                setting.num_nets = line.split(" ")[1]
            elif line.startswith("net"):
                setting.nets.append(Net(*line.split(" ")[1:]))
            elif line.startswith("pin"):
                setting.nets[-1].pins.append(PhysicalPin(line.split(" ")[1]))
            elif line.startswith("binwidth"):
                setting.bin_width = line.split(" ")[1]
            elif line.startswith("binheight"):
                setting.bin_height = line.split(" ")[1]
            elif line.startswith("binmaxutil"):
                setting.bin_max_util = line.split(" ")[1]
            elif line.startswith("placementrows"):
                setting.placement_rows.append(PlacementRows(*line.split(" ")[1:]))
            elif line.startswith("displacementdelay"):
                setting.displacement_delay = line.split(" ")[1]
            elif line.startswith("qpindelay"):
                setting.qpin_delay.append(QpinDelay(*line.split(" ")[1:]))
            elif line.startswith("timingslack"):
                setting.timing_slack.append(TimingSlack(*line.split(" ")[1:]))
            elif line.startswith("gatepower"):
                setting.gate_power.append(GatePower(*line.split(" ")[1:]))
    setting.convert_type()
    setting.check_integrity()
    return setting


def visualize(setting: Setting, resolution=None, file_name=None):
    P = PlotlyUtility(file_name=file_name if file_name else "output.html", margin=30)
    P.add_rectangle(
        BoxContainer(
            setting.die_size.xUpperRight - setting.die_size.xLowerLeft,
            setting.die_size.yUpperRight - setting.die_size.yLowerLeft,
            offset=(setting.die_size.xLowerLeft, setting.die_size.yLowerLeft),
        ).box,
        color_id="black",
        fill=False,
        group="die",
    )
    # for row in setting.placement_rows:
    #     for i in range(int(row.num_cols)):
    #         P.add_rectangle(
    #             BoxContainer(row.width, row.height, offset=(row.x + i * row.width, row.y)).box,
    #             color_id="black",
    #             fill=False,
    #             group=1,
    #             dash=True,
    #             line_width=1,
    #         )
    for input in setting.inputs:
        P.add_rectangle(
            BoxContainer(2, 0.8, offset=(input.x, input.y), centroid="c").box,
            color_id="red",
            group="input",
            text_position="top centerx",
            fill_color="red",
            text=input.name,
            show_marker=False,
        )
    for output in setting.outputs:
        P.add_rectangle(
            BoxContainer(2, 0.8, offset=(output.x, output.y), centroid="c").box,
            color_id="blue",
            group="output",
            text_position="top centerx",
            fill_color="blue",
            text=output.name,
            show_marker=False,
        )
    for inst in setting.instances:
        if inst.is_ff:
            flip_flop = inst.lib
            inst_box = BoxContainer(flip_flop.width, flip_flop.height, offset=(inst.x, inst.y))
            P.add_rectangle(
                inst_box.box,
                color_id="rgba(44, 160, 44, 0.7)",
                group="ff",
                line_color="black",
                bold=True,
                text=inst.name,
                label=inst.lib.name,
                text_position="centerxy",
                show_marker=False,
            )
            for pin in flip_flop.pins:
                pin_box = BoxContainer(0, offset=(inst.x + pin.x, inst.y + pin.y))
                P.add_rectangle(
                    pin_box.box,
                    group="ffpin",
                    text=pin.name,
                    text_location=(
                        "middle right" if pin_box.left < inst_box.centerx else "middle left"
                    ),
                    marker_size=8,
                    marker_color="rgb(255, 200, 23)",
                )
        else:
            gate = inst.lib
            inst_box = BoxContainer(gate.width, gate.height, offset=(inst.x, inst.y))
            P.add_rectangle(
                inst_box.box,
                color_id="rgba(255, 127, 14, 0.8)",
                group="gate",
                line_color="black",
                bold=True,
                text=inst.name,
                label=inst.lib.name,
                text_position="centerxy",
                show_marker=False,
            )
            for pin in gate.pins:
                pin_box = BoxContainer(0, offset=(inst.x + pin.x, inst.y + pin.y))
                P.add_rectangle(
                    pin_box.box,
                    group="gatepin",
                    text=pin.name,
                    text_location=(
                        "middle right" if pin_box.left < inst_box.centerx else "middle left"
                    ),
                    text_color="black",
                    marker_size=8,
                    marker_color="rgb(255, 200, 23)",
                )
    for net in setting.nets:
        starting_pin = net.pins[0]
        for pin in net.pins[1:]:
            P.add_line(
                start=starting_pin.pos,
                end=pin.pos,
                line_width=2,
                line_color="black",
                group="net",
                text=net.metadata,
            )
    P.show(save=True, resolution=resolution)


if __name__ == "__main__":
    from pprint import pprint

    input_path = "cases/sampleCase"
    # input_path = "cases/sample.txt"
    input_path = "v2.txt"
    setting = read_file(input_path)

    # pprint(setting)
    visualize(setting)
