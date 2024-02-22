

import logic_gate as lg
import sys

logic_gate = lg.LogicGate()

logic_gate.and_gate(1, 0)
logic_gate.print_output('AND')

logic_gate.or_gate(1, 0)
logic_gate.print_output('OR')

logic_gate.Nand_gate(1, 0)
logic_gate.print_output('NAND')

logic_gate.Nor_gate(1, 0)
logic_gate.print_output('NOR')
