# from omegaconf import OmegaConf
# from sklearn.linear_model import LinearRegression
# from gdsfactory.simulation.get_sparameters_path import get_sparameters_path_lumerical
import gdsfactory as gf
import matplotlib.pyplot as plt
import numpy as np

import sax
from sax.stax_types import StatefulModelBuilder

delta_length = 10
mzi_component = gf.components.mzi_phase_shifter_top_heater_metal(
    delta_length=delta_length
)
fig = mzi_component.plot()


def straight(wl=1.5, length=10.0, neff=2.4):
    wl0 = 1.5  # center wavelength for which the waveguide model is defined
    return sax.reciprocal({("o1", "o2"): np.exp(2j * np.pi * neff * length / wl)})


def mmi1x2():
    """Returns a perfect 1x2 splitter."""
    return sax.reciprocal(
        {
            ("o1", "o2"): 0.5**0.5,
            ("o1", "o3"): 0.5**0.5,
        }
    )


def bend_euler(wl=1.5, length=20.0):
    """Returns bend Sparameters with reduced transmission compared to a straight."""
    return {k: 0.99 * v for k, v in straight(wl=wl, length=length).items()}


class phase_shifter_heater:
    def __init__(
        self,
        wl: float = 1.55,
        neff: float = 2.34,
        length: float = 10,
        loss: float = 0.0,
        voltage=0,
        count=0,
    ):
        self.wl = wl
        self.neff = neff
        self.length = length
        self.loss = loss
        self.voltage = voltage
        self.count = count

    def __call__(self, wl=1.55, voltage=0):
        """Returns simple phase shifter model"""
        self.wl = wl
        self.voltage = voltage

        deltaphi = self.voltage * np.pi
        phase = 2 * np.pi * self.neff * self.length / self.wl + deltaphi
        amplitude = np.asarray(10 ** (-self.loss * self.length / 20), dtype=complex)
        transmission = amplitude * np.exp(1j * phase)
        sdict = sax.reciprocal(
            {
                ("o1", "o2"): transmission,
            }
        )
        self.count += 1
        return sdict


class phase_shifter_heater_empty_sig:
    def __init__(
        self,
        wl: float = 1.55,
        neff: float = 2.34,
        length: float = 10,
        loss: float = 0.0,
        voltage=0,
        count=0,
    ):
        self.wl = wl
        self.neff = neff
        self.length = length
        self.loss = loss
        self.voltage = voltage
        self.count = count

    def __call__(self):
        """Returns simple phase shifter model"""
        deltaphi = self.voltage * np.pi
        phase = 2 * np.pi * self.neff * self.length / self.wl + deltaphi
        amplitude = np.asarray(10 ** (-self.loss * self.length / 20), dtype=complex)
        transmission = amplitude * np.exp(1j * phase)
        sdict = sax.reciprocal(
            {
                ("o1", "o2"): transmission,
            }
        )
        self.count += 1
        return sdict


class phase_shifter_heater_builder(StatefulModelBuilder):
    def __init__(self):
        # state initialization
        self.count = 0

    def __call__(
        self,
        wl: float = 1.55,
        neff: float = 2.34,
        length: float = 10,
        loss: float = 0.0,
        voltage=0,
    ):
        """Returns simple phase shifter model"""

        deltaphi = voltage * np.pi
        phase = 2 * np.pi * neff * length / wl + deltaphi
        amplitude = np.asarray(10 ** (-loss * length / 20), dtype=complex)
        transmission = amplitude * np.exp(1j * phase)
        sdict = sax.reciprocal(
            {
                ("o1", "o2"): transmission,
            }
        )

        self.count += 1
        print(f"count={self.count}")

        return sdict


models = {
    "bend_euler": bend_euler,
    "mmi1x2": mmi1x2,
    "straight": straight,
    "straight_heater_metal_undercut": phase_shifter_heater_builder,
}
a = phase_shifter_heater_builder()

mzi_component = gf.components.mzi_phase_shifter_top_heater_metal(
    delta_length=delta_length,
)
# netlist = mzi_component.get_netlist(recursive=True)
netlist = mzi_component.get_netlist()
mzi_circuit, cir_info = sax.circuit(
    netlist=netlist,
    models=models,
    ignore_missing_ports=True,
)

mzi_circuit(sxt={"voltage": 0})
mzi_circuit(**{"sxt": {"voltage": 0}})

S = mzi_circuit(wl=1.55)
S

# Okay, so sax.circuit returns a flat_circuit. A flat circuit returns a _circuit function which can take as input

wl = np.linspace(1.5, 1.6, 256)
S = mzi_circuit(wl=wl)

plt.figure(figsize=(14, 4))
plt.title("MZI")
plt.plot(1e3 * wl, np.abs(S["o1", "o2"]) ** 2)  # type: ignore
plt.xlabel("λ [nm]")
plt.ylabel("T")
plt.grid(True)
plt.show()

voltages = np.linspace(-1, 1, num=5)
voltages = [-0.5, 0, 0.5]

for voltage in voltages:
    S = mzi_circuit(
        wl=wl,
        sxt={"voltage": voltage},
    )
    plt.plot(wl * 1e3, abs(S["o1", "o2"]) ** 2, label=str(voltage))  # type: ignore
    plt.xlabel("λ [nm]")
    plt.ylabel("T")
    plt.ylim(-0.05, 1.05)
    plt.grid(True)

plt.title("MZI vs voltage")
plt.legend()
