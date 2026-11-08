"""Extension point for hardware sensors.

Implement SensorProvider (e.g. read an ESP32/Arduino over serial, MQTT or an
HTTP API) and pass its Readings to care.build_care_report. Only ManualProvider
(values typed in by the user) ships with the project; no hardware is required.
"""
from typing import Protocol

from .schemas import Readings


class SensorProvider(Protocol):
    def read(self) -> Readings: ...


class ManualProvider:
    def __init__(self, readings: Readings):
        self._readings = readings

    def read(self) -> Readings:
        return self._readings
