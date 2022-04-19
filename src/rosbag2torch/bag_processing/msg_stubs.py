from dataclasses import dataclass
from enum import Enum

import numpy as np

# region: VehicleInput


class ManualGear(Enum):
    MANUAL_REVERSE = -1
    MANUAL_NEURAL = 0


class AutomaticGear(Enum):
    MANUAL = 0
    PARK = 1
    REVERSE = 2
    NEUTRAL = 3
    DRIVE = 4
    DRIVE_LOW = 5


AutomaticGearDirectionDict = {
    AutomaticGear.MANUAL.value: 1.0,
    AutomaticGear.PARK.value: 0.0,
    AutomaticGear.REVERSE.value: -1.0,
    AutomaticGear.NEUTRAL.value: 0.0,
    AutomaticGear.DRIVE.value: 1.0,
    AutomaticGear.DRIVE_LOW.value: 1.0,
}


@dataclass
class VehicleInput:
    steer: float  # Steering wheel position in ratio
    throttle: float  # Throttle pedal position in ratio
    brake: float  # Brake pedal position in ratio
    handbrake: float  # Handbrake position in ratio
    clutch: float  # Clutch pedal position in ratio
    manual_gear: int  # Stick position for manual gears: -1 (R), 0 (N), 1, 2, 3... Might be used together with AutomaticGear in Manual (M) mode.
    automatic_gear: int  # Stick position for automatic transmission: 0, 1, 2, 3, 4, 5 = M, P, R, N, D, L

    def __eq__(self, __o: object) -> bool:
        if not isinstance(__o, VehicleInput):
            return False
        # Only compare steer, brake, throttle, automatic_gear since these are what MPPI changes
        fields = ["steer", "brake", "throttle", "automatic_gear"]
        return all([getattr(self, x) == getattr(__o, x) for x in fields])


# end region: VehicleInput


# region: PIDInfo


class PolarisControlMode(Enum):
    MANUAL = 0
    AUTONOMOUS = 2
    MANUAL_TAKEOVER = 3


@dataclass
class PIDInfo:
    vel_des: float
    vel: float
    error: float
    integral_error: float
    control: float
    polaris_control_mode: np.int32
    polaris_control_health: np.int32
    brake_responding: bool

    def __eq__(self, __o: object) -> bool:
        if not isinstance(__o, PIDInfo):
            return False
        fields = [
            "vel_des",
            "vel",
            "error",
            "integral_error",
            "control",
            "polaris_control_mode",
            "polaris_control_health",
            "brake_responding",
        ]
        return all([getattr(self, x) == getattr(__o, x) for x in fields])


# end region: PIDInfo
