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
# region: VehicleState


@dataclass
class VehicleState:
    speed: float  # Speed of vehicle in m/s
    engine_rpm: float  # Engine RPMS
    engine_stalled: bool  # Is the engine stalled?
    engine_working: bool  # Is the engine up and running?
    engine_starting: bool  # Is the engine starting as per the ignition input?
    engine_limiter: bool  # Is the rpm limiter cutting engine power?
    engine_load: float  # How much load is demanded in %
    engine_torque: float  # Torque at the engine crankshaft in Nm
    engine_power: float  # Power developed by the engine in kW
    engine_fuel_rate: float  # Instant fuel consumption in g/s
    clutch_torque: float  # Torque at the output of the clutch in Nm
    clutch_lock: float  # Lock ratio of the clutch
    gearbox_gear: int  # Engaged gear. Negative = reverse, 0 = Neutral (or Park), Positive = forward
    gearbox_mode: int  # Gearbox working mode. 0, 1, 2, 3, 4, 5 = M, P, R, N, D, L
    gearbox_shifting: bool  # Is the gearbox in the middle of a gear shift?
    retarder_torque: float  # Torque injected by the retarder in Nm
    transmission_rpm: float  # Rpms at the output of the gearbox
    abs_engaged: bool  # Is the ABS being engaged in any wheel?
    tcs_enaged: bool  # Is the TCS limiting the engine throttle?
    esc_engaged: bool  # Is the ESC applying brakes for keeping stability?
    asr_engaged: bool  # Is the ASR applying brakes for reducing wheel slip?
    aided_steer: float  # Steering input after steering aids
    fuel_consumption: float  # Overall vehicle fuel consumption in l/100km
    steering_angle: float  # Steering angle in radians
    throttle: float  # Percent throttle [0.0-1.0]
    brake: float  # Percent brake [0.0-1.0]


# end region: VehicleState
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
