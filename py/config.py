from pathlib import Path

import yaml
from cerberus import Validator

zero_schema = dict(
    setpoint={"type": "float", "coerce": float},
    ts={"type": "float", "coerce": float},
)

pid_schema = dict(
    kp={"type": "float", "coerce": float},
    ti={"type": "float", "coerce": float},
    td={"type": "float", "coerce": float},
    minvalue={"type": "float", "coerce": float},
    maxvalue={"type": "float", "coerce": float},
)
pid_schema.update(zero_schema)

ift_schema = dict(
    stage_length={"type": "float", "coerce": float},
    gamma={"type": "float", "coerce": float},
    lam={"type": "float", "coerce": float},
    min_kp={"type": "float", "coerce": float},
    min_ti={"type": "float", "coerce": float},
)
ift_schema.update(pid_schema)
del ift_schema["td"]


class Config(object):
    schema = dict(
        Controller={
            "type": "string",
            "coerce": (str, lambda x: x.upper()),
            "default": "ZERO",
            "allowed": ("ZERO", "PID", "IFT"),
        },
        Modulation={
            "type": "string",
            "coerce": (str, lambda x: x.lower()),
            "default": "none",
            "allowed": ("amplitude", "frequency", "none", ""),
        },
        RandomSeed={"type": "integer", "coerce": int, "default": 3695},
        TimeStep={"type": "float", "coerce": float, "default": 0.01},
        SteadyStateDuration={"type": "float", "coerce": float, "default": 6000.0},
        RunTime={"type": "float", "coerce": float, "default": 32000.0},
        setpoint={"type": "float", "coerce": float, "default": 0},
        kp={"type": "float", "coerce": float, "default": 0.23},
        ti={"type": "float", "coerce": float, "default": 0.2},
        td={"type": "float", "coerce": float, "default": 0},
        ts={"type": "float", "coerce": float, "default": 0},
        minvalue={"type": "float", "coerce": float, "default": 0},
        maxvalue={"type": "float", "coerce": float, "default": 3},
        stage_length={"type": "float", "coerce": float, "default": 0},
        gamma={"type": "float", "coerce": float, "default": 0.01},
        lam={"type": "float", "coerce": float, "default": 1e-8},
        min_kp={"type": "float", "coerce": float, "default": 0.01},
        min_ti={"type": "float", "coerce": float, "default": 0.01},
    )

    def __init__(self, config_file):

        if config_file:
            self.config_file = Path(config_file).resolve()
            with self.config_file.open("r") as f:
                conf = yaml.safe_load(f)
        else:
            self.config_file = None
            conf = {}

        v = Validator(require_all=False)

        if not v.validate(conf, self.schema):
            raise RuntimeError(f"Invalid configuration file:\n{v.errors}")

        for key, value in v.document.items():
            self.__setattr__(key, value)

    def __str__(self):
        return str(vars(self)).strip("{}").replace(", ", ",\n")


def controller_interface(config):
    v = Validator(require_all=True, purge_unknown=True)
    controller = config.Controller
    if controller == "ZERO":
        schema = zero_schema
    elif controller == "PID":
        schema = pid_schema
    elif controller == "IFT":
        schema = ift_schema
    else:
        raise RuntimeError("Invalid controller type")

    if not v.validate(vars(config), schema):
        raise RuntimeError(f"Invalid controller options:\n{v.errors}")

    return v.document
