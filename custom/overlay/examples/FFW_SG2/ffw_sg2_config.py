from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)


ffw_sg2_config = {
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=[
            "cam_head",
            "cam_wrist_left",
            "cam_wrist_right",
        ],
    ),
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=[
            "arm_left",
            "gripper_left",
            "arm_right",
            "gripper_right",
            "head",
            "lift",
            "mobile_base",
        ],
    ),
    "action": ModalityConfig(
        delta_indices=list(range(0, 16)),  # 16-step action horizon
        modality_keys=[
            "arm_left",
            "gripper_left",
            "arm_right",
            "gripper_right",
            "head",
            "lift",
            "mobile_base",
        ],
        action_configs=[
            # arm_left (7 joints) - RELATIVE for smooth motion
            ActionConfig(
                rep=ActionRepresentation.RELATIVE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            # gripper_left (1 joint) - ABSOLUTE for precise control
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            # arm_right (7 joints) - RELATIVE for smooth motion
            ActionConfig(
                rep=ActionRepresentation.RELATIVE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            # gripper_right (1 joint) - ABSOLUTE for precise control
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            # head (2 joints) - RELATIVE
            ActionConfig(
                rep=ActionRepresentation.RELATIVE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            # lift (1 joint) - RELATIVE
            ActionConfig(
                rep=ActionRepresentation.RELATIVE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            # mobile_base (3 values: linear_x, linear_y, angular_z) - ABSOLUTE for velocity control
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
        ],
    ),
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.human.task_description"],
    ),
}

register_modality_config(ffw_sg2_config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
