# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Small G1 environment wrapper adding an interactive reset shortcut."""

from isaaclab.envs import ManagerBasedRLEnv


class G1TrackingEnv(ManagerBasedRLEnv):
    """Manager-based environment with ``R`` as a GUI-only manual reset key."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._keyboard_input = None
        self._keyboard = None
        self._keyboard_sub = None
        if self.sim.has_gui:
            import carb
            import omni.appwindow

            self._carb = carb
            app_window = omni.appwindow.get_default_app_window()
            self._keyboard_input = carb.input.acquire_input_interface()
            self._keyboard = app_window.get_keyboard()
            self._keyboard_sub = self._keyboard_input.subscribe_to_keyboard_events(
                self._keyboard, self._on_keyboard_event
            )
            self._enable_mouse_interaction()

    def _enable_mouse_interaction(self) -> None:
        """Enable Isaac Sim's Shift-drag physical rigid-body grabber."""
        import carb
        import omni.kit.app
        import omni.physx.bindings._physx as physx_bindings
        import omni.usd

        extension_manager = omni.kit.app.get_app().get_extension_manager()
        extension_manager.set_extension_enabled_immediate("omni.physx.ui", True)

        # A D6-joint grab behaves like pulling the selected link with a rope and
        # remains a physical disturbance instead of teleporting the robot.
        mouse_settings = {
            physx_bindings.SETTING_MOUSE_INTERACTION_ENABLED: True,
            physx_bindings.SETTING_MOUSE_GRAB: True,
            physx_bindings.SETTING_MOUSE_GRAB_WITH_FORCE: False,
            physx_bindings.SETTING_MOUSE_GRAB_IGNORE_INVISBLE: False,
        }
        settings = carb.settings.get_settings()
        for path, value in mouse_settings.items():
            settings.set_bool(path, value)

        # PhysX reloads per-stage settings when playback starts, so persist the
        # values on this session's anonymous root layer as well.
        stage = omni.usd.get_context().get_stage()
        if stage is not None:
            custom_data = dict(stage.GetRootLayer().customLayerData)
            physics_settings = dict(custom_data.get("physicsSettings", {}))
            physics_settings.update(mouse_settings)
            custom_data["physicsSettings"] = physics_settings
            stage.GetRootLayer().customLayerData = custom_data

    def _on_keyboard_event(self, event, *args):
        del args
        if (
            event.type == self._carb.input.KeyboardEventType.KEY_PRESS
            and event.input.name == "R"
        ):
            self.sim.request_reset()
            print("[INFO] Manual episode reset requested (R).")
        return True

    def close(self):
        if self._keyboard_sub is not None:
            self._keyboard_input.unsubscribe_to_keyboard_events(
                self._keyboard, self._keyboard_sub
            )
            self._keyboard_sub = None
        super().close()
