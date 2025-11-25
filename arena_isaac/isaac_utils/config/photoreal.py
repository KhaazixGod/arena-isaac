

import typing

import omni.kit.actions.core
from omni.kit.viewport.utility import get_active_viewport
import carb


class RenderSettings(typing.NamedTuple):
    class ViewportRender(typing.NamedTuple):
        resolution: typing.Literal['dynamic'] | tuple[int, int] = (1280, 720)
        scale: float = 1.0

        def apply(self):
            viewport_api = get_active_viewport()

            if self.resolution == 'dynamic':
                viewport_api.fill_frame = True
            else:
                viewport_api.resolution = self.resolution
            viewport_api.render_scale = self.scale

    class ViewportDisplay(typing.NamedTuple):
        axis: bool = True
        grid: bool = True
        bbox: bool = True

        def apply(self):

            action_registry = omni.kit.actions.core.get_action_registry()
            viewport_api = get_active_viewport()

            # Viewport Display settings
            action_registry.get_action("omni.kit.viewport.actions", "toggle_grid_visibility").execute(viewport_api=viewport_api, visible=self.grid)
            action_registry.get_action("omni.kit.viewport.actions", "toggle_axis_visibility").execute(viewport_api=viewport_api, visible=self.axis)
            action_registry.get_action("omni.kit.viewport.actions", "toggle_bounding_box_visibility").execute(viewport_api=viewport_api, visible=self.bbox)

    class RayTracing(typing.NamedTuple):
        DLSS: typing.Literal['Auto', 'Quality', 'Balanced', 'Performance'] = 'Auto'

        def apply(self):
            settings = carb.settings.get_settings()
            settings.set_int("/rtx/post/dlss/execMode", {'Auto': 3, 'Quality': 2, 'Balanced': 1, 'Performance': 0}[self.DLSS])

    class Lighting(typing.NamedTuple):
        preset: typing.Literal['Lights Off', 'Camera Light', 'Stage Lights', 'Colored Lights', 'Default', 'Grey Studio'] = 'Stage Lights'

        def apply(self):
            action_registry = omni.kit.actions.core.get_action_registry()
            action_registry.get_action("omni.kit.viewport.menubar.lighting", "set_lighting_mode_preset").execute(preset_name=self.preset)

    class PostProcessing(typing.NamedTuple):
        ...
        # TODO

        def apply(self):
            ...
            # TODO

    render: ViewportRender = ViewportRender()
    lighting: Lighting = Lighting()
    display: ViewportDisplay = ViewportDisplay()
    ray_tracing: RayTracing = RayTracing()
    post_processing: PostProcessing = PostProcessing()

    def apply(self):
        # Viewport Display settings
        try:
            self.display.apply()
        except Exception as e:
            carb.log_warn(f"[RenderSettings] Failed to apply viewport display settings: {e}")

        # Viewport Render settings
        try:
            self.render.apply()
        except Exception as e:
            carb.log_warn(f"[RenderSettings] Failed to apply viewport render settings: {e}")

        # Ray Tracing settings
        try:
            self.ray_tracing.apply()
        except Exception as e:
            carb.log_warn(f"[RenderSettings] Failed to apply ray tracing settings: {e}")

        # Lighting settings
        try:
            self.lighting.apply()
        except Exception as e:
            carb.log_warn(f"[RenderSettings] Failed to apply lighting settings: {e}")

        # Post Processing settings
        try:
            self.post_processing.apply()
        except Exception as e:
            carb.log_warn(f"[RenderSettings] Failed to apply post processing settings: {e}")


PRESET_DEFAULT: RenderSettings = RenderSettings()

PRESET_PHOTOREAL: RenderSettings = RenderSettings(
    render=RenderSettings.ViewportRender(
        resolution='dynamic',
        scale=1.0,
    ),
    lighting=RenderSettings.Lighting(
        preset='Default',
    ),
    display=RenderSettings.ViewportDisplay(
        axis=False,
        grid=False,
        bbox=False,
    ),
    ray_tracing=RenderSettings.RayTracing(
        DLSS='Quality',
    ),
    post_processing=RenderSettings.PostProcessing()
)
