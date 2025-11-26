from __future__ import annotations

import time
import typing

import attrs
import numpy as np
import omni
from isaac_utils.utils import geom
from pxr import Gf, Usd, UsdGeom
import carb

# try to import rclpy and message types for subscribing to external pose topics
try:
    # support the new topic type used by task_generator
    from arena_people_msgs.msg import Pedestrians as ArenaPedestrians
    from nav_msgs.msg import Odometry
    from people_msgs.msg import People
    from rclpy.qos import QoSProfile
    from std_msgs.msg import String as StdString
except Exception:
    rclpy = None
    People = None
    ArenaPedestrians = None
    Odometry = None
    QoSProfile = None
    StdString = None

# Prefer rclpy logger when available so messages appear in ros2/launch logs; fallback to print
try:
    from rclpy.logging import get_logger
    _LOGGER = get_logger('isaac_door_manager')
except Exception:
    _LOGGER = None


@attrs.define
class XFormAnimation:
    duration: float

    T: tuple[geom.Translation, geom.Translation] | None = None
    R: tuple[geom.Rotation, geom.Rotation] | None = None
    S: tuple[geom.Scale, geom.Scale] | None = None

    animation_fn: typing.Callable[[float], float] = lambda x: x

    _progress: float = 0.0
    _time: float = attrs.field(init=False, factory=time.time)

    def _interp_T(self, progress: float):
        if self.T is None:
            return None

        start = self.T[0].Vec3d()
        end = self.T[1].Vec3d()
        result = start + (end - start) * progress
        return geom.Translation(*result)

    def _interp_R(self, progress: float):
        if self.R is None:
            return None
        start = self.R[0].Quatd()
        end = self.R[1].Quatd()
        result = Gf.Slerp(progress, start, end)
        return geom.Rotation(*result)

    def _interp_S(self, progress: float):
        if self.S is None:
            return None
        start = self.S[0].Vec3d()
        end = self.S[1].Vec3d()
        result = start + (end - start) * progress
        return geom.Scale(*result)

    def step(self, reverse: bool = False):
        now = time.time()
        dt = now - self._time
        self._time = now
        if reverse:
            dt *= -1
        previous = self._progress
        self._progress = max(min(self._progress + dt / self.duration, 1.), 0.)
        if previous == self._progress:
            return None, None, None
        point = self.animation_fn(self._progress)
        return self._interp_T(point), self._interp_R(point), self._interp_S(point)


@attrs.define
class Door:
    prim: Usd.Prim
    T: geom.Translation
    R: geom.Rotation
    S: geom.Scale
    move_prim_path: str
    axis: np.ndarray
    angle: float
    open: bool = False
    last_toggle_time: float = 0.0
    animation: XFormAnimation | None = None


class DoorManager:
    def __init__(self):
        self._doors: dict[str, Door] = {}
        self._robots: list[str] = []
        self._pedestrians: list[str] = []
        # Distance thresholds (meters)
        self._door_open_distance = 3.0  # open when entity is closer than this
        self._door_close_margin = 0.5   # hysteresis margin; close when farther than open_distance + margin
        # Minimum seconds between toggles for a given door to avoid rapid spam
        # For instantaneous open/close set interval to 0.0
        self._door_min_toggle_interval = 0.0

        # Controller node (set by register_node)
        self._controller = None
        # Cached entity poses published over ROS topics: prim_path -> np.array([x,y,z])
        self._entity_poses: dict[str, np.ndarray] = {}
        # robot subscriptions by registered prim path
        self._robot_subs: dict[str, object] = {}
        # control verbose per-tick logging (DOOR_POS and DISTANCE). Set to False to silence.
        self._log_every_tick = False
        # Optional list of substrings to filter per-entity logs. If set, DISTANCE
        # logs will only be printed when any substring matches entity_path.
        # Example: door_manager._log_entity_filter = ['gazebo_actor']
        # or door_manager._log_entity_filter = ['jackal']
        self._log_entity_filter: list[str] | None = None
        # If True, hide door geometry by setting visibility instead of translating/scale
        # Useful for debugging and avoids moving shared ancestors (walls).
        self._use_visibility_toggle: bool = True

        self._debug_rate_limit = 1.0
        self._last_debug_time = 0.0

    def _rate_limited_debug(self, msg: str):
        try:
            now = time.time()
            if now - self._last_debug_time >= float(self._debug_rate_limit):
                carb.log_verbose(msg)
                self._last_debug_time = now
        except Exception:
            carb.log_verbose(msg)

    def register_node(self, controller):
        """Attach rclpy subscriptions to the provided controller node so DoorManager
        receives live poses for pedestrians and robots published by the task_generator.
        """
        self._controller = controller
        # subscribe to legacy people topic if available
        if People is not None:
            try:
                qos_depth = 10
                controller.create_subscription(People, '/task_generator_node/people', self._people_cb, qos_depth)
                carb.log_verbose('Subscribed to /task_generator_node/people for pedestrian poses')
            except Exception as e:
                carb.log_warn(f'Failed to subscribe to people topic: {e}')
        else:
            carb.log_warn('people_msgs.People message type not available; pedestrian topic not subscribed')

        # subscribe to new arena people topic if available
        if ArenaPedestrians is not None:
            try:
                controller.create_subscription(ArenaPedestrians, '/task_generator_node/arena_peds', self._people_cb, 10)
                carb.log_verbose('Subscribed to /task_generator_node/arena_peds for pedestrian poses')
            except Exception as e:
                carb.log_warn(f'Failed to subscribe to arena_peds topic: {e}')
        else:
            carb.log_verbose('arena_people_msgs.Pedestrians not available; /task_generator_node/arena_peds not subscribed')

        # Also subscribe to simple registration topic so external processes can register prims
        if StdString is not None:
            try:
                controller.create_subscription(StdString, '/isaac/register_entity', self._register_entity_cb, 10)
                carb.log_verbose('Subscribed to /isaac/register_entity for external entity registrations')
            except Exception as e:
                carb.log_warn(f'Failed to subscribe to /isaac/register_entity: {e}')
        else:
            carb.log_warn('std_msgs.String not available; registration topic not subscribed')

        # ensure robot subscriptions for already-registered robots
        for prim_path in list(self._robots):
            try:
                self._ensure_robot_subscription(prim_path)
            except Exception as e:
                carb.log_warn(f'Failed to ensure robot subscription for {prim_path}: {e}')

    def _ensure_robot_subscription(self, prim_path: str):
        if self._controller is None:
            return
        if prim_path in self._robot_subs:
            return
        # derive robot name from prim path: /World/<robot_name>/...
        try:
            parts = prim_path.split('/')
            if len(parts) >= 3 and parts[1] == 'World':
                robot_name = parts[2]
                topic = f'/task_generator_node/{robot_name}/odom'
                if Odometry is not None:
                    try:
                        sub = self._controller.create_subscription(
                            Odometry,
                            topic,
                            lambda msg, p=prim_path: self._odom_cb(msg, p),
                            10
                        )
                        self._robot_subs[prim_path] = sub
                        carb.log_verbose(f'Subscribed to {topic} for robot {robot_name}')
                    except Exception as e:
                        carb.log_warn(f'Failed to subscribe to {topic}: {e}')
                else:
                    carb.log_warn('nav_msgs.Odometry not available; robot odom not subscribed')
        except Exception as e:
            carb.log_verbose(f'_ensure_robot_subscription error: {e}')

    def _odom_cb(self, msg, prim_path: str):
        try:
            pose = getattr(msg, 'pose', None)
            if pose is None:
                return
            pb = getattr(pose, 'pose', pose)  # handle Odometry vs nested
            pos = getattr(pb, 'position', None)
            if pos is None:
                return
            self._entity_poses[prim_path] = np.array([pos.x, pos.y, pos.z])
            carb.log_verbose(f'odom update for {prim_path}: {self._entity_poses[prim_path]}')
        except Exception as e:
            carb.log_verbose(f'odom_cb error: {e}')

    def _people_cb(self, msg):
        """Unified people callback supporting multiple message types.

        Handles:
          - people_msgs/People (msg.people list)
          - arena_people_msgs/Pedestrians (msg.pedestrians list with nested position.position)
        """
        try:
            carb.log_verbose('people topic message received')
            # support both attribute names
            people = getattr(msg, 'people', None) or getattr(msg, 'pedestrians', None)
            if not people:
                return

            def _extract_xyz_from_field(field) -> np.ndarray | None:
                if field is None:
                    return None
                # common direct x,y,z
                if hasattr(field, 'x') and hasattr(field, 'y') and hasattr(field, 'z'):
                    return np.array([float(getattr(field, 'x', 0.0)),
                                     float(getattr(field, 'y', 0.0)),
                                     float(getattr(field, 'z', 0.0))])
                # nested .position (e.g. arena_people_msgs -> position.position)
                inner = getattr(field, 'position', None)
                if inner and hasattr(inner, 'x'):
                    return np.array([float(getattr(inner, 'x', 0.0)),
                                     float(getattr(inner, 'y', 0.0)),
                                     float(getattr(inner, 'z', 0.0))])
                # nested .pose.position
                pose = getattr(field, 'pose', None)
                if pose:
                    inner2 = getattr(pose, 'position', None)
                    if inner2 and hasattr(inner2, 'x'):
                        return np.array([float(getattr(inner2, 'x', 0.0)),
                                         float(getattr(inner2, 'y', 0.0)),
                                         float(getattr(inner2, 'z', 0.0))])
                return None

            for p in people:
                # identifier: prefer stage_prefix, then name, then id
                stage_prefix = getattr(p, 'stage_prefix', None) or getattr(p, 'name', None) or getattr(p, 'id', None)

                # fields vary: try common ones
                pose_field = getattr(p, 'pose', None) or getattr(p, 'position', None)
                xyz = _extract_xyz_from_field(pose_field)
                if xyz is None:
                    # some messages have nested structure: p.position.position
                    alt = getattr(p, 'position', None)
                    xyz = _extract_xyz_from_field(alt)

                if xyz is None:
                    carb.log_verbose(f'Could not extract position for pedestrian entry: {stage_prefix}')
                    continue

                # Map to a registered pedestrian prim path if any contains the name/id
                matched_prim = None
                if stage_prefix is not None:
                    for reg in self._pedestrians:
                        if stage_prefix in reg or reg in str(stage_prefix):
                            matched_prim = reg
                            break
                # fallback to use stage_prefix as prim path if it looks like a usd path
                if matched_prim is None and isinstance(stage_prefix, str) and stage_prefix.startswith('/'):
                    matched_prim = stage_prefix

                # if still None, use a name-based synthetic key so distance logic can still see it
                prim_key = matched_prim or f"ped|{stage_prefix}"

                self._entity_poses[prim_key] = xyz
                carb.log_verbose(f'people update for {prim_key}: {xyz}')

        except Exception as e:
            carb.log_verbose(f'people_cb error: {e}')

    def register_door(self, prim_path: str, kind: str, start: Gf.Vec3f, end: Gf.Vec3f):
        stage = omni.usd.get_context().get_stage()
        prim = stage.GetPrimAtPath(prim_path)
        if prim.IsValid():
            start_v = start
            end_v = end
            dx = end_v[0] - start_v[0]
            dy = end_v[1] - start_v[1]
            axis = np.array([dx, dy, 0])
            axis = axis / (np.linalg.norm(axis) or 1)
            angle = np.arctan2(axis[1], axis[0]) + np.pi / 2

            animation = None

            door = Door(
                prim=prim,
                move_prim_path=prim.GetPath().pathString,
                T=geom.Translation.parse(prim.GetAttribute('xformOp:translate').Get() or (0, 0, 0)),
                R=geom.Rotation.parse(prim.GetAttribute('xformOp:rotateXYZ').Get() or (0, 0, 0)),
                S=geom.Scale.parse(prim.GetAttribute('xformOp:scale').Get() or (1, 1, 1)),
                open=False,
                last_toggle_time=0.0,
                axis=axis,
                angle=angle,
                animation=None
            )

            if kind == 'sliding':
                door.animation = XFormAnimation(
                    duration=3.0,
                    T=(
                        geom.Translation(0., 0., 0.),
                        geom.Translation(axis[0] * door.S.x, axis[1] * door.S.x, 0.)
                    ),
                )
            elif kind == 'sliding_top':
                door.animation = XFormAnimation(
                    duration=3.0,
                    T=(
                        geom.Translation(0., 0., 0.),
                        geom.Translation(0., 0., door.S.z)
                    ),
                )

            self._doors[prim_path] = door
            carb.log_verbose(f"Added door to door manager: {prim_path}")
        else:
            carb.log_warn(f"Failed to add door - invalid prim: {prim_path}")

    def reset(self):
        self._pedestrians.clear()
        # Close all doors on reset
        for door_path, door in self._doors.items():
            self._close_door(door)

    def update(self):
        stage = omni.usd.get_context().get_stage()
        entities_to_check = self._robots + self._pedestrians

        if not self._doors:
            carb.log_verbose("No doors registered with DoorManager.")

        # Always print something each tick. If there are no entities, still print door positions.
        for door_path, door in self._doors.items():
            door_prim = door.prim
            if not door_prim.IsValid():
                carb.log_warn(f"Door prim invalid: {door_path}")
                continue

            door_pos = door.T.Vec3d()
            if self._log_every_tick:
                self._rate_limited_debug(f"[DOOR_POS] {door_path} -> {door_pos.tolist()}")

            if entities_to_check:
                # Compute positions for all entities and their distances to the door
                distances = []
                entity_positions = {}
                for entity_path in entities_to_check:
                    try:
                        entity_prim = None
                        if entity_path in self._entity_poses:
                            entity_pos = self._entity_poses[entity_path]
                        else:
                            entity_prim = stage.GetPrimAtPath(entity_path)
                            if not (entity_prim and entity_prim.IsValid()):
                                resolved_path = self._resolve_entity_prim(entity_path)
                                entity_prim = stage.GetPrimAtPath(resolved_path)
                                if entity_prim and entity_prim.IsValid():
                                    carb.log_verbose(f"Resolved entity prim for distance checks: {entity_path} -> {resolved_path}")
                                    entity_path = resolved_path
                        if not (entity_prim and entity_prim.IsValid()):
                            continue
                        entity_pos = self._get_prim_position(entity_prim)
                        dist = float(np.linalg.norm(door_pos - entity_pos))
                        distances.append((dist, entity_path))
                        entity_positions[entity_path] = entity_pos
                    except Exception as e:
                        carb.log_verbose(f'Error computing distance for {entity_path}: {e}')

                if not distances:
                    if self._log_every_tick:
                        carb.log_verbose(f"DISTANCE: Door {door_path} -> (no valid entities found)")
                    continue

                # Use the minimum distance across all entities to decide open/close
                distances.sort(key=lambda x: x[0])
                min_distance, closest_entity = distances[0]

                # Optionally print per-entity distance for the closest entity only
                if self._log_every_tick:
                    if (self._log_entity_filter is None or any(substr in closest_entity for substr in self._log_entity_filter)):
                        carb.log_verbose(f"DISTANCE: Door {door_path} -> Entity {closest_entity} = {min_distance:.3f} m (open={door.open})")

                # Hysteresis + cooldown: decide once per door using min_distance
                now = time.time()
                if min_distance < self._door_open_distance:
                    if (not door.open and (now - door.last_toggle_time) > self._door_min_toggle_interval):
                        carb.log_verbose(f"Opening door {door_path} (closest entity {closest_entity} within {self._door_open_distance}m)")
                        self._open_door(door)
                elif min_distance > (self._door_open_distance + self._door_close_margin):
                    if (door.open and (now - door.last_toggle_time) > self._door_min_toggle_interval):
                        carb.log_verbose(f"Closing door {door_path} (no entities within {self._door_open_distance}m)")
                        self._close_door(door)
                door.last_toggle_time = now
            else:
                # No entities registered — optionally print a placeholder distance message
                if self._log_every_tick:
                    carb.log_verbose(f"DISTANCE: Door {door_path} -> (no entities registered)")

        for door_path, door in self._doors.items():
            if not door.animation:
                continue
            anim = door.animation
            dT, dR, dS = anim.step(not door.open)
            prim = door.move_prim_path
            if dT:
                geom.move(prim, translation=door.T + dT, local=True)
            if dR:
                geom.move(prim, rotation=door.R * dR, local=True)
            if dS:
                geom.rescale(prim, scale=door.S + dS)

    def _get_prim_position(self, prim):
        try:
            xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
            matrix = xform_cache.GetLocalToWorldTransform(prim)
            tx = matrix[3][0]
            ty = matrix[3][1]
            tz = matrix[3][2]
            return np.array([tx, ty, tz])
        except Exception:
            xformable = UsdGeom.Xformable(prim)
            translation = xformable.GetLocalTransformation().GetRow(3)
            return np.array([translation[0], translation[1], translation[2]])

    def _dump_prim_transform_info(self, prim):
        """Diagnostic: log transform ops and attributes for prim, parents and children."""
        try:
            if not prim or not prim.IsValid():
                carb.log_verbose('Diagnostic: prim invalid or not found')
                return
            carb.log_verbose(f'Diagnostic for prim: {prim.GetPath().pathString}')
            try:
                carb.log_verbose(f'  typeName={prim.GetTypeName()}, IsInstance={prim.IsInstance()}')
            except Exception:
                pass
            # Xform ops for the prim
            try:
                xf = UsdGeom.Xformable(prim)
                ops = xf.GetOrderedXformOps()
                carb.log_verbose('  ordered xform ops: ' + str([o.GetOpName() for o in ops]))
            except Exception as e:
                carb.log_verbose(f'  prim not xformable: {e}')

            # common transform attributes
            for name in ('xformOp:translate', 'xformOp:scale', 'xformOp:transform'):
                try:
                    a = prim.GetAttribute(name)
                    if a and a.HasAuthoredValue():
                        carb.log_verbose(f'  {name} = {a.Get()}')
                    else:
                        carb.log_verbose(f'  {name} = <not authored>')
                except Exception as e:
                    carb.log_verbose(f'  reading {name} failed: {e}')

            # walk parents up to a few levels
            parent = prim.GetParent()
            depth = 0
            while parent and parent.IsValid() and depth < 5:
                try:
                    carb.log_verbose(f'  parent: {parent.GetPath().pathString} type={parent.GetTypeName()} IsInstance={parent.IsInstance()}')
                    try:
                        pxf = UsdGeom.Xformable(parent)
                        pops = pxf.GetOrderedXformOps()
                        carb.log_verbose('    ordered xform ops: ' + str([o.GetOpName() for o in pops]))
                    except Exception:
                        carb.log_verbose('    parent not xformable')
                    for name in ('xformOp:translate', 'xformOp:scale', 'xformOp:transform'):
                        try:
                            a = parent.GetAttribute(name)
                            if a and a.HasAuthoredValue():
                                carb.log_verbose(f'    {name} = {a.Get()}')
                            else:
                                carb.log_verbose(f'    {name} = <not authored>')
                        except Exception:
                            pass
                except Exception as e:
                    carb.log_verbose(f'  parent info failed: {e}')
                parent = parent.GetParent()
                depth += 1

            # list children attributes
            for child in prim.GetAllChildren():
                try:
                    carb.log_verbose(f'  child: {child.GetPath().pathString} type={child.GetTypeName()} IsInstance={child.IsInstance()}')
                    try:
                        cxf = UsdGeom.Xformable(child)
                        cops = cxf.GetOrderedXformOps()
                        carb.log_verbose('    ordered xform ops: ' + str([o.GetOpName() for o in cops]))
                    except Exception:
                        carb.log_verbose('    child not xformable')
                    for name in ('xformOp:translate', 'xformOp:scale', 'xformOp:transform'):
                        try:
                            a = child.GetAttribute(name)
                            if a and a.HasAuthoredValue():
                                carb.log_verbose(f'    {name} = {a.Get()}')
                            else:
                                carb.log_verbose(f'    {name} = <not authored>')
                        except Exception:
                            pass
                except Exception as e:
                    carb.log_verbose(f'  child info failed: {e}')
        except Exception as e:
            carb.log_verbose(f'_dump_prim_transform_info error: {e}')

    def _set_visibility(self, prim, visible: bool, recursive: bool = False):
        """Set Usd visibility on prim. visible=True -> 'inherited', False -> 'invisible'.
        If recursive=True, apply to all descendants as well.
        """
        try:
            if prim is None or not prim.IsValid():
                return False
            try:
                img = UsdGeom.Imageable(prim)
            except Exception:
                img = None
            if img is not None:
                val = 'inherited' if visible else 'invisible'
                try:
                    img.GetVisibilityAttr().Set(val)
                except Exception as e:
                    carb.log_verbose(f'Failed to set visibility on {prim.GetPath().pathString}: {e}')
            # optionally apply to children
            if recursive:
                for child in prim.GetAllChildren():
                    try:
                        self._set_visibility(child, visible, recursive=True)
                    except Exception:
                        pass
            carb.log_verbose(f"Set visibility {val} on {prim.GetPath().pathString}")
            return True
        except Exception as e:
            carb.log_verbose(f'_set_visibility error: {e}')
            return False

    def _open_door(self, door: Door):
        if not door.animation:
            self._set_visibility(door.prim, visible=False, recursive=True)
        door.open = True

    def _close_door(self, door: Door):
        if not door.animation:
            self._set_visibility(door.prim, visible=True, recursive=True)
        door.open = False

    def _resolve_entity_prim(self, prim_path: str) -> str:
        """Try to resolve a dynamic/moving sub-prim for a given USD prim path.
        If the given prim is valid return it, otherwise attempt common suffixes
        and a shallow child-name search. Always return a string path (fallback
        to the original prim_path).
        """
        try:
            stage = omni.usd.get_context().get_stage()
            prim = stage.GetPrimAtPath(prim_path)
            if prim and prim.IsValid():
                return prim_path

            # Common candidate suffixes for robots/actors
            candidates = [
                'base_link',
                'base_footprint',
                'base',
                'man_root',
                'ManRoot',
                'root',
                'actor',
            ]
            for c in candidates:
                p = prim_path.rstrip('/') + '/' + c
                pr = stage.GetPrimAtPath(p)
                if pr and pr.IsValid():
                    return p

            # Shallow search children for likely moving prims
            if prim and prim.IsValid():
                for child in prim.GetChildren():
                    name = child.GetName().lower()
                    if any(k in name for k in ('base', 'root', 'man', 'actor')):
                        return child.GetPath().pathString
        except Exception as e:
            carb.log_verbose(f'_resolve_entity_prim error: {e}')
        return prim_path

    def add_robot(self, prim_path: str):
        """Register a robot prim for distance checks. Resolves to a moving
        sub-prim if possible and ensures an odom subscription is created when
        a controller node is registered.
        """
        resolved = self._resolve_entity_prim(prim_path)
        if resolved not in self._robots:
            self._robots.append(resolved)
        carb.log_verbose(f"Added robot to door manager: {prim_path} -> resolved: {resolved}")
        try:
            self._ensure_robot_subscription(resolved)
        except Exception as e:
            carb.log_warn(f'Failed to ensure robot subscription on add: {e}')

    def add_pedestrian(self, prim_path: str):
        """Register a pedestrian prim for distance checks. Poses for
        pedestrians are primarily updated via the people topic; this method
        just records the prim path.
        """
        resolved = self._resolve_entity_prim(prim_path)
        if resolved not in self._pedestrians:
            self._pedestrians.append(resolved)
        carb.log_verbose(f"Added pedestrian to door manager: {prim_path} -> resolved: {resolved}")

    def _register_entity_cb(self, msg):
        """Handle simple registration messages published on /isaac/register_entity.
        Expected payload: '<role>|<prim_path>' where role is 'robot' or 'pedestrian'.
        """
        try:
            data = getattr(msg, 'data', '')
            carb.log_verbose(f'REGISTER_ENTITY received: {data}')
            if not data:
                return
            parts = data.split('|', 1)
            if len(parts) != 2:
                carb.log_warn(f'Invalid register_entity payload: {data}')
                return
            role, prim_path = parts[0], parts[1]
            role = role.strip().lower()
            prim_path = prim_path.strip()
            if role == 'robot':
                self.add_robot(prim_path)
                carb.log_verbose(f'Registered robot: {prim_path}')
            elif role == 'pedestrian' or role == 'ped':
                self.add_pedestrian(prim_path)
                carb.log_verbose(f'Registered pedestrian: {prim_path}')
            else:
                carb.log_warn(f'Unknown role in register_entity: {role}')
            # show current registry
            carb.log_verbose(f'Current robots: {self._robots}, pedestrians: {self._pedestrians}')
        except Exception as e:
            carb.log_verbose(f'_register_entity_cb error: {e}')


door_manager = DoorManager()
