import os
import re


def sanitize_path_component(component: str) -> str:
    if not re.match(r'^[a-zA-Z_]', component):
        return f'_{component}'
    return component


def world_path(*path: str) -> str:
    if len(path) == 1:
        path = path[0].split(os.sep)
    return os.path.join('/World', *map(sanitize_path_component, path))
