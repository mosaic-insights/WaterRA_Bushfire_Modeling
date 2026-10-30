"""
The names eWater Source keeps for a data source, and the ones it changes.

Kept apart from fire_impacts.source so study.toml can be checked against the
rule when it loads, without importing veneer-py.
"""

from typing import Optional

# Characters the Source interface replaces with '_' in a data source's name.
RESERVED_CHARACTERS = '.@>,\\'


def restricted_data_source_name(name: str) -> str:
    """
    Return the name the Source interface gives a data source called `name`.

    Parameters:
    - name: A data source name, as given to Veneer.

    Returns:
    - The name with '.', '@', '>', ',' and backslash each replaced by an
      underscore, cut to 200 characters, and stripped of surrounding
      spaces.
    ------------------------------------------------------------------------
    Notes:
    - This mirrors Source's DataManager.RestrictName. Source applies it
      when its interface shows a data source, and the new name sticks; a
      Veneer command line keeps the name as given. veneer-py 0.3.0 and
      later apply it to every name they give a data source, so a name it
      changes is not the one asked for, with or without the interface.
    ------------------------------------------------------------------------
    """
    for reserved in RESERVED_CHARACTERS:
        name = name.replace(reserved, '_')
    return name[:200].strip(' ')


def data_source_name_problem(name: str) -> Optional[str]:
    """
    Say why Source would not keep `name` for a data source, or return None.

    The message names the name Source would use instead, and suggests it.
    """
    restricted = restricted_data_source_name(name)
    if restricted == name:
        return None
    return (
        f"Source would rename data source '{name}' to '{restricted}' "
        f"when its interface is open, and then not find it under the "
        f"name asked for. Use a name without '.', '@', '>', ',' or "
        f"backslashes, such as '{restricted}'.")
