import os
import subprocess

import zed_params

def print_objects(objects: dict) -> None:
    """
    Prints detected objects (debug only).

    Args:
        objects (dict): Detected objects.

    Returns:
        None.
    """

    if __name__ != '__main__':
        return
    
    subprocess.run("cls" if os.name == "nt" else "clear", shell=True)

    for obj in objects:
        print(f"{obj['label_name']} {obj['score']}%")