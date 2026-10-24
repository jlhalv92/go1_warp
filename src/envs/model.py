"""Load the supplied model, optionally stripping visual meshes for debugging."""
from pathlib import Path
import xml.etree.ElementTree as ET
import mujoco
from src import ROOT


def load_model(xml_path=None, collision_only=False):
    path = Path(xml_path) if xml_path else Path(__file__).resolve().parents[1] / 'models' / 'walk_go1.xml'
    path = (path if path.is_absolute() else ROOT / path).resolve()
    if not collision_only:
        return mujoco.MjModel.from_xml_path(str(path))
    # Resolve nested includes relative to their containing file before stripping meshes.
    def expand(filename):
        root = ET.parse(filename).getroot()
        def visit(parent):
            for child in list(parent):
                if child.tag == 'include':
                    index = list(parent).index(child)
                    parent.remove(child)
                    for item in list(expand(filename.parent / child.attrib['file'])):
                        parent.insert(index, item)
                        index += 1
                else:
                    visit(child)
        visit(root)
        return root
    root = expand(path)
    for parent in root.iter():
        for child in list(parent):
            if child.tag == 'mesh' or (child.tag == 'geom' and 'mesh' in child.attrib):
                parent.remove(child)
    return mujoco.MjModel.from_xml_string(ET.tostring(root, encoding='unicode'))
