"""docs/swarm_api.md against the source: every flag, camera keyword and environment switch the fork
carries has a row, and every constant the document names exists. Reads the source as text, so it
runs without a built wheel."""
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DOC = ROOT / "docs" / "swarm_api.md"
HEADER = ROOT / "src" / "SharedMemory" / "SharedMemoryPublic.h"
BINDING = ROOT / "src" / "pybullet" / "pybullet.c"
SETUP = ROOT / "setup.py"
EMBREE_SCRIPT = ROOT / "examples" / "ThirdPartyLibs" / "embree" / "build_embree.sh"
# The enums whose every value the document lists, upstream rows included, so a taken bit is visible.
ENUMS = ("EnumRendererAuxFlags", "eEnumUpdateVisualShapeFlags", "eURDF_Flags", "eUrdfCollisionFlags")
# The calls whose keyword lists the document describes.
CALLS = ("pybullet_getCameraImage", "pybullet_getDepthImagesBatch")


def enum_values(name):
  """Name to value of every member of one enum in the public header, as written there."""
  body = re.search(r"enum\s+" + name + r"\s*\{(.*?)\};", HEADER.read_text(), re.S).group(1)
  return dict(re.findall(r"^\s*(\w+)\s*=\s*([^,/]+?)\s*(?:,|$)", body, re.M))


def keyword_list(function):
  """The kwlist names of one binding function in pybullet.c."""
  text = BINDING.read_text()
  start = text.index("static PyObject* " + function + "(")
  kw = re.search(r'kwlist\[\] = \{(.*?)\}', text[start:], re.S).group(1)
  return [n for n in re.findall(r'"(\w+)"', kw)]


def documented(doc):
  """Every identifier inside a backticked span of the document, a lone name or a whole call."""
  return set(n for span in re.findall(r"`([^`]+)`", doc) for n in re.findall(r"[A-Za-z_]\w*", span))


class TestSwarmApiDoc(unittest.TestCase):
  """The reference and the source agree on names, values and the next free bit."""

  def setUp(self):
    """Reads the document once."""
    self.doc = DOC.read_text()
    self.names = documented(self.doc)

  def test_every_enum_value_has_a_row(self):
    """Each member of the four flag enums appears in the document."""
    missing = [n for e in ENUMS for n in enum_values(e) if n.lstrip("e") not in self.names and n not in self.names]
    self.assertEqual(missing, [], "flags without a row in docs/swarm_api.md: %s" % missing)

  def test_render_flag_values_match(self):
    """The value beside each render flag in the document is the value in the header."""
    for name, value in enum_values("EnumRendererAuxFlags").items():
      row = re.search(r"^\| `" + name + r"` \| (\S+) \|", self.doc, re.M)
      self.assertIsNotNone(row, name)
      self.assertEqual(row.group(1), value, name)

  def test_next_free_render_flag_is_stated(self):
    """The document names the next free render flag bit, twice the highest one in the header."""
    highest = max(int(v) for v in enum_values("EnumRendererAuxFlags").values())
    stated = set(re.findall(r"next free (?:value|bit)[^0-9]*?(\d+)", self.doc))
    self.assertEqual(stated, {str(highest * 2)}, "next free render flag bit is %d" % (highest * 2))

  def test_every_camera_keyword_has_a_row(self):
    """Each keyword of getCameraImage and getDepthImagesBatch appears in the document."""
    missing = [k for c in CALLS for k in keyword_list(c) if k not in self.names]
    self.assertEqual(missing, [], "camera keywords without a row: %s" % missing)

  def test_every_environment_switch_has_a_row(self):
    """Each SWARM_ variable read by the engine, setup.py or the Embree script appears in the document."""
    found = set()
    for source in (ROOT / "src").rglob("*.cpp"):
      found.update(re.findall(r'getenv\("(SWARM_\w+)"\)', source.read_text(errors="ignore")))
    found.update(re.findall(r"environ\.get\('(SWARM_\w+)'", SETUP.read_text()))
    found.update(re.findall(r"\$\{(SWARM_\w+)", EMBREE_SCRIPT.read_text()))
    missing = sorted(v for v in found if v not in self.names)
    self.assertEqual(missing, [], "environment switches without a row: %s" % missing)

  def test_every_constant_in_the_document_exists(self):
    """Each flag constant the document names is exported by pybullet.c or declared in the header."""
    known = set(re.findall(r'PyModule_AddIntConstant\(m, "(\w+)"', BINDING.read_text()))
    for e in ENUMS:
      known.update(n.lstrip("e") for n in enum_values(e))
    named = [n for n in self.names if re.match(r"(ER|VISUAL_SHAPE|URDF|GEOM|MJCF)_", n)]
    missing = sorted(n for n in named if n not in known)
    self.assertEqual(missing, [], "constants named in the document that do not exist: %s" % missing)


if __name__ == "__main__":
  unittest.main()
