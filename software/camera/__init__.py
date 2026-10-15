"""software.camera -- camera layer package root.

Intentionally empty: importing ``software.camera`` (or ``software.camera.contracts`` /
``software.camera.capture_io``) MUST NOT pull in any vendor-specific subpackage such as
``software.camera.x3``.  The algorithm layer is allowed to import the contract modules but is
forbidden to import ``software.camera.x3`` -- see
``software/algorithms/tests/test_dependency_boundary.py``.
"""
