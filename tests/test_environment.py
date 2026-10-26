import unittest
from software.environment import python_executable


class InterpreterSelectionTests(unittest.TestCase):
    def test_new_environment_takes_priority_and_old_installations_still_work(self):
        self.assertEqual(python_executable({'UMI_PY': 'new', 'KIWI_PY': 'old'}), 'new')
        self.assertEqual(python_executable({'KIWI_PY': 'old'}), 'old')
        self.assertEqual(python_executable({}, default='fallback'), 'fallback')
        self.assertEqual(python_executable({'UMI_PY_TOOLS': 'tools', 'KIWI_PY_TOOLS': 'oldtools', 'UMI_PY': 'base'}, tools=True), 'tools')
        with self.assertRaisesRegex(ValueError, 'UMI_PY'):
            python_executable({})
