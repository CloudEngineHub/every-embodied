"""Acceptance checks for malformed motion exports and valid alternate schemas."""
import io
import unittest

import numpy as np

from inspect_motion import inspect


def raw(**changes):
    arrays = dict(root_trans=np.zeros((4, 3)),
                  root_rot_quat=np.tile([1., 0., 0., 0.], (4, 1)),
                  dof=np.zeros((4, 29)), fps=np.array([30]))
    arrays.update(changes)
    stream = io.BytesIO()
    np.savez_compressed(stream, **arrays)
    stream.seek(0)
    return inspect("sample.npz", stream)


class MotionAcceptanceTests(unittest.TestCase):
    def test_valid_raw(self):
        result = raw()
        self.assertTrue(result["passed"])
        self.assertEqual((result["frames"], result["dofs"]), (4, 29))

    def test_invalid_predictions_are_rejected(self):
        for changes in (
            {"dof": np.full((4, 29), np.nan)},
            {"root_trans": np.zeros((3, 3))},
            {"root_rot_6d": np.zeros((3, 6))},
            {"root_rot_quat": np.zeros((4, 4))},
            {"fps": np.array([0])},
            {"fps": np.array([np.inf])},
            {"dof": np.zeros((0, 29))},
        ):
            with self.subTest(changes=list(changes)):
                self.assertFalse(raw(**changes)["passed"])

    def test_valid_body_export(self):
        stream = io.BytesIO()
        np.savez_compressed(stream, joint_pos=np.zeros((4, 29)),
                            body_pos_w=np.zeros((4, 32, 3)),
                            body_quat_w=np.tile([1., 0., 0., 0.], (4, 32, 1)), fps=30)
        stream.seek(0)
        self.assertTrue(inspect("body.npz", stream)["passed"])

    def test_contact_only_is_not_motion(self):
        stream = io.BytesIO()
        np.savez_compressed(stream, contact=np.zeros((4, 2)))
        stream.seek(0)
        self.assertFalse(inspect("contact.npz", stream)["passed"])

    def test_object_arrays_are_rejected(self):
        self.assertFalse(raw(dof=np.array([object()], dtype=object))["passed"])

    def test_corrupt_archive_is_rejected(self):
        self.assertFalse(inspect("broken.npz", io.BytesIO(b"broken"))["passed"])


if __name__ == "__main__":
    unittest.main()
