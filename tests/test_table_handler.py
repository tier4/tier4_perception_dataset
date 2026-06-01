#!/usr/bin/env python3
"""Unit tests for TableHandler with optimized duplicate detection."""
import unittest
from unittest.mock import patch

import attrs
from t4_devkit.schema import Category, EgoPose, Instance, SampleAnnotation

from perception_dataset.t4_dataset.table_handler import TableHandler


class TestTableHandler(unittest.TestCase):
    """Test TableHandler functionality and performance"""

    def setUp(self):
        """Set up test fixtures"""
        self.handler = TableHandler(SampleAnnotation)

    def test_insert_unique_records(self):
        """Test inserting unique records"""
        token1 = self.handler.insert_into_table(
            sample_token="sample_1",
            instance_token="instance_1",
            attribute_tokens=[],
            visibility_token="vis_1",
            translation=(1.0, 2.0, 3.0),
            velocity=(0.0, 0.0, 0.0),
            acceleration=(0.0, 0.0, 0.0),
            size=(1.0, 1.0, 1.0),
            rotation=(0.0, 0.0, 0.0, 1.0),
            prev="",
            next="",
            num_lidar_pts=100,
            num_radar_pts=0,
        )

        token2 = self.handler.insert_into_table(
            sample_token="sample_2",  # Different sample_token
            instance_token="instance_1",
            attribute_tokens=[],
            visibility_token="vis_1",
            translation=(1.0, 2.0, 3.0),
            velocity=(0.0, 0.0, 0.0),
            acceleration=(0.0, 0.0, 0.0),
            size=(1.0, 1.0, 1.0),
            rotation=(0.0, 0.0, 0.0, 1.0),
            prev="",
            next="",
            num_lidar_pts=100,
            num_radar_pts=0,
        )

        self.assertNotEqual(token1, token2)
        self.assertEqual(len(self.handler), 2)

    def test_duplicate_detection(self):
        """Test that duplicate records are detected and rejected"""
        # Insert first record
        self.handler.insert_into_table(
            sample_token="sample_1",
            instance_token="instance_1",
            attribute_tokens=[],
            visibility_token="vis_1",
            translation=(1.0, 2.0, 3.0),
            velocity=(0.0, 0.0, 0.0),
            acceleration=(0.0, 0.0, 0.0),
            size=(1.0, 1.0, 1.0),
            rotation=(0.0, 0.0, 0.0, 1.0),
            prev="",
            next="",
            num_lidar_pts=100,
            num_radar_pts=0,
        )

        # Try to insert identical record (should raise ValueError)
        with self.assertRaises(ValueError) as context:
            self.handler.insert_into_table(
                sample_token="sample_1",
                instance_token="instance_1",
                attribute_tokens=[],
                visibility_token="vis_1",
                translation=(1.0, 2.0, 3.0),
                velocity=(0.0, 0.0, 0.0),
                acceleration=(0.0, 0.0, 0.0),
                size=(1.0, 1.0, 1.0),
                rotation=(0.0, 0.0, 0.0, 1.0),
                prev="",
                next="",
                num_lidar_pts=100,
                num_radar_pts=0,
            )

        self.assertIn("Duplicate record found", str(context.exception))
        self.assertEqual(len(self.handler), 1)

    def test_hash_collision_handling(self):
        """Test that hash collisions are handled correctly"""
        # Create a mock hash function that always returns the same value
        with patch.object(self.handler, "_get_record_content_hash", return_value="same_hash"):
            # Insert first record
            token1 = self.handler.insert_into_table(
                sample_token="sample_1",
                instance_token="instance_1",
                attribute_tokens=[],
                visibility_token="vis_1",
                translation=(1.0, 2.0, 3.0),
                velocity=(0.0, 0.0, 0.0),
                acceleration=(0.0, 0.0, 0.0),
                size=(1.0, 1.0, 1.0),
                rotation=(0.0, 0.0, 0.0, 1.0),
                prev="",
                next="",
                num_lidar_pts=100,
                num_radar_pts=0,
            )

            # Insert different record with same hash (should succeed)
            token2 = self.handler.insert_into_table(
                sample_token="sample_2",  # Different content
                instance_token="instance_2",
                attribute_tokens=[],
                visibility_token="vis_2",
                translation=(4.0, 5.0, 6.0),
                velocity=(1.0, 1.0, 1.0),
                acceleration=(0.0, 0.0, 0.0),
                size=(2.0, 2.0, 2.0),
                rotation=(0.0, 0.0, 0.0, 1.0),
                prev="",
                next="",
                num_lidar_pts=200,
                num_radar_pts=10,
            )

            # Verify both records were added
            self.assertNotEqual(token1, token2)
            self.assertEqual(len(self.handler), 2)

            # Try to insert actual duplicate (should fail)
            with self.assertRaises(ValueError):
                self.handler.insert_into_table(
                    sample_token="sample_1",
                    instance_token="instance_1",
                    attribute_tokens=[],
                    visibility_token="vis_1",
                    translation=(1.0, 2.0, 3.0),
                    velocity=(0.0, 0.0, 0.0),
                    acceleration=(0.0, 0.0, 0.0),
                    size=(1.0, 1.0, 1.0),
                    rotation=(0.0, 0.0, 0.0, 1.0),
                    prev="",
                    next="",
                    num_lidar_pts=100,
                    num_radar_pts=0,
                )

    def test_update_record_hash_consistency(self):
        """Test that hash mappings are correctly updated when records are modified"""
        # Insert a record
        token = self.handler.insert_into_table(
            sample_token="sample_1",
            instance_token="instance_1",
            attribute_tokens=[],
            visibility_token="vis_1",
            translation=(1.0, 2.0, 3.0),
            velocity=(0.0, 0.0, 0.0),
            acceleration=(0.0, 0.0, 0.0),
            size=(1.0, 1.0, 1.0),
            rotation=(0.0, 0.0, 0.0, 1.0),
            prev="",
            next="",
            num_lidar_pts=100,
            num_radar_pts=0,
        )

        # Update the record
        self.handler.update_record_from_token(
            token, translation=(4.0, 5.0, 6.0), num_lidar_pts=200
        )

        # Try to insert a record with the old values (should succeed)
        new_token = self.handler.insert_into_table(
            sample_token="sample_1",
            instance_token="instance_1",
            attribute_tokens=[],
            visibility_token="vis_1",
            translation=(1.0, 2.0, 3.0),  # Old translation
            velocity=(0.0, 0.0, 0.0),
            acceleration=(0.0, 0.0, 0.0),
            size=(1.0, 1.0, 1.0),
            rotation=(0.0, 0.0, 0.0, 1.0),
            prev="",
            next="",
            num_lidar_pts=100,  # Old num_lidar_pts
            num_radar_pts=0,
        )

        self.assertNotEqual(token, new_token)
        self.assertEqual(len(self.handler), 2)

        # Try to insert a record with the new values (should fail)
        with self.assertRaises(ValueError):
            self.handler.insert_into_table(
                sample_token="sample_1",
                instance_token="instance_1",
                attribute_tokens=[],
                visibility_token="vis_1",
                translation=(4.0, 5.0, 6.0),  # New translation
                velocity=(0.0, 0.0, 0.0),
                acceleration=(0.0, 0.0, 0.0),
                size=(1.0, 1.0, 1.0),
                rotation=(0.0, 0.0, 0.0, 1.0),
                prev="",
                next="",
                num_lidar_pts=200,  # New num_lidar_pts
                num_radar_pts=0,
            )

    def test_set_record_to_table_replaces_hash_index_for_same_token(self):
        """Test hash index stays consistent when replacing an existing token via set_record_to_table."""
        token = self.handler.insert_into_table(
            sample_token="sample_1",
            instance_token="instance_1",
            attribute_tokens=[],
            visibility_token="vis_1",
            translation=(1.0, 2.0, 3.0),
            velocity=(0.0, 0.0, 0.0),
            acceleration=(0.0, 0.0, 0.0),
            size=(1.0, 1.0, 1.0),
            rotation=(0.0, 0.0, 0.0, 1.0),
            prev="",
            next="",
            num_lidar_pts=100,
            num_radar_pts=0,
        )

        # Replace record content while keeping the same token.
        updated_record = attrs.evolve(
            self.handler.get_record_from_token(token),
            translation=(9.0, 8.0, 7.0),
            num_lidar_pts=999,
        )
        self.handler.set_record_to_table(updated_record)

        # Old content should no longer be considered duplicate.
        new_token = self.handler.insert_into_table(
            sample_token="sample_1",
            instance_token="instance_1",
            attribute_tokens=[],
            visibility_token="vis_1",
            translation=(1.0, 2.0, 3.0),
            velocity=(0.0, 0.0, 0.0),
            acceleration=(0.0, 0.0, 0.0),
            size=(1.0, 1.0, 1.0),
            rotation=(0.0, 0.0, 0.0, 1.0),
            prev="",
            next="",
            num_lidar_pts=100,
            num_radar_pts=0,
        )
        self.assertNotEqual(token, new_token)

    def test_duplicate_check_is_not_called_without_hash_collision(self):
        """Duplicate equality check should be skipped when hashes do not collide."""
        with patch.object(
            TableHandler, "_is_duplicate_record", wraps=TableHandler._is_duplicate_record
        ) as mock_is_duplicate:
            handler = TableHandler(Instance)
            for i in range(1000):
                handler.insert_into_table(
                    instance_name=f"instance_{i}",
                    category_token=f"cat_{i % 10}",
                    nbr_annotations=0,
                    first_annotation_token="",
                    last_annotation_token="",
                )
            self.assertEqual(len(handler), 1000)

        # No hash collision expected in this deterministic dataset.
        # Therefore _is_duplicate_record should never be called.
        # If this regresses to scan-based detection, this assertion fails.
        self.assertEqual(mock_is_duplicate.call_count, 0)

    def test_stable_hash_across_runs(self):
        """Test that hash values are consistent across different instances"""
        handler1 = TableHandler(Category)
        handler2 = TableHandler(Category)

        # Insert same data in both handlers
        token1 = handler1.insert_into_table(name="car", description="A four-wheeled vehicle")

        token2 = handler2.insert_into_table(name="car", description="A four-wheeled vehicle")

        # Get hash values (accessing private method for testing)
        record1 = handler1.get_record_from_token(token1)
        record2 = handler2.get_record_from_token(token2)

        hash1 = handler1._get_record_content_hash(record1)
        hash2 = handler2._get_record_content_hash(record2)

        # Hashes should be the same for identical content
        self.assertEqual(hash1, hash2)

        # Verify duplicate detection works across instances
        with self.assertRaises(ValueError):
            handler1.insert_into_table(name="car", description="A four-wheeled vehicle")

    def test_field_cache_invalidated_on_update(self):
        """Test get_token_from_field cache is invalidated after update."""
        handler = TableHandler(Category)
        token = handler.insert_into_table(name="car", description="desc")

        # Populate cache for old field value.
        self.assertEqual(handler.get_token_from_field("name", "car"), token)

        # Update name and ensure stale cache doesn't return old token.
        handler.update_record_from_token(token, name="truck")
        self.assertIsNone(handler.get_token_from_field("name", "car"))
        self.assertEqual(handler.get_token_from_field("name", "truck"), token)

    def test_set_record_to_table_exception_keeps_indexes_consistent(self):
        """Failed set_record_to_table should not leave hash index in a broken state."""
        token = self.handler.insert_into_table(
            sample_token="sample_1",
            instance_token="instance_1",
            attribute_tokens=[],
            visibility_token="vis_1",
            translation=(1.0, 2.0, 3.0),
            velocity=(0.0, 0.0, 0.0),
            acceleration=(0.0, 0.0, 0.0),
            size=(1.0, 1.0, 1.0),
            rotation=(0.0, 0.0, 0.0, 1.0),
            prev="",
            next="",
            num_lidar_pts=100,
            num_radar_pts=0,
        )
        original = self.handler.get_record_from_token(token)
        replacement = attrs.evolve(original, translation=(9.0, 8.0, 7.0))

        old_hash = self.handler._get_record_content_hash(original)
        with patch.object(
            self.handler, "_get_record_content_hash", side_effect=[old_hash, RuntimeError("boom")]
        ):
            with self.assertRaises(RuntimeError):
                self.handler.set_record_to_table(replacement)

        # Old record remains valid and duplicate detection still works.
        self.assertEqual(
            tuple(self.handler.get_record_from_token(token).translation),
            tuple(original.translation),
        )
        with self.assertRaises(ValueError):
            self.handler.insert_into_table(
                sample_token="sample_1",
                instance_token="instance_1",
                attribute_tokens=[],
                visibility_token="vis_1",
                translation=(1.0, 2.0, 3.0),
                velocity=(0.0, 0.0, 0.0),
                acceleration=(0.0, 0.0, 0.0),
                size=(1.0, 1.0, 1.0),
                rotation=(0.0, 0.0, 0.0, 1.0),
                prev="",
                next="",
                num_lidar_pts=100,
                num_radar_pts=0,
            )

    def test_update_record_exception_keeps_indexes_consistent(self):
        """Failed update_record_from_token should not remove old hash mapping."""
        token = self.handler.insert_into_table(
            sample_token="sample_1",
            instance_token="instance_1",
            attribute_tokens=[],
            visibility_token="vis_1",
            translation=(1.0, 2.0, 3.0),
            velocity=(0.0, 0.0, 0.0),
            acceleration=(0.0, 0.0, 0.0),
            size=(1.0, 1.0, 1.0),
            rotation=(0.0, 0.0, 0.0, 1.0),
            prev="",
            next="",
            num_lidar_pts=100,
            num_radar_pts=0,
        )
        current = self.handler.get_record_from_token(token)
        old_hash = self.handler._get_record_content_hash(current)

        with patch.object(
            self.handler, "_get_record_content_hash", side_effect=[old_hash, RuntimeError("boom")]
        ):
            with self.assertRaises(RuntimeError):
                self.handler.update_record_from_token(token, translation=(4.0, 5.0, 6.0))

        # Old record remains, and old duplicate is still recognized.
        self.assertEqual(
            tuple(self.handler.get_record_from_token(token).translation),
            tuple(current.translation),
        )
        with self.assertRaises(ValueError):
            self.handler.insert_into_table(
                sample_token="sample_1",
                instance_token="instance_1",
                attribute_tokens=[],
                visibility_token="vis_1",
                translation=(1.0, 2.0, 3.0),
                velocity=(0.0, 0.0, 0.0),
                acceleration=(0.0, 0.0, 0.0),
                size=(1.0, 1.0, 1.0),
                rotation=(0.0, 0.0, 0.0, 1.0),
                prev="",
                next="",
                num_lidar_pts=100,
                num_radar_pts=0,
            )


class TestTableHandlerReuseIfDuplicate(unittest.TestCase):
    """Test the opt-in reuse_if_duplicate behavior (used by EgoPose / VehicleState)."""

    def setUp(self):
        self.handler = TableHandler(EgoPose)

    @staticmethod
    def _ego_pose_kwargs(**overrides):
        kwargs = dict(
            translation=(1.0, 2.0, 3.0),
            rotation=(0.0, 0.0, 0.0, 1.0),
            timestamp=1_000_000,
            twist=None,
            acceleration=None,
            geocoordinate=None,
        )
        kwargs.update(overrides)
        return kwargs

    def test_reuse_returns_existing_token(self):
        """Identical content with reuse_if_duplicate=True reuses the first token."""
        token1 = self.handler.insert_into_table(
            reuse_if_duplicate=True, **self._ego_pose_kwargs()
        )
        token2 = self.handler.insert_into_table(
            reuse_if_duplicate=True, **self._ego_pose_kwargs()
        )

        self.assertEqual(token1, token2)
        self.assertEqual(len(self.handler), 1)

    def test_default_still_raises_on_duplicate(self):
        """The default (reuse_if_duplicate=False) keeps the strict duplicate-error invariant."""
        self.handler.insert_into_table(**self._ego_pose_kwargs())

        with self.assertRaises(ValueError) as context:
            self.handler.insert_into_table(**self._ego_pose_kwargs())

        self.assertIn("Duplicate record found", str(context.exception))
        self.assertEqual(len(self.handler), 1)

    def test_reuse_does_not_over_collapse_distinct_content(self):
        """Records differing only by timestamp are distinct, even with reuse enabled."""
        token1 = self.handler.insert_into_table(
            reuse_if_duplicate=True, **self._ego_pose_kwargs(timestamp=1_000_000)
        )
        token2 = self.handler.insert_into_table(
            reuse_if_duplicate=True, **self._ego_pose_kwargs(timestamp=1_000_001)
        )

        self.assertNotEqual(token1, token2)
        self.assertEqual(len(self.handler), 2)

    def test_reuse_path_does_not_corrupt_hash_index(self):
        """A reuse hit must not mutate the hash index or break later strict inserts."""
        token_a = self.handler.insert_into_table(
            reuse_if_duplicate=True, **self._ego_pose_kwargs(timestamp=1_000_000)
        )
        # Reuse A (no new record added).
        self.assertEqual(
            self.handler.insert_into_table(
                reuse_if_duplicate=True, **self._ego_pose_kwargs(timestamp=1_000_000)
            ),
            token_a,
        )

        # Distinct record B inserts under the default strict mode.
        token_b = self.handler.insert_into_table(**self._ego_pose_kwargs(timestamp=2_000_000))
        self.assertNotEqual(token_a, token_b)
        self.assertEqual(len(self.handler), 2)

        # An exact duplicate of B under strict mode still raises.
        with self.assertRaises(ValueError):
            self.handler.insert_into_table(**self._ego_pose_kwargs(timestamp=2_000_000))

    def test_reuse_honors_equality_gate_under_hash_collision(self):
        """With a forced hash collision, content-distinct records still insert as new tokens."""
        with patch.object(self.handler, "_get_record_content_hash", return_value="same_hash"):
            token1 = self.handler.insert_into_table(
                reuse_if_duplicate=True, **self._ego_pose_kwargs(timestamp=1_000_000)
            )
            # Same hash but different content -> must NOT be reused.
            token2 = self.handler.insert_into_table(
                reuse_if_duplicate=True, **self._ego_pose_kwargs(timestamp=1_000_001)
            )
            self.assertNotEqual(token1, token2)
            self.assertEqual(len(self.handler), 2)

            # Same hash AND identical content -> reused.
            token3 = self.handler.insert_into_table(
                reuse_if_duplicate=True, **self._ego_pose_kwargs(timestamp=1_000_000)
            )
            self.assertEqual(token1, token3)
            self.assertEqual(len(self.handler), 2)


if __name__ == "__main__":
    unittest.main()
