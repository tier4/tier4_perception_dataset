#!/usr/bin/env python3
"""Unit tests for TableHandler with optimized duplicate detection."""
import time
import unittest
from unittest.mock import patch

from t4_devkit.schema import Category, Instance, SampleAnnotation

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

    def test_performance_improvement(self):
        """Test that performance scales linearly with optimized implementation"""
        sizes = [100, 500, 1000]
        times = []

        for size in sizes:
            handler = TableHandler(Instance)
            start_time = time.time()

            for i in range(size):
                handler.insert_into_table(
                    instance_name=f"instance_{i}",
                    category_token=f"cat_{i % 10}",  # Some variety in categories
                    nbr_annotations=0,
                    first_annotation_token="",
                    last_annotation_token="",
                )

            elapsed = time.time() - start_time
            times.append(elapsed)

            # Verify all records were inserted
            self.assertEqual(len(handler), size)

        # Check that time doesn't grow quadratically
        # With O(n²), doubling size would ~4x the time
        # With O(n), doubling size would ~2x the time
        # Allow some margin for system variance
        time_ratio_500_to_100 = times[1] / times[0]
        time_ratio_1000_to_500 = times[2] / times[1]

        # These ratios should be roughly similar if O(n)
        # and much smaller than 4 (which would indicate O(n²))
        self.assertLess(time_ratio_500_to_100, 10)  # Should be ~5 for O(n)
        self.assertLess(time_ratio_1000_to_500, 4)  # Should be ~2 for O(n)

        print("\nPerformance test results:")
        print(f"100 records: {times[0]:.3f}s")
        print(f"500 records: {times[1]:.3f}s (ratio: {time_ratio_500_to_100:.2f})")
        print(f"1000 records: {times[2]:.3f}s (ratio: {time_ratio_1000_to_500:.2f})")

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


if __name__ == "__main__":
    unittest.main()
