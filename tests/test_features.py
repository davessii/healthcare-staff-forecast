import unittest
import numpy as np
import pandas as pd
from scripts.make_demo_data import generate
from src.features import add_ml_features, clean_feature_rows, get_feature_columns


class FeatureTests(unittest.TestCase):
    def setUp(self):
        self.data = pd.DataFrame(generate())

    def test_demo_is_reproducible_and_fictional(self):
        self.assertEqual(generate(), generate())
        self.assertTrue(self.data['region'].str.startswith('Демо регион ').all())
        self.assertEqual(len(self.data), 180)
        self.assertFalse(self.data.duplicated(['region', 'year']).any())

    def test_lags_and_targets_do_not_cross_regions(self):
        features = add_ml_features(self.data)
        for _, group in features.groupby('region'):
            self.assertTrue(np.isnan(group.iloc[0]['doctors_count_lag_1']))
            self.assertEqual(group.iloc[1]['doctors_count_lag_1'], group.iloc[0]['doctors_count'])
            self.assertEqual(group.iloc[0]['target_doctors_next_year'], group.iloc[1]['doctors_count'])
            self.assertTrue(np.isnan(group.iloc[-1]['target_doctors_next_year']))

    def test_targets_are_not_input_features(self):
        features = add_ml_features(self.data)
        names = get_feature_columns(features)
        self.assertNotIn('target_doctors_next_year', names)
        self.assertNotIn('target_nurses_next_year', names)
        self.assertFalse(clean_feature_rows(features, 'target_doctors_next_year')[names].isna().any().any())

    def test_future_observation_does_not_change_past_features(self):
        modified = self.data.copy()
        modified.loc[modified['year'] == 2024, 'doctors_count'] *= 2
        left, right = add_ml_features(self.data), add_ml_features(modified)
        names = get_feature_columns(left)
        pd.testing.assert_frame_equal(left.loc[left['year'] <= 2023, names], right.loc[right['year'] <= 2023, names])


if __name__ == '__main__':
    unittest.main()
