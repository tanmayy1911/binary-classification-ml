
from src.data import prepare_data

def test_prepare_data_shapes():
    bundle = prepare_data()
    assert bundle.X_train.shape[0] > 0
    assert bundle.X_test.shape[0] > 0
    assert bundle.X_train.shape[1] == bundle.X_test.shape[1]
    assert len(bundle.feature_names) == bundle.X_train.shape[1]
