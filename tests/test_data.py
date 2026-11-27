import pytest

from churn.config import ID_COL, TARGET_COL
from churn.data import clean, split


def test_blank_total_charges_become_zero(raw_df):
    out = clean(raw_df)
    new_customers = raw_df["tenure"] == 0
    assert out["TotalCharges"].dtype == float
    assert out["TotalCharges"].isna().sum() == 0
    assert (out.loc[new_customers, "TotalCharges"] == 0).all()


def test_target_is_binary(raw_df):
    out = clean(raw_df)
    assert set(out[TARGET_COL].unique()) <= {0, 1}


def test_customer_id_is_dropped(raw_df):
    assert ID_COL not in clean(raw_df).columns


def test_clean_does_not_modify_input(raw_df):
    before = raw_df.copy()
    clean(raw_df)
    assert raw_df.equals(before)


def test_missing_column_raises(raw_df):
    with pytest.raises(ValueError, match="Missing columns"):
        clean(raw_df.drop(columns=["Contract"]))


def test_unexpected_target_value_raises(raw_df):
    raw_df.loc[0, TARGET_COL] = "Maybe"
    with pytest.raises(ValueError):
        clean(raw_df)


def test_split_has_no_overlap_and_keeps_churn_rate(raw_df):
    X_train, X_test, y_train, y_test = split(clean(raw_df))
    assert set(X_train.index).isdisjoint(X_test.index)
    assert len(X_train) + len(X_test) == len(raw_df)
    assert abs(y_train.mean() - y_test.mean()) < 0.05
