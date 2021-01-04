import cudf
import numpy as np
import pytest

import nvtabular as nvt

json_sample = {
    "conts": {
        "cont_1": {"dtype": np.float32, "min_val": 0, "max_val": 1},
        "cont_2": {"dtype": np.float32, "min_val": 0, "max_val": 1},
        "cont_3": {"dtype": np.float32, "min_val": 0, "max_val": 1},
        "cont_4": {"dtype": np.float32, "min_val": 0, "max_val": 1},
        "cont_5": {"dtype": np.float32, "min_val": 0, "max_val": 1},
    },
    "cats": {
        "cat_1": {"dtype": None, "cardinality": 50, "max_entry_size": 1, "min_entry_size": 5},
        "cat_2": {"dtype": None, "cardinality": 50, "max_entry_size": 1, "min_entry_size": 5},
        "cat_3": {"dtype": None, "cardinality": 50, "max_entry_size": 1, "min_entry_size": 5},
        "cat_4": {"dtype": None, "cardinality": 50, "max_entry_size": 1, "min_entry_size": 5},
        "cat_5": {"dtype": None, "cardinality": 50, "max_entry_size": 1, "min_entry_size": 5},
    },
}


@pytest.mark.parametrize("num_rows", [1000, 10000])
def test_powerlaw(num_rows):
    cats = ["CAT_0", "CAT_1", "CAT_2", "CAT_3", "CAT_4"]

    cols = nvt.data_gen._get_cols_from_schema(json_sample)
    conts_rep = cols["conts"]
    cats_rep = cols["cats"]

    df_gen = nvt.data_gen.DatasetGen(nvt.data_gen.PowerLawDistro(0.1))
    df_pw = cudf.DataFrame()
    for x in range(10):
        df_pw_1 = df_gen.create_df(num_rows, conts_rep, cats_rep, dist=df_gen)
        df_pw = cudf.concat([df_pw, df_pw_1], axis=0)
    sts, ps = df_gen.verify_df(df_pw[cats])
    assert all(s > 0.9 for s in sts)


@pytest.mark.parametrize("num_rows", [1000, 10000])
def test_uniform(num_rows):
    cats = ["CAT_0", "CAT_1", "CAT_2", "CAT_3", "CAT_4"]

    cols = nvt.data_gen._get_cols_from_schema(json_sample)
    conts_rep = cols["conts"]
    cats_rep = cols["cats"]

    df_gen = nvt.data_gen.DatasetGen(nvt.data_gen.UniformDistro())
    df_uni = df_gen.create_df(num_rows, conts_rep, cats_rep, dist=df_gen)
    sts, ps = df_gen.verify_df(df_uni[cats])
    assert all(s > 0.9 for s in sts)


@pytest.mark.parametrize("num_rows", [1000, 10000])
def test_cat_rep(num_rows):
    cats = ["CAT_0", "CAT_1", "CAT_2", "CAT_3", "CAT_4"]

    cols = nvt.data_gen._get_cols_from_schema(json_sample)
    conts_rep = cols["conts"]
    cats_rep = cols["cats"]

    df_gen = nvt.data_gen.DatasetGen(nvt.data_gen.UniformDistro())
    df_uni = df_gen.create_df(num_rows, conts_rep, cats_rep, dist=df_gen, entries=True)
    df_cats = df_uni.select_dtypes("O")
    assert df_cats.shape[1] == len(cats)
    assert df_cats.shape[0] == num_rows
    for idx, cat in enumerate(cats):
        df_cats[cat].nunique() == cats_rep[idx][1]
        len(df_cats[cat].min()) == cats_rep[idx][2]
        len(df_cats[cat].max()) == cats_rep[idx][3]


def test_json_convert():
    cols = nvt.data_gen._get_cols_from_schema(json_sample)
    assert len(cols["conts"]) == len(json_sample["conts"].keys())
    assert len(cols["cats"]) == len(json_sample["cats"].keys())
