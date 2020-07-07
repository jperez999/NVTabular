#
# Copyright (c) 2020, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import cudf
import cupy
import numpy as np
from cudf._lib.nvtx import annotate
from dask.delayed import Delayed

import nvtabular.categorify as nvt_cat

CONT = "continuous"
CAT = "categorical"
ALL = "all"


class Operator:
    """
    Base class for all operator classes.
    """

    def __init__(self, columns=None):
        self.columns = columns

    @property
    def _id(self):
        return str(self.__class__.__name__)

    def describe(self):
        raise NotImplementedError("All operators must have a desription.")

    def get_columns(self, cols_ctx, cols_grp, target_cols):
        # providing any operator with direct list of columns overwrites cols dict
        # burden on user to ensure columns exist in dataset (as discussed)
        if self.columns:
            return self.columns
        tar_cols = []
        for tar in target_cols:
            if tar in cols_ctx[cols_grp].keys():
                tar_cols = tar_cols + cols_ctx[cols_grp][tar]
        return tar_cols


class TransformOperator(Operator):
    """
    Base class for transformer operator classes.
    """

    default_in = None
    default_out = None

    def __init__(self, columns=None, preprocessing=True, replace=True):
        super().__init__(columns=columns)
        self.preprocessing = preprocessing
        self.replace = replace

    def get_default_in(self):
        if self.default_in is None:
            raise NotImplementedError(
                "default_in columns have not been specified for this operator"
            )
        return self.default_in

    def get_default_out(self):
        if self.default_out is None:
            raise NotImplementedError(
                "default_out columns have not been specified for this operator"
            )
        return self.default_out

    def update_columns_ctx(self, columns_ctx, input_cols, new_cols, origin_targets, pro=False):
        """
        columns_ctx: columns context, belonging to the container workflow object
        input_cols: input columns; columns actioned on origin columns context key
        new_cols: new columns; new columns generated by operator to be added to columns context
        ----
        This function generalizes the action of updating the columns context dictionary
        of the container workflow object, after an operator has created new columns via a
        new transformation of a subset or entire dataset.
        """

        new_key = self._id

        columns_ctx[input_cols][new_key] = []
        if self.replace and self.preprocessing:
            # not making new columns instead using old ones
            # must reference original target with new operator for chaining
            columns_ctx[input_cols][new_key] = origin_targets
            return
        columns_ctx[input_cols][new_key] = list(new_cols)
        if not self.preprocessing and self._id not in columns_ctx["final"]["ctx"][input_cols]:
            columns_ctx["final"]["ctx"][input_cols].append(self._id)

    def apply_op(
        self,
        gdf: cudf.DataFrame,
        columns_ctx: dict,
        input_cols,
        target_cols=["base"],
        stats_context=None,
    ):
        target_columns = self.get_columns(columns_ctx, input_cols, target_cols)
        new_gdf = self.op_logic(gdf, target_columns, stats_context=stats_context)
        self.update_columns_ctx(columns_ctx, input_cols, new_gdf.columns, target_columns)
        return self.assemble_new_df(gdf, new_gdf, target_columns)

    def assemble_new_df(self, origin_gdf, new_gdf, target_columns):
        if self.replace and self.preprocessing and target_columns:
            if new_gdf.shape[0] < origin_gdf.shape[0]:
                return new_gdf
            else:
                origin_gdf[target_columns] = new_gdf
                return origin_gdf
        return cudf.concat([origin_gdf, new_gdf], axis=1)

    def op_logic(self, gdf, target_columns, stats_context=None):
        raise NotImplementedError(
            """Must implement transform in the op_logic method,
                                     The return value must be a dataframe with all required
                                     transforms."""
        )


class DFOperator(TransformOperator):
    """
    Base class for data frame operator classes.
    """

    @property
    def req_stats(self):
        raise NotImplementedError(
            "Should consist of a list of identifiers, that should map to available statistics"
        )


class StatOperator(Operator):
    """
    Base class for statistical operator classes.
    """

    def __init__(self, columns=None):
        super(StatOperator, self).__init__(columns)

    def stat_logic(self, ddf, columns_ctx, input_cols, target_cols):
        raise NotImplementedError(
            """The dask operations needed to return a dictionary of uncomputed statistics."""
        )

    def finalize(self, dask_stats):
        raise NotImplementedError(
            """Follow-up operations to convert dask statistics in to member variables"""
        )

    def registered_stats(self):
        raise NotImplementedError(
            """Should return a list of statistics this operator will collect.
                The list is comprised of simple string values."""
        )

    def stats_collected(self):
        raise NotImplementedError(
            """Should return a list of tuples of name and statistics operator."""
        )

    def clear(self):
        raise NotImplementedError("""zero and reinitialize all relevant statistical properties""")


class MinMax(StatOperator):
    """
    MinMax operation calculates min and max statistics of features.

    Parameters
    -----------
    columns :
    batch_mins : list of float, default None
    batch_maxs : list of float, default None
    mins : list of float, default None
    maxs : list of float, default None
    """

    def __init__(self, columns=None, batch_mins=None, batch_maxs=None, mins=None, maxs=None):
        super().__init__(columns=columns)
        self.batch_mins = batch_mins if batch_mins is not None else {}
        self.batch_maxs = batch_maxs if batch_maxs is not None else {}
        self.mins = mins if mins is not None else {}
        self.maxs = maxs if maxs is not None else {}

    @annotate("MinMax_op", color="green", domain="nvt_python")
    def stat_logic(self, ddf, columns_ctx, input_cols, target_cols):
        cols = self.get_columns(columns_ctx, input_cols, target_cols)
        dask_stats = {}
        dask_stats["mins"] = ddf[cols].min()
        dask_stats["maxs"] = ddf[cols].max()
        return dask_stats

    @annotate("MinMax_finalize", color="green", domain="nvt_python")
    def finalize(self, stats):
        for col in stats["mins"].index:
            self.mins[col] = stats["mins"][col]
            self.maxs[col] = stats["maxs"][col]

    def registered_stats(self):
        return ["mins", "maxs", "batch_mins", "batch_maxs"]

    def stats_collected(self):
        result = [
            ("mins", self.mins),
            ("maxs", self.maxs),
            ("batch_mins", self.batch_mins),
            ("batch_maxs", self.batch_maxs),
        ]
        return result

    def clear(self):
        self.batch_mins = {}
        self.batch_maxs = {}
        self.mins = {}
        self.maxs = {}
        return


class Moments(StatOperator):
    """
    Moments operation calculates some of the statistics of features including
    mean, variance, standarded deviation, and count.

    Parameters
    -----------
    columns :
    counts : list of float, default None
    means : list of float, default None
    varis : list of float, default None
    stds : list of float, default None
    """

    def __init__(self, columns=None, counts=None, means=None, varis=None, stds=None):
        super().__init__(columns=columns)
        self.counts = counts if counts is not None else {}
        self.means = means if means is not None else {}
        self.varis = varis if varis is not None else {}
        self.stds = stds if stds is not None else {}

    @annotate("Moments_op", color="green", domain="nvt_python")
    def stat_logic(self, ddf, columns_ctx, input_cols, target_cols):
        cols = self.get_columns(columns_ctx, input_cols, target_cols)
        dask_stats = {}
        dask_stats["count"] = ddf[cols].count()
        dask_stats["mean"] = ddf[cols].mean()
        dask_stats["std"] = ddf[cols].std()
        return dask_stats

    @annotate("Moments_finalize", color="green", domain="nvt_python")
    def finalize(self, dask_stats):
        for col in dask_stats["count"].index:
            self.counts[col] = float(dask_stats["count"][col])
            self.means[col] = float(dask_stats["mean"][col])
            self.stds[col] = float(dask_stats["std"][col])
            self.varis[col] = float(self.stds[col] * self.stds[col])

    def registered_stats(self):
        return ["means", "stds", "vars", "counts"]

    def stats_collected(self):
        result = [
            ("means", self.means),
            ("stds", self.stds),
            ("vars", self.varis),
            ("counts", self.counts),
        ]
        return result

    def clear(self):
        self.counts = {}
        self.means = {}
        self.varis = {}
        self.stds = {}
        return


class Median(StatOperator):
    """
    This operation calculates median of features.

    Parameters
    -----------
    columns :
    fill : float, default None
    batch_medians : list, default None
    medians : list, default None
    """

    def __init__(self, columns=None, fill=None, batch_medians=None, medians=None):
        super().__init__(columns=columns)
        self.fill = fill
        self.batch_medians = batch_medians if batch_medians is not None else {}
        self.medians = medians if medians is not None else {}

    @annotate("Median_op", color="green", domain="nvt_python")
    def stat_logic(self, ddf, columns_ctx, input_cols, target_cols):
        cols = self.get_columns(columns_ctx, input_cols, target_cols)
        # TODO: Use `method="tidigest"` when crick supports device
        dask_stats = ddf[cols].quantile(q=0.5, method="dask")
        return dask_stats

    @annotate("Median_finalize", color="green", domain="nvt_python")
    def finalize(self, dask_stats):
        for col in dask_stats.index:
            self.medians[col] = float(dask_stats[col])

    def registered_stats(self):
        return ["medians"]

    def stats_collected(self):
        result = [("medians", self.medians)]
        return result

    def clear(self):
        self.batch_medians = {}
        self.medians = {}
        return


class Encoder(StatOperator):
    """
    This is an internal operation. Encoder operation is used by
    the Categorify operation to calculate the unique numerical
    values to transform the categorical features.

    Parameters
    -----------
    use_frequency : bool
        use frequency based transformation or not.
    freq_threshold : int, default 0
        threshold value for frequency based transformation.
    limit_frac : float, default 0.5
        fraction of memory to use during unique id calculation.
    gpu_mem_util_limit : float, default 0.8
        GPU memory utilization limit during frequency based
        calculation. If limit is exceeded, unique ids are moved
        to host memory.
    gpu_mem_trans_use : float, default 0.8
        GPU memory utilization limit during transformation. How much
        GPU memory will be used during transformation is calculated
        using this parameter.
    split_out : dict, optional
        Used for multi-GPU category calculation.  Each key in the dict
        should correspond to a column name, and the value is the number
        of hash partitions to use for the categorical tree reduction.
        Only a single partition is used by default.
    out_path : str, optional
        Used for multi-GPU category calculation.  Root directory where
        unique categories will be written out in parquet format.
    columns :
    preprocessing : bool
    replace : bool
    """

    def __init__(
        self,
        use_frequency=False,
        freq_threshold=0,
        limit_frac=0.5,
        gpu_mem_util_limit=0.5,
        gpu_mem_trans_use=0.5,
        columns=None,
        categories=None,
        out_path=None,
        split_out=None,
        on_host=True,
    ):
        super(Encoder, self).__init__(columns)
        self.use_frequency = use_frequency
        self.freq_threshold = freq_threshold
        self.limit_frac = limit_frac
        self.gpu_mem_util_limit = gpu_mem_util_limit
        self.gpu_mem_trans_use = gpu_mem_trans_use
        self.categories = categories if categories is not None else {}
        self.out_path = out_path or "./"
        self.split_out = split_out
        self.on_host = on_host

    @annotate("Encoder_op", color="green", domain="nvt_python")
    def stat_logic(self, ddf, columns_ctx, input_cols, target_cols):
        cols = self.get_columns(columns_ctx, input_cols, target_cols)
        dsk, key = nvt_cat._get_categories(
            ddf, cols, self.out_path, self.freq_threshold, self.split_out, self.on_host
        )
        return Delayed(key, dsk)

    @annotate("Encoder_finalize", color="green", domain="nvt_python")
    def finalize(self, dask_stats):
        for col in dask_stats:
            self.categories[col] = dask_stats[col]

    def registered_stats(self):
        return ["categories"]

    def stats_collected(self):
        return [("categories", self.categories)]

    def clear(self):
        self.categories = {}


class ZeroFill(TransformOperator):
    """
    This operation sets negative values to zero.

    Although you can directly call methods of this class to
    transform your continuous features, it's typically used within a
    Workflow class.
    """

    default_in = CONT
    default_out = CONT

    @annotate("ZeroFill_op", color="darkgreen", domain="nvt_python")
    def op_logic(self, gdf: cudf.DataFrame, target_columns: list, stats_context=None):
        cont_names = target_columns
        if not cont_names:
            return gdf
        z_gdf = gdf[cont_names].fillna(0)
        z_gdf.columns = [f"{col}_{self._id}" for col in z_gdf.columns]
        z_gdf[z_gdf < 0] = 0
        return z_gdf


class Dropna(TransformOperator):
    """
    This operation detects missing values, and returns
    a cudf DataFrame with Null entries dropped from it.

    Although you can directly call methods of this class to
    transform your categorical and/or continuous features, it's typically used within a
    Workflow class.
    """

    default_in = ALL
    default_out = ALL

    @annotate("Dropna_op", color="darkgreen", domain="nvt_python")
    def apply_op(
        self,
        gdf: cudf.DataFrame,
        columns_ctx: dict,
        input_cols,
        target_cols=["base"],
        stats_context=None,
    ):
        target_columns = self.get_columns(columns_ctx, input_cols, target_cols)
        new_gdf = gdf.dropna(subset=target_columns or None)
        new_gdf.reset_index(drop=True, inplace=True)
        self.update_columns_ctx(columns_ctx, input_cols, new_gdf.columns, target_columns)
        return new_gdf


class LogOp(TransformOperator):

    """
    Standardizing the features around 0 with a standard deviation
    of 1 is a common technique to compare measurements that have
    different units. This operation can be added to the workflow
    to standardize the features.

    Although you can directly call methods of this class to
    transform your continuous features, it's typically used within a
    Workflow class.
    """

    default_in = CONT
    default_out = CONT

    @annotate("LogOp_op", color="darkgreen", domain="nvt_python")
    def op_logic(self, gdf: cudf.DataFrame, target_columns: list, stats_context=None):
        cont_names = target_columns
        if not cont_names:
            return gdf
        new_gdf = np.log(gdf[cont_names].astype(np.float32) + 1)
        new_cols = [f"{col}_{self._id}" for col in new_gdf.columns]
        new_gdf.columns = new_cols
        return new_gdf


class HashBucket(TransformOperator):
    default_in = CAT
    default_out = CAT

    def __init__(self, num_buckets, columns=None, **kwargs):
        if isinstance(num_buckets, dict):
            columns = [i for i in num_buckets.keys()]
            self.num_buckets = num_buckets
        elif isinstance(num_buckets, (tuple, list)):
            assert columns is not None
            assert len(columns) == len(num_buckets)
            self.num_buckets = {col: nb for col, nb in zip(columns, num_buckets)}
        elif isinstance(num_buckets, int):
            self.num_buckets = num_buckets
        else:
            raise TypeError(
                "`num_buckets` must be dict, iterable, or int, got type {}".format(
                    type(num_buckets)
                )
            )
        super(HashBucket, self).__init__(columns=columns, **kwargs)

    @annotate("HashBucket_op", color="darkgreen", domain="nvt_python")
    def op_logic(self, gdf: cudf.DataFrame, target_columns: list, stats_context=None):
        cat_names = target_columns
        if isinstance(self.num_buckets, int):
            num_buckets = {name: self.num_buckets for name in cat_names}
        else:
            num_buckets = self.num_buckets

        new_gdf = cudf.DataFrame()
        for col, nb in num_buckets.items():
            new_col = f"{col}_{self._id}"
            new_gdf[new_col] = gdf[col].hash_values() % nb
        return new_gdf


class Normalize(DFOperator):
    """
    Standardizing the features around 0 with a standard deviation
    of 1 is a common technique to compare measurements that have
    different units. This operation can be added to the workflow
    to standardize the features.

    It performs Normalization using the mean std method.

    Although you can directly call methods of this class to
    transform your continuous features, it's typically used within a
    Workflow class.
    """

    default_in = CONT
    default_out = CONT

    @property
    def req_stats(self):
        return [Moments()]

    @annotate("Normalize_op", color="darkgreen", domain="nvt_python")
    def op_logic(self, gdf: cudf.DataFrame, target_columns: list, stats_context=None):
        cont_names = target_columns
        if not cont_names or not stats_context["stds"]:
            return
        gdf = self.apply_mean_std(gdf, stats_context, cont_names)
        return gdf

    def apply_mean_std(self, gdf, stats_context, cont_names):
        new_gdf = cudf.DataFrame()
        for name in cont_names:
            if stats_context["stds"][name] > 0:
                new_col = f"{name}_{self._id}"
                new_gdf[new_col] = (gdf[name] - stats_context["means"][name]) / (
                    stats_context["stds"][name]
                )
                new_gdf[new_col] = new_gdf[new_col].astype("float32")
        return new_gdf


class NormalizeMinMax(DFOperator):
    """
    Standardizing the features around 0 with a standard deviation
    of 1 is a common technique to compare measurements that have
    different units. This operation can be added to the workflow
    to standardize the features.

    It performs Normalization using the min max method.

    Although you can directly call methods of this class to
    transform your continuous features, it's typically used within a
    Workflow class.
    """

    default_in = CONT
    default_out = CONT

    @property
    def req_stats(self):
        return [MinMax()]

    @annotate("NormalizeMinMax_op", color="darkgreen", domain="nvt_python")
    def op_logic(self, gdf: cudf.DataFrame, target_columns: list, stats_context=None):
        cont_names = target_columns
        if not cont_names or not stats_context["mins"]:
            return
        gdf = self.apply_min_max(gdf, stats_context, cont_names)
        return gdf

    def apply_min_max(self, gdf, stats_context, cont_names):
        new_gdf = cudf.DataFrame()
        for name in cont_names:
            dif = stats_context["maxs"][name] - stats_context["mins"][name]
            new_col = f"{name}_{self._id}"
            if dif > 0:
                new_gdf[new_col] = (gdf[name] - stats_context["mins"][name]) / dif
            elif dif == 0:
                new_gdf[new_col] = gdf[name] / (2 * gdf[name])
            new_gdf[new_col] = new_gdf[new_col].astype("float32")
        return new_gdf


class FillMissing(DFOperator):
    """
    This operation replaces missing values with a constant pre-defined value

    Although you can directly call methods of this class to
    transform your continuous features, it's typically used within a
    Workflow class.

    Parameters
    -----------
    fill_val : float, default 0
        The constant value to replace missing values with
    columns :
    preprocessing : bool, default True
    replace : bool, default True
    """

    default_in = CONT
    default_out = CONT

    def __init__(self, fill_val=0, columns=None, preprocessing=True, replace=True):
        super().__init__(columns=columns, preprocessing=preprocessing, replace=replace)
        self.fill_val = fill_val

    @property
    def req_stats(self):
        return []

    @annotate("FillMissing_op", color="darkgreen", domain="nvt_python")
    def op_logic(self, gdf: cudf.DataFrame, target_columns: list, stats_context=None):
        cont_names = target_columns
        if not cont_names:
            return gdf
        z_gdf = gdf[cont_names].fillna(self.fill_val)
        z_gdf.columns = [f"{col}_{self._id}" for col in z_gdf.columns]
        return z_gdf


class FillMedian(DFOperator):
    """
    This operation replaces missing values with the median value for the column.
    Although you can directly call methods of this class to
    transform your continuous features, it's typically used within a
    Workflow class.

    Parameters
    -----------
    columns :
    preprocessing : bool, default True
    replace : bool, default True
    """

    default_in = CONT
    default_out = CONT

    @property
    def req_stats(self):
        return [Median()]

    @annotate("FillMedian_op", color="darkgreen", domain="nvt_python")
    def op_logic(self, gdf: cudf.DataFrame, target_columns: list, stats_context=None):
        if not target_columns:
            return gdf

        new_gdf = cudf.DataFrame()
        for col in target_columns:
            new_gdf[col] = gdf[col].fillna(stats_context["medians"][col])
        new_gdf.columns = [f"{col}_{self._id}" for col in new_gdf.columns]
        return new_gdf


class GroupByMoments(StatOperator):
    """
    One of the ways to create new features is to calculate
    the basic statistics of the data that is grouped by a categorical
    feature. This operator groups the data by the given categorical
    feature(s) and calculates the std, variance, and sum of requested continuous
    features along with count of every group. Then, merges these new statistics
    with the data using the unique ids of categorical data.

    Although you can directly call methods of this class to
    transform your categorical features, it's typically used within a
    Workflow class.

    Parameters
    -----------
    cat_names : list of str
        names of the categorical columns
    cont_names : list of str
        names of the continuous columns
    stats : list of str, default ['count']
        count of groups = ['count']
        sum of cont_col = ['sum']
    limit_frac : float, default 0.5
        fraction of memory to use during unique id calculation.
    gpu_mem_util_limit : float, default 0.5
        GPU memory utilization limit during frequency based
        calculation. If limit is exceeded, unique ids are moved
        to host memory.
    gpu_mem_trans_use : float, default 0.5
        GPU memory utilization limit during transformation. How much
        GPU memory will be used during transformation is calculated
        using this parameter.
    columns :
    order_column_name : str, default "order-nvtabular"
        a column name to be used to preserve the order of input data.
        cudf's merge function doesn't preserve the order of the data
        and this column name is used to create a column with integer
        values in ascending order.
    split_out : dict, optional
        Used for multi-GPU groupby reduction.  Each key in the dict
        should correspond to a column name, and the value is the number
        of hash partitions to use for the categorical tree reduction.
        Only a single partition is used by default.
    out_path : str, optional
        Used for multi-GPU groupby output.  Root directory where
        groupby statistics will be written out in parquet format.
    """

    def __init__(
        self,
        cat_names=None,
        cont_names=None,
        stats=["count"],
        limit_frac=0.5,
        gpu_mem_util_limit=0.5,
        gpu_mem_trans_use=0.5,
        columns=None,
        order_column_name="order-nvtabular",
        split_out=None,
        out_path=None,
        on_host=True,
    ):
        super(GroupByMoments, self).__init__(columns)
        self.cat_names = cat_names
        self.cont_names = cont_names
        self.stats = stats
        self.limit_frac = limit_frac
        self.gpu_mem_util_limit = gpu_mem_util_limit
        self.gpu_mem_trans_use = gpu_mem_trans_use
        self.order_column_name = order_column_name
        self.moments = {}
        self.categories = {}
        self.out_path = out_path or "./"
        self.split_out = split_out
        self.on_host = on_host

    def stat_logic(self, ddf, columns_ctx, input_cols, target_cols):
        cols = self.get_columns(columns_ctx, input_cols, target_cols)

        supported_ops = ["count", "sum", "mean", "std", "var"]
        for op in self.stats:
            if op not in supported_ops:
                raise ValueError(op + " operation is not supported.")

        agg_cols = self.cont_names
        agg_list = self.stats
        dsk, key = nvt_cat._groupby_stats(
            ddf, cols, agg_cols, agg_list, self.out_path, 0, self.split_out, self.on_host
        )
        return Delayed(key, dsk)

    def finalize(self, dask_stats):
        for col in dask_stats:
            self.categories[col] = dask_stats[col]

    def registered_stats(self):
        return ["moments", "gb_categories"]

    def stats_collected(self):
        result = [("moments", self.moments), ("gb_categories", self.categories)]
        return result

    def clear(self):
        self.moments = {}
        self.categories = {}
        return


class GroupBy(DFOperator):
    """
    One of the ways to create new features is to calculate
    the basic statistics of the data that is grouped by a categorical
    feature. This operator groups the data by the given categorical
    feature(s) and calculates the std, variance, and sum of requested continuous
    features along with count of every group. Then, merges these new statistics
    with the data using the unique ids of categorical data.

    Although you can directly call methods of this class to
    transform your categorical features, it's typically used within a
    Workflow class.

    Parameters
    -----------
    cat_names : list of str
        names of the categorical columns
    cont_names : list of str
        names of the continuous columns
    stats : list of str, default ['count']
        count of groups = ['count']
        sum of cont_col = ['sum']
    limit_frac : float, default 0.5
        fraction of memory to use during unique id calculation.
    gpu_mem_util_limit : float, default 0.5
        GPU memory utilization limit during frequency based
        calculation. If limit is exceeded, unique ids are moved
        to host memory.
    gpu_mem_trans_use : float, default 0.5
        GPU memory utilization limit during transformation. How much
        GPU memory will be used during transformation is calculated
        using this parameter.
    columns :
    preprocessing : bool, default True
        Sets if this is a pre-processing operation or not
    replace : bool, default False
        This parameter is ignored
    order_column_name : str, default "order-nvtabular"
        a column name to be used to preserve the order of input data.
        cudf's merge function doesn't preserve the order of the data
        and this column name is used to create a column with integer
        values in ascending order.
    """

    default_in = CAT
    default_out = CAT

    def __init__(
        self,
        cat_names=None,
        cont_names=None,
        stats=["count"],
        limit_frac=0.5,
        gpu_mem_util_limit=0.5,
        gpu_mem_trans_use=0.5,
        columns=None,
        preprocessing=True,
        replace=False,
        order_column_name="order-nvtabular",
        split_out=None,
        cat_cache="host",
        out_path=None,
        on_host=True,
    ):
        super().__init__(columns=columns, preprocessing=preprocessing, replace=False)
        self.cat_names = cat_names
        self.cont_names = cont_names
        self.stats = stats
        self.order_column_name = order_column_name
        self.limit_frac = limit_frac
        self.gpu_mem_util_limit = gpu_mem_util_limit
        self.gpu_mem_trans_use = gpu_mem_trans_use
        self.split_out = split_out
        self.out_path = out_path
        self.on_host = on_host
        self.cat_cache = cat_cache
        if isinstance(self.cat_cache, str):
            self.cat_cache = {name: cat_cache for name in self.cat_names}

    @property
    def req_stats(self):
        return [
            GroupByMoments(
                cat_names=self.cat_names,
                cont_names=self.cont_names,
                stats=self.stats,
                limit_frac=self.limit_frac,
                gpu_mem_util_limit=self.gpu_mem_util_limit,
                gpu_mem_trans_use=self.gpu_mem_trans_use,
                order_column_name=self.order_column_name,
                split_out=self.split_out,
                out_path=self.out_path,
                on_host=self.on_host,
            )
        ]

    def op_logic(self, gdf: cudf.DataFrame, target_columns: list, stats_context=None):
        if self.cat_names is None:
            raise ValueError("cat_names cannot be None.")

        new_gdf = cudf.DataFrame()
        if stats_context["moments"]:
            for name in stats_context["moments"]:
                tran_gdf = stats_context["moments"][name].merge(gdf)
                new_gdf[tran_gdf.columns] = tran_gdf
        else:  # Dask-based version
            tmp = "__tmp__"  # Temporary column for sorting
            gdf[tmp] = cupy.arange(len(gdf), dtype="int32")
            for col, path in stats_context["gb_categories"].items():
                stat_gdf = nvt_cat._read_groupby_stat_df(path, col, self.cat_cache)
                tran_gdf = gdf[[col, tmp]].merge(stat_gdf, on=col, how="left")
                tran_gdf = tran_gdf.sort_values(tmp)
                tran_gdf.drop(columns=[col, tmp], inplace=True)
                new_cols = [c for c in tran_gdf.columns if c not in new_gdf.columns]
                new_gdf[new_cols] = tran_gdf[new_cols].reset_index(drop=True)
            gdf.drop(columns=[tmp], inplace=True)
        return new_gdf


class Categorify(DFOperator):
    """
    Most of the data set will contain categorical features,
    and these variables are typically stored as text values.
    Machine Learning algorithms don't support these text values.
    Categorify operation can be added to the workflow to
    transform categorical features into unique integer values.

    Although you can directly call methods of this class to
    transform your categorical features, it's typically used within a
    Workflow class.

    Parameters
    -----------
    use_frequency : bool
        freq
    freq_threshold : float
        threshold
    limit_frac : float, default 0.5
        fraction of memory to use during unique id calculation.
    gpu_mem_util_limit : float, default 0.5
        GPU memory utilization limit during frequency based
        calculation. If limit is exceeded, unique ids are moved
        to host memory.
    gpu_mem_trans_use : float, default 0.5
        GPU memory utilization limit during transformation. How much
        GPU memory will be used during transformation is calculated
        using this parameter.
    columns :
    preprocessing : bool, default True
        Sets if this is a pre-processing operation or not
    replace : bool, default True
        Replaces the transformed column with the original input
        if set Yes
    cat_names :
    """

    default_in = CAT
    default_out = CAT

    def __init__(
        self,
        use_frequency=False,
        freq_threshold=0,
        limit_frac=0.5,
        gpu_mem_util_limit=0.5,
        gpu_mem_trans_use=0.5,
        columns=None,
        preprocessing=True,
        replace=True,
        cat_names=None,
        out_path=None,
        split_out=None,
        na_sentinel=None,
        cat_cache="host",
        dtype=None,
        on_host=True,
    ):
        super().__init__(columns=columns, preprocessing=preprocessing, replace=replace)
        self.use_frequency = use_frequency
        self.freq_threshold = freq_threshold
        self.limit_frac = limit_frac
        self.gpu_mem_util_limit = gpu_mem_util_limit
        self.gpu_mem_trans_use = gpu_mem_trans_use
        self.cat_names = cat_names if cat_names else []
        self.out_path = out_path or "./"
        self.split_out = split_out
        self.na_sentinel = na_sentinel or 0
        self.dtype = dtype
        self.on_host = on_host
        self.cat_cache = cat_cache
        # Allow user to specify a single string value for all columns
        # E.g. cat_cache = "device"
        if isinstance(self.cat_cache, str):
            self.cat_cache = {name: cat_cache for name in self.cat_names}

    @property
    def req_stats(self):
        return [
            Encoder(
                use_frequency=self.use_frequency,
                freq_threshold=self.freq_threshold,
                limit_frac=self.limit_frac,
                gpu_mem_util_limit=self.gpu_mem_util_limit,
                gpu_mem_trans_use=self.gpu_mem_trans_use,
                out_path=self.out_path,
                split_out=self.split_out,
                on_host=self.on_host,
            )
        ]

    @annotate("Categorify_op", color="darkgreen", domain="nvt_python")
    def op_logic(self, gdf: cudf.DataFrame, target_columns: list, stats_context={}):
        cat_names = target_columns
        new_gdf = cudf.DataFrame()
        if not cat_names:
            return gdf
        cat_names = [name for name in cat_names if name in gdf.columns]
        new_cols = []
        for name in cat_names:
            new_col = f"{name}_{self._id}"
            new_cols.append(new_col)
            path = stats_context["categories"][name]
            new_gdf[new_col] = nvt_cat._encode(
                name,
                path,
                gdf,
                self.cat_cache,
                na_sentinel=self.na_sentinel,
                freq_threshold=self.freq_threshold,
            )
            if self.dtype:
                new_gdf[new_col] = new_gdf[new_col].astype(self.dtype, copy=False)
        return new_gdf


def get_embedding_order(cat_names):
    """ Returns a consistent sorder order for categorical variables

    Parameters
    -----------
    cat_names : list of str
        names of the categorical columns
    """
    return sorted(cat_names)


def get_embedding_size(encoders, cat_names):
    """ Returns a suggested size of embeddings based off cardinality of encoding categorical
    variables

    Parameters
    -----------
    encoders : dict
        The encoding statistics of the categorical variables (ie. from workflow.stats["categories"])
    cat_names : list of str
        names of the categorical columns
    """
    # sorted key required to ensure same sort occurs for all values
    ret_list = [(n, _emb_sz_rule(encoders[n])) for n in get_embedding_order(cat_names)]
    return ret_list


def get_embeddings(workflow):
    cols = get_embedding_order(workflow.columns_ctx["categorical"]["base"])
    return get_embeddings_dask(workflow.stats["categories"], cols) 


def get_embeddings_dask(paths, cat_names):
    embeddings = {}
    for col in sorted(cat_names):
        path = paths[col]
        num_rows, _, _ = cudf.io.read_parquet_metadata(path)
        embeddings[col] =  _emb_sz_rule(num_rows)
    return embeddings


def _emb_sz_rule(n_cat: int) -> int:
    return n_cat, int(min(16, round(1.6 * n_cat ** 0.56)))


class LambdaOp(TransformOperator):
    """
    Enables to call Methods to cudf.Series

    Parameters
    -----------
    op_name : str
        name of the operator column. It is used as a post_fix for the
        modified column names (if replace=False)
    f : lambda function
        defines the function executed on dataframe level, expectation is lambda col, gdf: ...
        col is the cudf.Series defined by the context
        gdf is the full cudf.DataFrame
    columns :
    preprocessing : bool, default True
        Sets if this is a pre-processing operation or not
    replace : bool, default True
        Replaces the transformed column with the original input
        if set Yes
    """

    default_in = ALL
    default_out = ALL

    def __init__(self, op_name, f, columns=None, preprocessing=True, replace=True):
        super().__init__(columns=columns, preprocessing=preprocessing, replace=replace)
        if op_name is None:
            raise ValueError("op_name cannot be None. It is required for naming the column.")
        if f is None:
            raise ValueError("f cannot be None. LambdaOp op applies f to dataframe")
        self.f = f
        self.op_name = op_name

    @property
    def _id(self):
        return str(self.op_name)

    @annotate("DFLambda_op", color="darkgreen", domain="nvt_python")
    def op_logic(self, gdf: cudf.DataFrame, target_columns: list, stats_context=None):
        new_gdf = cudf.DataFrame()
        for col in target_columns:
            new_gdf[col] = self.f(gdf[col], gdf)
        new_gdf.columns = [f"{col}_{self._id}" for col in new_gdf.columns]
        return new_gdf
