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

from .operator import Operator


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
        self.delim = None

    def _set_id(self, id_to_set):
        # only set one time
        if not self._id_set:
            self._id_set = id_to_set

    def _sanitized_id(self, delim=None):
        if self._id_set:
            self.delim = delim if delim else self.delim
            # should split on delim for added id for multi op support
            return self._id.split(delim)[0]

    def out_columns(self, tar_cols, extra_cols, delim):
        new_cols = []
        if not self.replace:
            new_cols = [f"{col}{delim}{self._id}" for col in tar_cols]
        return new_cols + tar_cols, extra_cols

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
            if "base" in columns_ctx["final"]["ctx"][input_cols]:
                columns_ctx["final"]["ctx"][input_cols].remove("base")
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
