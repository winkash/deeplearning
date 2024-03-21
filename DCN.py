from datetime import timedelta, date
import os
from typing import List

import tensorflow as tf
import tensorflow_recommenders as tfrs
from tensorflow.python.ops.math_ops import bucketize
from numpy import np

from pyspark.sql import SparkSession

time_features = ["impression_ts"]
id_features = ["employer", "freelancer_id"]
int_features = [
    'page_number',
    'position',
    'fl_profile_complete_percentage',
    'fl_invite_cnt_accepted', 'fl_invite_cnt', 'fl_assignment_cnt',
    'fl_bid_cnt_recommended', 'fl_bid_cnt',
]
float_features = []

fl_numerical_features = [
    'fl_profile_complete_percentage',
    'fl_profile_hourly_rate',
    'fl_job_success_score',
    'fl_freelancer_nps',
    'fl_freelancer_nss',
    'fl_freelancer_private_feedback',
    'fl_agency_cnt',
    'fl_invite_cnt_accepted',
    'fl_invite_cnt',
    'fl_assignment_cnt',
    'fl_bid_cnt_recommended',
    'fl_bid_cnt',
    'fl_assignment_cnt_active_billed',
    'fl_assignment_cnt_active',
    'fl_gsv_total',
    'fl_gsv_total_current_week',
    'fl_gsv_total_current_month',
    'fl_gsv_total_current_quarter',
    'fl_gsv_total_last_month',
    'fl_gsv_total_last_quarter',
    'fl_gsv_total_last_12_months',
    'fl_hours_worked_current_week',
    'fl_hours_worked',
    'fl_gsv_total_last_week',
    'fl_hours_worked_last_quarter',
    'fl_hours_worked_last_week',
    'fl_hours_worked_current_quarter',
    'fl_earning_total',
    'fl_hours_worked_last_month',
    'fl_hours_worked_last_12_months',
    'fl_earning_total_last_week',
    'fl_hours_worked_current_month',
    'fl_earning_total_current_week',
    'fl_earning_total_current_quarter',
    'fl_earning_total_current_month',
    'fl_earning_total_last_quarter',
    'fl_earning_total_last_month',
    'fl_earning_total_last_12_months',
    'fl_earning_total_catalog',
    'fl_total_project_cnt'
]

fl_categorical_features = [
    'fl_default_agency_team_id',
    'fl_profile_access_override',
    'fl_reg_country_name',
    'fl_membership_tier',
    'fl_is_mobile_reg',
    'fl_reg_device_browser',
    'fl_reg_region_name',
    'fl_reg_device_os',
    'fl_city',
    'fl_state',
    'fl_timezone',
    'fl_is_location_verified',
    'fl_original_profile_access',
    'fl_lock_status',
    'fl_profile_tier',
    'fl_acquisition_channel',
    'fl_freelancer_quality_segment',
    'fl_profile_available_hours',
    'fl_top_rated_status',
    'fl_top_rated_plus_status',
    'fl_is_pro',
    'fl_is_open_to_contract_to_hire',
    'fl_is_ac_exclusive',
    'fl_first_vetted_l3_ontology_name',
    'fl_first_vetted_l1_ontology_name',
    'fl_has_reputation',
    'fl_s_evt_medium',
    'fl_s_evt_campaign',
    'fl_has_ptc_engagement',
    'fl_is_enterprise_6m',
    'fl_fivs_status',
    'fl_current_payment_plan']

cl_numerical_features = [
    'cl_business_entity_employee_count',
    'cl_cancelled_post_cnt',
    'cl_post_cnt',
    'cl_filled_post_cnt',
    'cl_open_post_cnt',
    'cl_assignment_cnt',
    'cl_assignment_cnt_active',
    'cl_parent_assignment_cnt',
    'cl_invite_cnt',
    'cl_reg_to_post_days',
    'cl_reg_to_invite_days',
    'cl_hire_to_spend_days',
    'cl_reg_to_hire_days',
    'cl_post_to_invite_avg_days',
    'cl_post_to_hire_avg_days',
    'cl_hiring_manager_cnt_billing',
    'cl_hiring_manager_cnt',
    'cl_spend_total_current_quarter',
    'cl_spend_total_last_quarter',
    'cl_hiring_manager_gsv_avg',
    'cl_spend_total',
    'cl_spend_total_current_month',
    'cl_spend_total_last_month',
    'cl_spend_total_last_week',
    'cl_spend_total_current_week',
    'cl_spend_total_last_30d',
    'cl_spend_total_last_180d',
    'cl_spend_total_last_60d',
    'cl_spend_total_last_90d',
    'cl_spend_total_hr',
    'cl_spend_total_last_365d',
    'cl_spend_hr_365d',
    'cl_spend_total_fp',
    'cl_spend_fp_365d',
    'cl_spend_total_catalog',
]

cl_categorical_features = [
    'cl_id',
    'cl_agency_id',
    'cl_company_size_tier',
    'cl_is_ach',
    'cl_membership_type',
    'cl_premium_services_type',
    'cl_premium_services_manager_name',
    'cl_client_use_case_segment',
    'cl_client_market_segment',
    'cl_client_class',
    'cl_client_market_subsegment',
    'cl_nps_score',
    'cl_cpf_score',
    'cl_user_id',
    'cl_platform',
    'cl_timezone',
    'cl_acquisition_sub_channel',
    'cl_acquisition_channel',
    'cl_client_lifecycle_phase',
    'cl_city',
    'cl_longitude',
    'cl_latitude',
    'cl_state',
    'cl_zip',
    'cl_country_name',
    'cl_is_mobile_reg',
    'cl_team_id',
    'cl_region_name',
    'cl_reg_flow',
    'cl_client_type',
    'cl_compliance_specialist_name',
    'cl_email_domain',
    'cl_has_10k_relationship',
    'cl_business_entity_size_segment',
    'cl_projected_client_quality_segment',
    'cl_current_payment_plan'
]
str_features = id_features + fl_categorical_features + cl_categorical_features
fl_categorical_features = ["fl_reg_country_name"]
targets = ["clicked"]

data_dir = "path to train_test dir"
if not os.path.exists(data_dir):
    os.makedirs(data_dir)
data_path = os.path.join(data_dir, "raw_data.parquet")
train_data_path = os.path.join(data_dir, "train_data.parquet")
valid_data_path = os.path.join(data_dir, "valid_data.parquet")
test_data_path = os.path.join(data_dir, "test_data.parquet")
df_ = SparkSession.read.parquet(data_path).limit(1000000).toPandas()
df = df_.iloc[:100000, :].copy()
features = id_features + int_features + fl_categorical_features + cl_categorical_features + target


class Bucketizer(tf.keras.layers.layer):
    def __init__(self, buckets: List[float], **kwargs):
        super.__init__(**kwargs)
        self.buckets = buckets

    def call(self, x: tf.Tensor):
        return bucketize(x, boundaries=self.buckets)


class DCN(tfrs.Model):
    def __init__(self, use_cross_layer, deep_layer_sizes, target, projection_dim=None):
        self.embedding_dimenstion = 32
        self._all_features = str_features + int_features
        self._float_features = float_features
        self._target = target
        self._vocabularies = {}

        data = tf.data.Dataset.from_tensor_slices(dict(df.loc[:, features + target]))
        for i, feature_name in enumerate(features):
            print(i + 1, feature_name)
            vocab = data.batch(100_000).map(lambda x: x[feature_name])
            self._vocabularies[feature_name] = np.unique(np.concatenate(list(vocab)))

        for feature_name in str_features:
            vocabulary = self._vocabularies[feature_name]
            self._embeddings[feature_name] = tf.keras.Sequential(
                [tf.keras.layers.StringLookup(vocabulary=vocabulary, mask_token=None),
                 tf.keras.layeers.Embedding(len(vocabulary) + 1, self.embedding_dimenstion)])
        for feature_name in int_features:
            vocabulary = self._vocabularies[feature_name]
            self._embeddings[feature_name] = tf.keras.Sequential(
                tf.keras.layers.IntegerLookup(vocabulary=vocabulary, mask_token=None),
                tf.keras.layers.Embedding(len(vocabulary) + 1, self.embedding_dimenstion))

        if use_cross_layer:
            self._cross_layer = tfrs.layers.dcn.Cross(projection_dim=projection_dim,
                                                      kernel_initializer="glorot_uniform")
        else:
            self._cross_layer = None

        self._deep_layers = [tf.keras.layers.Dense(layer_size, activation="relu") for layer_size in deep_layer_sizes]
        self._logit_layer = tf.keras.layers.Dense(1)
        self.task = tfrs.tasks.Ranking(loss=tf.keras.losses.BinaryCrossentropy(), metrics=[tf.keras.metrics.AUC(name="AUC")])


    def call(self, features):
        embeddings = []
        float_feature_list= []
        for feature_name in self._all_features:
            embedding_fn = self._embeddings[feature_name]
            embeddings.append(embedding_fn(features[feature_name]))
        x = tf.concat(embeddings, axis=1)

        # Build Cross network
        if self._cross_layer is not None:
            x = self._cross_layer(x)
        # Build Deep network
        for deep_layer in self._deep_layers:
            x = deep_layer(x)

    def compute_loss(self, features, training=False):
        labels = features.pop(self._target)
        scores = self(features)
        return self.task(labels=labels, predictions=scores)


def run_models(use_crosslayer, deep_layer_sizes, projection_dim=None, num_runs=5, target="clicked"):
    models = []
    AUCs = []
    epochs = 100
    learning_rate = 0.4

    train = df[df["date"] <= date(2023, 5, 25)]
    test = df[df["date"] > date(2023, 5, 25)]
    train_df = tf.data.Dataset.from_tensor_slices(dict(train.loc[:, features + target]))
    test_df = tf.data.Dataset.from_tensor_slices(dict(test.loc[:, features + target]))
    cached_train = train_df.shuffle(100_000).batch(512).cache()
    cached_test = test_df.batch(256).cache()

    for run in range(num_runs):
        model = DCN(use_crosslayer, deep_layer_sizes, target, projection_dim)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate))
        models.append(model)
        model.fit(cached_train, epochs=epochs, veerbose=False)
        metrics = model.evaluate(cached_test, return_dict=True)
        AUCs.append(metrics['AUC'])

    mean, stdv = np.average(AUCs), np.std(AUCs)
    return {"model": models, "mean": mean, "stdv": stdv}


# We first train a DCN model with a stacked structure, that is, the inputs are fed to a cross network followed by a
# deep network.
dcn_result = run_models(use_crosslayer=True, deep_layer_sizes=[192, 192], num_runs=5, target=targets[0])
#  To reduce the training and serving cost, we leverage low-rank techniques to approximate the DCN weight matrices.
#  The rank is passed in through argument projection_dim; a smaller projection_dim results in a lower cost.
#  Note that projection_dim needs to be smaller than (input size)/2 to reduce the cost. In practice, we've
#  observed using low-rank DCN with rank (input size)/4 consistently preserved the accuracy of a full-rank DCN.
dcn_lr_result = run_models(use_crosslayer=True, deep_layer_sizes=[192,192,192], projection_dim=20, target=targets[0])
# We train a same-sized DNN model as a reference
dnn_result = run_models(use_crosslayer=False, deep_layer_sizes=[192, 192, 192])
