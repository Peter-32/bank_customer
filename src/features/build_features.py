from numpy import c_, searchsorted
from sklearn.cluster import KMeans
from featexp import univariate_plotter
from pandas import DataFrame, get_dummies
from sklearn.preprocessing import KernelCenterer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Binarizer


def get_scaler_columns(sample_train):
    # Find boolean and object columns and ignore them
    bool_cols = [col for col in sample_train if sample_train[col].\
                 value_counts().index.isin([0,1]).all()]
    obj_cols = [col for col in sample_train if sample_train[col].\
                 dtype == object]
    ignore_cols = bool_cols + obj_cols

    # Columns to scale
    columns_to_scale = [x for x in sample_train.columns if x not in ignore_cols]

    # Dict(original column name, Dict(scaling_name, Series))
        # i.e. {"col1": {"standard_scaler": numpy.ndarray}}
    scaler_columns = {}
    for column in columns_to_scale:
        scaler_columns[column] = {}

    # Scalers (Remember to transform the final test dataset after each fit
    ss = StandardScaler()
    mms = MinMaxScaler()
    for column in columns_to_scale:
        ss = StandardScaler()
        scaler_columns[column]["standard_scaler"] = DataFrame(ss.\
                        fit_transform(sample_train[[column]].values), \
                    columns=[column + "__ss"])
        mms = MinMaxScaler()
        scaler_columns[column]["minmax_scaler"] = DataFrame(mms.\
                        fit_transform(sample_train[[column]].values), \
                    columns=[column + "__mms"])
        mms = MinMaxScaler()
        scaler_columns[column]["standard_minmax_scaler"] = \
        DataFrame(mms.fit_transform(scaler_columns[column]\
            ["standard_scaler"]), columns=[column + "__ss_mms"])


    # FeatExp mean target
    for column in columns_to_scale:
        binned_data_train = univariate_plotter(data=sample_train, \
                                                    target_col='y', \
                                                    feature=column, \
                                                    bins=10)
        temp_df = binned_data_train[[column, "y_mean"]].values
        X = temp_df[:, 1].astype(float).reshape(-1, 1)

        # KMeans
        try:
            kmeans = KMeans(n_clusters=3)
            kmeans.fit(X)
        except:
            try:
                kmeans = KMeans(n_clusters=2)
                kmeans.fit(X)
            except:
                kmeans = KMeans(n_clusters=1)
                kmeans.fit(X)

        # Bounds
        upper_bounds = []
        for val in temp_df:
            bound = val[0].split(",")
            lower_bound = float(bound[0][1:])
            upper_bound = float(bound[1][:-1].replace(" ", ""))
            upper_bounds.append(upper_bound)

        # Cluster Names
        cluster_names = []
        min_val = min(kmeans.cluster_centers_)
        max_val = max(kmeans.cluster_centers_)
        for x in kmeans.cluster_centers_:
            if min_val == max_val:
                cluster_names.append("medium")
            elif x == min_val:
                cluster_names.append("low")
            elif x == max(kmeans.cluster_centers_):
                cluster_names.append("high")
            else:
                cluster_names.append("medium")

        # Map cluster id to mean target value
        temp = sample_train[[column]].apply(lambda x: \
                     searchsorted(upper_bounds, x))


        # BROKEN !!!!!!
        len_temp_df = len(temp_df)
        scaler_columns[column]["featexp_mean_target"] = \
                    DataFrame(temp[column].apply(lambda x: temp_df[min(x,len_temp_df-1)][1]).\
                            values.reshape(-1, 1), \
                            columns=[column + "__featexp_mean_target"])


    # FeatExp cluster mean target

        # Map values to cluster id
        temp_df2 = sample_train[[column]].apply(lambda x: \
                              kmeans.labels_[searchsorted(upper_bounds, x)])

        # Map cluster id to mean target value
        scaler_columns[column]["featexp_cluster_mean_target"] = \
                DataFrame(temp_df2[column].\
                               apply(lambda x: kmeans.cluster_centers_[x][0]).\
                                     values.reshape(-1, 1), \
                             columns=[column + "__featexp_cluster_mean_target"])


    # FeatExp binarize clusters (high/medium/low)

        # Map values to cluster id
        temp_df2 = sample_train[[column]].apply(lambda x: \
                              kmeans.labels_[searchsorted(upper_bounds, x)])

        # Map cluster id to mean target value
        scaler_columns[column]["featexp_binarize_clusters"] = \
                get_dummies(temp_df2[column].\
                                   apply(lambda x: cluster_names[x]).values, \
                               prefix=column, \
                               prefix_sep='__featexp_', drop_first=True)


    # Binarizer
    binarizers = [("commission", 0.0),
                  ("margin", 0.0),
                  ("shipping", 0.0),
                  ("last_event_to_conversion_seconds", 758.0),
                  ("conversion_hour", 15),
                  ("revenue", 128.88),
                  ("steps_in_path", 3),
                  ("path_length_days", 1.122),
                  ("num_event_device_type", 1.0),
                  ("num_ms_ad_group_name", 1.0),
                  ("num_ms_ad_campaign_name", 1.0),
                  ("num_ms_ad_distribution_type", 0.0),
                  ("num_media_partners", 0.0),
                  ("num_media_sources", 1.0),
                  ("num_referring_urls", 1.0),
                  ("recent_tv_impressions", 88532465.0),
                  ("recent_tv_impressions_matching_geo_split_out_national", 1936394.0),
                  ("recent_tv_impressions_matching_geo_without_splitting_out_national", 136.0)
                  ]
    for column, threshold in binarizers:
        bin = Binarizer(threshold=threshold)
        scaler_columns[column]["binarizer"] = \
            DataFrame(bin.fit_transform(sample_train[[column]].values), \
                columns=[column + "__binarizer"])


    # Related groups
    related_groups = [["commission", "margin", "shipping", "revenue"],
                      ["last_event_to_conversion_seconds", "steps_in_path", "path_length_days", "conversion_hour"],
                      ["num_ms_ad_group_name", "num_ms_ad_group_name", "num_ms_ad_distribution_type"],
                      ["num_media_partners", "num_media_sources"]]

    # Kernel Centerer
    for group in related_groups:
        kc = KernelCenterer()
        temp_df = DataFrame(kc.fit_transform(sample_train[group].\
                         values), columns=group)
        for column in group:
            scaler_columns[column]["kernel_centerer"] = \
                temp_df[[column]]


    return scaler_columns
