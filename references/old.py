df["loan_unknown_and_housing_unknown"] = \
    df[["loan_unknown", "housing_unknown"]].\
        apply(lambda x: _get_loan_unknown_and_housing_unknown(*x), axis=1)
df.drop(["education_university.degree", "loan_unknown", "housing_unknown"], axis=1, inplace=True)


def _add_numeric_features(df):
    pass
    # Add/remove features






def _add_category_features(df):
    # df = pd.concat([df.select_dtypes(exclude="O"), \
    #        pd.get_dummies(df.select_dtypes(include="O"), drop_first=True)], axis=1)
    df = pd.get_dummies(df.select_dtypes(include="O"), drop_first=True)



    # # Prepare the data
    # _add_numeric_features(train); _add_category_features(train);
    # _add_numeric_features(test);  _add_category_features(test);
    # _remove_correlated_variables(train); _remove_correlated_variables(test)
    #
    # # Save the data
    # train.to_csv("../../data/interim/prepared_train.csv")
    # test.to_csv("../../data/interim/prepared_test.csv")
