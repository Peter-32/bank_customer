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


# prob_y = cross_val_predict(estimator=full_pipeline, X=train, y=train_y,
#                          cv=10, n_jobs=1, method='predict_proba')







train_sizes, train_scores, test_scores = \
    learning_curve(estimator=full_pipeline,
                   X=train,
                   y=train_y,
                   train_sizes=np.linspace(0.1, 1.0, 10),
                   cv=custom_cv,
                   n_jobs=1)
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)
plt.plot(train_sizes, train_mean,
         color='blue', marker='o', markersize=5, label='training accuracy')
plt.fill_between(train_sizes, train_mean + train_std,
                 train_mean - train_std, alpha=0.15, color='blue')
plt.plot(train_sizes, test_mean,
         color='green', linestyle='--',
         marker='s', markersize=5,
         label='validation accuracy')
plt.fill_between(train_sizes, test_mean + test_std, test_mean - test_std,
                 alpha=0.15, color='green')
plt.grid()
plt.xlabel('Number of training samples')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.ylim([0.875, 0.925])
plt.show()
