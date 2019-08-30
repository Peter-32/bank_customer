import warnings
warnings.filterwarnings("ignore")

def get_score(full_pipeline, train, train_y):
    scores = cross_val_score(estimator=full_pipeline, X=train, y=train_y,
                             cv=10, n_jobs=1)
    print('CV accuracy scores: %s' % scores)
    print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))
