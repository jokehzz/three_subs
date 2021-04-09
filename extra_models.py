from sklearn.ensemble import ExtraTreesClassifier

def extra_model1():
    max_depth = 10
    max_features = 0.1
    min_samples_leaf = 1
    min_samples_split = 2
    n_estimmators = 101
    model = ExtraTreesClassifier(n_estimators=n_estimmators,max_depth=max_depth,
                                 min_samples_leaf=min_samples_leaf,min_samples_split=min_samples_split,max_features=max_features)
    return model

def extra_model2():
    max_depth = 30
    max_features = 0.999
    min_samples_leaf = 4
    min_samples_split = 25
    n_estimators = 108
    model = ExtraTreesClassifier(n_estimators=n_estimators,max_depth=max_depth,min_samples_leaf=min_samples_leaf,
                                 min_samples_split=min_samples_split,max_features=max_features)
    return model

def extra_model3():
    max_depth = 27
    max_features = 0.8769
    min_samples_leaf = 5
    min_samples_split = 37
    n_estimators = 64
    model = ExtraTreesClassifier(n_estimators=n_estimators, max_depth=max_depth, min_samples_leaf=min_samples_leaf,
                                 min_samples_split=min_samples_split, max_features=max_features)
    return model

def extra_model4():
    max_depth = 26
    max_features = 0.1028
    min_samples_leaf = 29
    min_samples_split = 48
    n_estimators = 128
    model = ExtraTreesClassifier(n_estimators=n_estimators, max_depth=max_depth, min_samples_leaf=min_samples_leaf,
                                 min_samples_split=min_samples_split, max_features=max_features)
    return model

def extra_model5():
    max_depth = 30
    max_features = 0.999
    min_samples_leaf = 7
    min_samples_split = 8
    n_estimators = 127
    model = ExtraTreesClassifier(n_estimators=n_estimators, max_depth=max_depth, min_samples_leaf=min_samples_leaf,
                                 min_samples_split=min_samples_split, max_features=max_features)
    return model

def extra_model6():
    max_depth = 25
    max_features = 0.999
    min_samples_leaf = 8
    min_samples_split = 50
    n_estimators = 43
    model = ExtraTreesClassifier(n_estimators=n_estimators, max_depth=max_depth, min_samples_leaf=min_samples_leaf,
                                 min_samples_split=min_samples_split, max_features=max_features)
    return model

def extra_model7():
    max_depth = 25
    max_features = 0.7253
    min_samples_leaf = 40
    min_samples_split = 47
    n_estimators = 85
    model = ExtraTreesClassifier(n_estimators=n_estimators, max_depth=max_depth, min_samples_leaf=min_samples_leaf,
                                 min_samples_split=min_samples_split, max_features=max_features)
    return model

def extra_model8():
    max_depth = 29
    max_features = 0.7123
    min_samples_leaf = 10
    min_samples_split = 45
    n_estimators = 48
    model = ExtraTreesClassifier(n_estimators=n_estimators, max_depth=max_depth, min_samples_leaf=min_samples_leaf,
                                 min_samples_split=min_samples_split, max_features=max_features)
    return model

def extra_model9():
    max_depth = 26
    max_features = 0.9987
    min_samples_leaf = 9
    min_samples_split = 20
    n_estimators = 97
    model = ExtraTreesClassifier(n_estimators=n_estimators, max_depth=max_depth, min_samples_leaf=min_samples_leaf,
                                 min_samples_split=min_samples_split, max_features=max_features)
    return model

def extra_model10():
    max_depth = 18
    max_features = 0.9009
    min_samples_leaf = 5
    min_samples_split = 24
    n_estimators = 106
    model = ExtraTreesClassifier(n_estimators=n_estimators, max_depth=max_depth, min_samples_leaf=min_samples_leaf,
                                 min_samples_split=min_samples_split, max_features=max_features)
    return model

def extra_model11():
    max_depth = 22
    max_features = 0.999
    min_samples_leaf = 11
    min_samples_split = 26
    n_estimators = 77
    model = ExtraTreesClassifier(n_estimators=n_estimators, max_depth=max_depth, min_samples_leaf=min_samples_leaf,
                                 min_samples_split=min_samples_split, max_features=max_features)
    return model

def extra_model12():
    max_depth = 23
    max_features = 0.999
    min_samples_leaf = 9
    min_samples_split = 25
    n_estimators = 107
    model = ExtraTreesClassifier(n_estimators=n_estimators, max_depth=max_depth, min_samples_leaf=min_samples_leaf,
                                 min_samples_split=min_samples_split, max_features=max_features)
    return model

def main():
    pass

if __name__ == '__main__':
    main()