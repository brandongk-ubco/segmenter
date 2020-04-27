def parse_fold(fold_string):
    fold = fold_string.split("b")[0]
    fold = int(fold.replace("fold", "")) if fold != "all" else None
    boost_fold = int(fold_string.split("b")[1]) if "b" in fold_string else None
    return fold, boost_fold
