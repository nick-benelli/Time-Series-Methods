from sklearn.model_selection import TimeSeriesSplit


def k_folds_split(time_series, n_splits, print_results=False):
    tss = TimeSeriesSplit(n_splits=n_splits)
    print(tss)

    train_idx_list = []
    val_idx_list = []
    for i, (train_index, test_index) in enumerate(tss.split(time_series)):
        if print_results:
            print(f"Fold {i}:")
            print(f"  Train: index={train_index.shape}")
            print(f"  Test:  index={test_index.shape}")
        train_idx_list.append(train_index)
        val_idx_list.append(test_index)

    if print_results:
        print("Trian Val Idx Lengths:", len(train_idx_list), len(val_idx_list))

    return {"train": train_idx_list, "test": val_idx_list}
