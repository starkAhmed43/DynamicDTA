from sklearn.model_selection import KFold

def get_kfold_s1_splits(df, n_splits=5, random_state=42):
    """
    Generate K-Fold splits for the given DataFrame.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the data.
    n_splits (int): Number of splits for K-Fold.
    random_state (int): Random state for reproducibility.

    Returns:
    list: A list of tuples containing train and test indices for each fold.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    return [(train_index, test_index) for train_index, test_index in kf.split(df)][:1]

def get_kfold_s2_splits(df, n_splits=5, random_state=42):
    """
    Generate K-Fold splits for S2: test drugs are unseen, test targets are seen.

    Parameters:
    df (pd.DataFrame): The DataFrame containing at least 'smiles' and 'sequence' columns.
    n_splits (int): Number of splits for K-Fold.
    random_state (int): Random state for reproducibility.

    Returns:
    list: A list of (train_indices, test_indices) tuples for each fold.
    """
    unique_drugs = df['smiles'].unique()
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    splits = []
    for train_drug_idx, test_drug_idx in kf.split(unique_drugs):
        train_drugs = unique_drugs[train_drug_idx]
        test_drugs = unique_drugs[test_drug_idx]

        train_mask = df['smiles'].isin(train_drugs)
        test_mask = df['smiles'].isin(test_drugs)

        # S2: test targets must be seen in train
        train_targets = set(df.loc[train_mask, 'sequence'])
        test_mask = test_mask & df['sequence'].isin(train_targets)

        train_indices = df.index[train_mask].tolist()
        test_indices = df.index[test_mask].tolist()
        splits.append((train_indices, test_indices))
    return splits[:1]  # Return only the first split for S2

def get_kfold_s3_splits(df, n_splits=5, random_state=42):
    """
    Generate K-Fold splits for S3: test targets are unseen, test drugs are seen.

    Parameters:
    df (pd.DataFrame): The DataFrame containing at least 'smiles' and 'sequence' columns.
    n_splits (int): Number of splits for K-Fold.
    random_state (int): Random state for reproducibility.

    Returns:
    list: A list of (train_indices, test_indices) tuples for each fold.
    """
    unique_targets = df['sequence'].unique()
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    splits = []
    for train_target_idx, test_target_idx in kf.split(unique_targets):
        train_targets = unique_targets[train_target_idx]
        test_targets = unique_targets[test_target_idx]

        train_mask = df['sequence'].isin(train_targets)
        test_mask = df['sequence'].isin(test_targets)

        # S3: test drugs must be seen in train
        train_drugs = set(df.loc[train_mask, 'smiles'])
        test_mask = test_mask & df['smiles'].isin(train_drugs)

        train_indices = df.index[train_mask].tolist()
        test_indices = df.index[test_mask].tolist()
        splits.append((train_indices, test_indices))
    return splits[:1]

def get_kfold_s4_splits(df, n_splits=5, random_state=42):
    """
    Generate K-Fold splits for S4: both test drugs and targets are unseen.

    Parameters:
    df (pd.DataFrame): The DataFrame containing at least 'smiles' and 'sequence' columns.
    n_splits (int): Number of splits for K-Fold.
    random_state (int): Random state for reproducibility.

    Returns:
    list: A list of (train_indices, test_indices) tuples for each fold.
    """
    unique_drugs = df['smiles'].unique()
    unique_targets = df['sequence'].unique()
    drug_kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    target_kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state + 1)  # Different seed

    drug_folds = list(drug_kf.split(unique_drugs))
    target_folds = list(target_kf.split(unique_targets))

    splits = []
    for i in range(n_splits):
        test_drugs = unique_drugs[drug_folds[i][1]]
        test_targets = unique_targets[target_folds[i][1]]

        test_mask = df['smiles'].isin(test_drugs) & df['sequence'].isin(test_targets)
        train_mask = ~test_mask

        train_indices = df.index[train_mask].tolist()
        test_indices = df.index[test_mask].tolist()
        splits.append((train_indices, test_indices))
    return splits[:1]