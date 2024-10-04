def onehot_labeling(df, col_name, values, label_name):

    if label_name not in df.columns:
        df[label_name] = 0

    series = df[col_name]

    for value in values:
        # Find the index of the closest time_rel in the seismic data
        closest_idx = (series - value).abs().idxmin()

        # Get the time_rel value of the closest row
        closest_val = df.loc[closest_idx][col_name]

        # Set the label for the closest time
        if value >= closest_val:
            df.loc[closest_idx, label_name] = 1
        else:
            df.loc[closest_idx - 1, label_name] = 1

    return df

def calculate_confusion_matrix(predictions, targets):
    """
    Calculates the confusion matrix (TP, FP, TN, FN) for binary classification.

    Args:
        predictions (torch.Tensor): Predicted class labels.
        targets (torch.Tensor): Ground truth class labels.

    Returns:
        TP (int): True positives.
        FP (int): False positives.
        TN (int): True negatives.
        FN (int): False negatives.
    """
    TP = ((predictions == 1) & (targets == 1)).sum().item()
    FP = ((predictions == 1) & (targets == 0)).sum().item()
    TN = ((predictions == 0) & (targets == 0)).sum().item()
    FN = ((predictions == 0) & (targets == 1)).sum().item()

    return TP, FP, TN, FN
