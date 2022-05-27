def calc_f1_scores(actuals, preds, num_of_classes):
    if len(actuals) != len(preds):
        raise "Actuals array should be the same length as preds array."

    # True negatives aren't needed for f1-scores
    tps = [0] * num_of_classes
    fps = [0] * num_of_classes
    fns = [0] * num_of_classes

    for i in range(len(actuals)):
        actual = actuals[i]
        pred = preds[i]

        if actual >= num_of_classes or pred >= num_of_classes:
            raise "A class number in the actuals/preds array exceeded the number of classes given."

        if actual == pred:
            # We got it right, so give this class a true positive.
            tps[actual] = tps[actual] + 1
        elif actual != pred:
            # We go it wrong, so the actual class has a false negative and the
            # predicted class has a false positive.
            fns[actual] = fns[actual] + 1
            fps[pred] = fps[pred] + 1
    
    return [calc_f1_score(tps[i], fps[i], fns[i]) for i in range(num_of_classes)]
        
    
def calc_f1_score(tp, fp, fn):
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    if precision + recall == 0:
        return 0
    return 2 * precision * recall / (precision + recall)