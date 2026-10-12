from sklearn.metrics import precision_score,recall_score,f1_score,matthews_corrcoef,accuracy_score,confusion_matrix



def scores(y_test,y_pred):
    macro_precision = precision_score(y_test, y_pred, average='macro')
    macro_recall = recall_score(y_test, y_pred, average='macro')
    macro_f1 = f1_score(y_test, y_pred, average='macro')
    matthews_corr = matthews_corrcoef(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    return [macro_precision,macro_recall,macro_f1,matthews_corr,accuracy]



def AbsoluteTrue(y_hat, y):
    '''
    same
    '''

    n, m = y_hat.shape
    sorce_k = 0
    for v in range(n):
        if list(y_hat[v]) == list(y[v]):
            sorce_k += 1
    return sorce_k/n


def AbsoluteFalse(y_hat, y):
    '''
    hamming loss
    '''

    n, m = y_hat.shape

    sorce_k = 0
    for v in range(n):
        union = 0
        intersection = 0
        for h in range(m):
            if y_hat[v,h] == 1 or y[v,h] == 1:
                union += 1
            if y_hat[v,h] == 1 and y[v,h] == 1:
                intersection += 1
        sorce_k += (union-intersection)/m
    return sorce_k/n


def evaluate(y_hat, y, x_laber, y_laber):
    res = scores(x_laber,y_laber)
    macro_precision = res[0]
    macro_recall = res[1]
    macro_f1 = res[2]
    matthews_corr = res[3]
    accuracy = res[4]
    absolute_true = AbsoluteTrue(y_hat, y)
    absolute_false = AbsoluteFalse(y_hat, y)
    return macro_precision,macro_recall,macro_f1,matthews_corr,accuracy,absolute_true,absolute_false