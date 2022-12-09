"""
Import the necessary packages
"""
import numpy as np
import torch
import pandas as pd
from sklearn import metrics
from torch.nn import functional as F
import io
import tensorflow as tf
from sklearn.preprocessing import label_binarize
import itertools
import scikitplot as skplt
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('seaborn')



def disable_dropout(model):
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.train(False)


def merge_result(id_list, label_list, score_list, method):
    """Merge predicted results of all bags for each patient"""

    assert method in ["max", "mean"]
    merge_method = np.max if method == "max" else np.mean

    df = pd.DataFrame()
    df["id"] = id_list
    df["label"] = label_list
    df["score"] = score_list
    df = df.groupby(by=["id", "label"])["score"].apply(list).reset_index()
    df["bag_num"] = df["score"].apply(len)
    df["score"] = df["score"].apply(merge_method, args=(0,))

    return df["id"].tolist(), df["label"].tolist(), df["score"].tolist(), df["bag_num"].tolist()


def compute_confusion_matrix(label_list, predicted_label_list, num_classes=2):
    """Compute the confusion matrix"""
    label_array = np.array(label_list)
    predicted_label_array = np.array(predicted_label_list)
    confusion_matrix = np.bincount(num_classes * label_array + predicted_label_array, minlength=num_classes**2).reshape((num_classes, num_classes))

    return confusion_matrix

def compute_cm(label_list, predicted_label_list):
    """Compute the confusion matrix
        
        Parameters
        ----------
        label_list: np.array, 
        predicted_label_list: np.array

    """
    confusion_matrix = compute_confusion_matrix(label_list, predicted_label_list)
    return confusion_matrix

def compute_metrics(label_list, predicted_label_list):
    """ Compute the evaluation metrices: accuracy, sensitivity, specificity, PPV, NPV, precision, recall, F1-score"""
    confusion_matrix = compute_confusion_matrix(label_list, predicted_label_list)
    tn, fp, fn, tp = confusion_matrix.flatten()

    acc = (tn + tp) / (tn + fp + fn + tp)
    sens = tp / (tp + fn)
    spec = tn / (tn + fp)
    ppv = tp / (tp + fp)
    npv = tn / (tn + fn)

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * (precision * recall) / (precision + recall)

    return {"acc": acc, "sens": sens, "spec": spec, "ppv": ppv, "npv": npv, "f1": f1}


def plot_roc_curve(y_test, y_pred, classes, my_title):
    """
    Function to plot the multi-class ROC curve.

        Parameters
        ----------
        y_true: np.array, 
        y_pred: np.array,
        classes: np.array,
        my_title: String

        Returns
        -------
        Plot of the multi-class ROC curve and display ROC area for each class. 
    """
    n_classes = len(classes)
    y_test=np.array(y_test)
    y_pred=np.array(y_pred)
     # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    thresholds = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], thresholds[i] = metrics.roc_curve(y_test[:, i], y_pred[:, i], drop_intermediate=False)
        roc_auc[i] = metrics.auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = metrics.roc_curve(y_test.ravel(), y_pred.ravel())
    roc_auc["micro"] = metrics.auc(fpr["micro"], tpr["micro"])

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = metrics.auc(fpr["macro"], tpr["macro"])

     # Plot all ROC curves
     #plt.figure(figsize=(10,5))
    plt.figure(dpi=600)
    plt.plot(fpr["micro"], tpr["micro"],
      label="micro-average ROC curve (area = {0:0.3f})".format(roc_auc["micro"]),
      color="deeppink", linestyle=":", linewidth=1,)

    plt.plot(fpr["macro"], tpr["macro"],label="macro-average ROC curve (area = {0:0.3f})".format(roc_auc["macro"]), color="navy", linestyle=":", linewidth=1,)

    colors = itertools.cycle(["aqua", "darkorange", "darkgreen", "yellow", "blue"])
    for i, color, name in zip(range(n_classes), colors, classes):
        plt.plot(fpr[i], tpr[i], color=color, lw=1, label="ROC curve of class {0} (area = {1:0.3f})".format(name, roc_auc[i]),)
        
     # plotting 

    plt.plot([0, 1], [0, 1], "k--", lw=2)
    plt.title(my_title)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive rate')
    plt.legend(loc='best', prop={'size': 6})
    buf=io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image=tf.image.decode_png(buf.getvalue(), channels=4)
    image = tf.expand_dims(image, 0)
    return image


def plot_multiple_roc(label_list, score_list):
    """
    Function to plot the multi-class ROC curve.

        Parameters
        ----------
        label_list: np.array, 
        score_list: np.array,

        Returns
        -------
        Plot of the multi-class ROC curve. 
    """
    skplt.metrics.plot_roc_curve(label_list, score_list)
    plt.show()
    buf=io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image=tf.image.decode_png(buf.getvalue(), channels=4)
    image = tf.expand_dims(image, 0)
    return image

def get_best_metrics_and_threshold(label_list, score_list):
    best_metric_dict = {"acc": 0, "sens": 0, "spec": 0, "ppv": 0, "npv": 0, "f1": 0}
    best_threshold = 0

    # search the best metrcis with F1 score (the greater is better)
    for threshold in np.linspace(0, 1, 1000):
        metric_dict, _ = compute_metrics_by_threshold(label_list, score_list, threshold)
        if metric_dict["f1"] > best_metric_dict["f1"]:
            best_metric_dict = metric_dict
            best_threshold = threshold
    best_metric_dict["auc"] = compute_auc(label_list, score_list)

    return best_metric_dict, best_threshold


def compute_metrics_by_threshold(label_list, score_list, threshold):
    # bag will be predicted as the positive (label is 1) when the score is greater than threshold
    predicted_label_list = [1 if score >= threshold else 0 for score in score_list]
    metric_dict = compute_metrics(label_list, predicted_label_list)
    metric_dict["auc"] = compute_auc(label_list, score_list)

    return metric_dict, threshold


def compute_auc(label_list, score_list, multi_class="raise"):
    try:
        # set "multi_class" for computing auc of 2 classes ("raise") and multiple classes ("ovr" or "ovo"), https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html
        auc = metrics.roc_auc_score(label_list, score_list, multi_class=multi_class)
    except ValueError:
        auc = 0  # handle error when there is only 1 classes in "label_list"
    return auc

def plot_auc(label_list, score_list, task_type, epoch, color):
    """
    Compute and plot the AUC score per epoch.
    
    Args:
        label_list: ndarray of shape (n_samples,)
        score_list: ndarray of shape (n_samples,)
    """
    fpr, tpr, thresholds = metrics.roc_curve(label_list, score_list)
    plt.figure()
    plt.plot(fpr, tpr, linestyle='--', color=color, label='{}_{}'.format(task_type,epoch))
    plt.xlabel("False Positive Rate")
    plt.ylabel('True Positive Rate')
    plt.legend()
    buf=io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image=tf.image.decode_png(buf.getvalue(), channels=4)
    # Add the batch dimension
    image = tf.expand_dims(image, 0)
    return image

def plot_confusion_matrix(cm, class_names):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.

    Args:
      cm (array, shape = [n, n]): a confusion matrix of integer classes
      class_names (array, shape = [n]): String names of the integer classes
    """
    figure = plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Compute the labels from the normalized confusion matrix.
    labels = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > threshold else "black"
        plt.text(j, i, labels[i, j], horizontalalignment="center", color=color)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    buf=io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image=tf.image.decode_png(buf.getvalue(), channels=4)
    # Add the batch dimension
    image = tf.expand_dims(image, 0)
    return image

def plot_confusion_matrix_seaborn(cm, class_names, auc):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.

    Args:
      cm (array, shape = [n, n]): a confusion matrix of integer classes
      class_names (array, shape = [n]): String names of the integer classes
    """
    plt.figure(figsize=(9,9))
    sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r');
    plt.ylabel('Actual label');
    plt.xlabel('Predicted label');
    all_sample_title = 'AUC score {0}'.format(auc)
    plt.title(all_sample_title, size = 15);
    
    buf=io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image=tf.image.decode_png(buf.getvalue(), channels=4)
    # Add the batch dimension
    image = tf.expand_dims(image, 0)
    return image

def save_checkpoint(model, save_path):
    torch.save(model.state_dict(), save_path)


def train_val_test_binary_class(task_type, epoch, model, data_loader, optimizer, recoder, writer, merge_method):
    """
    Train and evalute the binary classification task 
    
    Paramters
        task_type: string ["train", "val", "test"]
        epoch: integer 
        model: MILNetWithClinicalData()
    """
    total_loss = 0
    label_list = []
    score_list = []  # [score_bag_0, score_bag_1, ..., score_bag_n]
    id_list = []
    patch_path_list = []
    attention_value_list = []  # [attention_00, attention_01, ..., attention_10, attention_11, ..., attention_n0, attention_n1, ...]

    if task_type == "train":
        model.train()
        for index, item in enumerate(data_loader, start=1):
            print(f"\repoch: {epoch}, {task_type}, [{index}/{len(data_loader)}]", end="")
            bag_tensor, label = item["bag_tensor"].cuda(), item["label"].cuda()
            clinical_data = item["clinical_data"][0].cuda() if "clinical_data" in item else None

            optimizer.zero_grad()
            output, attention_value = model(bag_tensor, clinical_data)
            loss = F.cross_entropy(output, label)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            id_list.append(item["patient_id"][0])
            label_list.append(label.item())
            score = F.softmax(output, dim=-1).squeeze(dim=0)[1].cpu().item()  # use the predicted positive probability as score
            score_list.append(score)
            patch_path_list.extend([p[0] for p in item["patch_paths"]])
            attention_value_list.extend(attention_value[0].cpu().tolist())
    else:
        # evaluation
        disable_dropout(model)
        with torch.no_grad():
            for index, item in enumerate(data_loader, start=1):
                print(f"\repoch: {epoch}, {task_type}, [{index}/{len(data_loader)}]", end="")
                bag_tensor, label = item["bag_tensor"].cuda(), item["label"].cuda()
                clinical_data = item["clinical_data"][0].cuda() if "clinical_data" in item else None

                output, attention_value = model(bag_tensor, clinical_data)
                loss = F.cross_entropy(output, label)
                total_loss += loss.item()

                id_list.append(item["patient_id"][0])
                label_list.append(label.item())
                score = F.softmax(output, dim=-1).squeeze(dim=0)[1].cpu().item()  # use the predicted positive probability as score
                score_list.append(score)
                patch_path_list.extend([p[0] for p in item["patch_paths"]])
                attention_value_list.extend(attention_value[0].cpu().tolist())

    recoder.record_attention_value(patch_path_list, attention_value_list, epoch)
    if merge_method != "not_use":
        id_list, label_list, score_list, bag_num_list = merge_result(id_list, label_list, score_list, merge_method)
        recoder.record_score_value(id_list, label_list, bag_num_list, score_list, epoch)

    average_loss = total_loss / len(data_loader)
    metrics_dict, threshold = compute_metrics_by_threshold(label_list, score_list, 0.5)
    predicted_label_list = [1 if score >= threshold else 0 for score in score_list]
    confusion_matrix = compute_cm(label_list, predicted_label_list)
    print(f"confusion matrix: {confusion_matrix}")
    
    print(
        f"\repoch: {epoch}, {task_type}, loss: {average_loss:.3f}, threshold: {threshold}, auc: {metrics_dict['auc']:.3f}, acc: {metrics_dict['acc']:.3f}, sens: {metrics_dict['sens']:.3f}, spec: {metrics_dict['spec']:.3f}, ppv: {metrics_dict['ppv']:.3f}, npv: {metrics_dict['npv']:.3f}, f1: {metrics_dict['f1']:.3f}"
    )

    writer.add_scalars("comparison/loss", {f"{task_type}_loss": average_loss}, epoch)
    writer.add_scalars("comparison/auc", {f"{task_type}_auc": metrics_dict["auc"]}, epoch)
    writer.add_scalars(f"metrics/{task_type}", metrics_dict, epoch)

    if task_type == "train":
        plot_cm=plot_confusion_matrix_seaborn(confusion_matrix, class_names=['N0', 'N+(>0)'], auc=metrics_dict['auc'])
        file_writer = tf.summary.create_file_writer('./plots/vggbn_13/runs_cm_0/')
        with file_writer.as_default():
            tf.summary.image("CM_train", plot_cm, epoch)
    elif task_type == "val":
        plot_cm=plot_confusion_matrix_seaborn(confusion_matrix, class_names=['N0', 'N+(>0)'], auc=metrics_dict['auc'])
        file_writer = tf.summary.create_file_writer('./plots/vggbn_13/runs_cm_0/')
        with file_writer.as_default():
            tf.summary.image("CM_val", plot_cm, epoch)
    elif task_type == "test":
        plot_cm=plot_confusion_matrix_seaborn(confusion_matrix, class_names=['N0', 'N+(>0)'], auc=metrics_dict['auc'])
        file_writer = tf.summary.create_file_writer('./plots/vggbn_13/runs_cm_0/')
        with file_writer.as_default():
            tf.summary.image("CM_test", plot_cm, epoch)
    
    if task_type == "train":
        plot_buf=plot_auc(label_list, score_list, task_type, epoch, color='green')
        file_writer = tf.summary.create_file_writer('./plots/vggbn_13/runs_roc_0/')
        with file_writer.as_default():
            tf.summary.image("roc_train", plot_buf, epoch)
    elif task_type == "val":
        plot_buf=plot_auc(label_list, score_list, task_type, epoch, color='blue')
        file_writer = tf.summary.create_file_writer('./plots/vggbn_13/runs_roc_0/')
        with file_writer.as_default():
            tf.summary.image("roc_val", plot_buf, epoch)
    elif task_type == "test":
        plot_buf=plot_auc(label_list, score_list, task_type, epoch, color='red')
        file_writer = tf.summary.create_file_writer('./plots/vggbn_13/runs_roc_0/')
        with file_writer.as_default():
            tf.summary.image("roc_test", plot_buf, epoch)
            
    return metrics_dict["auc"]

def train_val_test_multi_class(task_type, epoch, model, data_loader, optimizer, recoder, writer, merge_method):
    """
    Train and evalute the multi-class classification task 
    
    Paramters
        task_type: string ["train", "val", "test"]
        epoch: integer 
        model: MILNetWithClinicalData()
        data_loader: [train_loader, val_loader, test_loader]
        optimizer: ["Adam", "SGD"]
    """
    total_loss = 0
    label_list = []
    score_list = []  # [[score_0_bag_0, score_1_bag_0, score_2_bag_0], [score_0_bag_1, score_1_bag_1, score_2_bag_1], ...]
    id_list = []
    patch_path_list = []
    attention_value_list = []  # [attention_00, attention_01, ..., attention_10, attention_11, ..., attention_n0, attention_n1, ...]

    if task_type == "train":
        model.train()
        for index, item in enumerate(data_loader, start=1):
            print(f"\repoch: {epoch}, {task_type}, [{index}/{len(data_loader)}]", end="")
            bag_tensor, label = item["bag_tensor"].cuda(), item["label"].cuda()
            clinical_data = item["clinical_data"][0].cuda() if "clinical_data" in item else None

            optimizer.zero_grad()
            output, attention_value = model(bag_tensor, clinical_data)
            loss = F.cross_entropy(output, label)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            id_list.append(item["patient_id"][0])
            label_list.append(label.item())
            score = F.softmax(output, dim=-1).squeeze(dim=0).detach().cpu().numpy()
            score_list.append(score)
            patch_path_list.extend([p[0] for p in item["patch_paths"]])
            attention_value_list.extend(attention_value[0].cpu().tolist())
    else:
        # evaluation
        disable_dropout(model)
        with torch.no_grad():
            for index, item in enumerate(data_loader, start=1):
                print(f"\repoch: {epoch}, {task_type}, [{index}/{len(data_loader)}]", end="")
                bag_tensor, label = item["bag_tensor"].cuda(), item["label"].cuda()
                clinical_data = item["clinical_data"][0].cuda() if "clinical_data" in item else None

                output, attention_value = model(bag_tensor, clinical_data)
                loss = F.cross_entropy(output, label)
                total_loss += loss.item()

                id_list.append(item["patient_id"][0])
                label_list.append(label.item())
                score = F.softmax(output, dim=-1).squeeze(dim=0).detach().cpu().numpy()
                score_list.append(score)
                patch_path_list.extend([p[0] for p in item["patch_paths"]])
                attention_value_list.extend(attention_value[0].cpu().tolist())

    recoder.record_attention_value(patch_path_list, attention_value_list, epoch)
    if merge_method != "not_use":
        id_list, label_list, score_list, bag_num_list = merge_result(id_list, label_list, score_list, merge_method)
        recoder.record_score_value(id_list, label_list, bag_num_list, score_list, epoch)

    average_loss = total_loss / len(data_loader)
    # compute AUC of multiple classification with "ovr" setting, https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html
    auc = compute_auc(label_list, score_list, multi_class="ovr")
    label_predicted = np.argmax(np.asarray(score_list), axis=-1)
    confusion_matrix = metrics.confusion_matrix(label_list, label_predicted)

    print(f"\repoch: {epoch}, {task_type}, loss: {average_loss:.3f}, auc: {auc:.3f}")
    print(f"confusion matrix: {confusion_matrix}")
    writer.add_scalars("comparison/loss", {f"{task_type}_loss": average_loss}, epoch)
    writer.add_scalars("comparison/auc", {f"{task_type}_auc": auc}, epoch)

    
    if task_type == "train":
        plot_cm=plot_confusion_matrix_seaborn(confusion_matrix, class_names=['N0', 'N+(1-2)', 'N+(>2)'], auc=auc)
        file_writer = tf.summary.create_file_writer('./plots/vggbn_16/runs_cm_2/')
        with file_writer.as_default():
            tf.summary.image("CM_train", plot_cm, epoch)
    elif task_type == "val":
        plot_cm=plot_confusion_matrix_seaborn(confusion_matrix, class_names=['N0', 'N+(1-2)', 'N+(>2)'], auc=auc)
        file_writer = tf.summary.create_file_writer('./plots/vggbn_16/runs_cm_2/')
        with file_writer.as_default():
            tf.summary.image("CM_val", plot_cm, epoch)
    elif task_type == "test":
        plot_cm=plot_confusion_matrix_seaborn(confusion_matrix, class_names=['N0', 'N+(1-2)', 'N+(>2)'], auc=auc)
        file_writer = tf.summary.create_file_writer('./plots/vggbn_16/runs_cm_2/')
        with file_writer.as_default():
            tf.summary.image("CM_test", plot_cm, epoch)
     
    if task_type == "train":
        plot_roc_multi=plot_multiple_roc(label_list, score_list)
        file_writer = tf.summary.create_file_writer('./plots/vggbn_16/runs_roc_2/')
        with file_writer.as_default():
            tf.summary.image("AUC_train", plot_roc_multi, epoch)
    elif task_type == "val":
        plot_roc_multi=plot_multiple_roc(label_list, score_list)
        file_writer = tf.summary.create_file_writer('./plots/vggbn_16/runs_roc_2/')
        with file_writer.as_default():
            tf.summary.image("AUC_val", plot_roc_multi, epoch)
    elif task_type == "test":
        plot_roc_multi=plot_multiple_roc(label_list, score_list)
        file_writer = tf.summary.create_file_writer('./plots/vggbn_16/runs_roc_2/')
        with file_writer.as_default():
            tf.summary.image("AUC_test", plot_roc_multi, epoch)
    
    return auc
