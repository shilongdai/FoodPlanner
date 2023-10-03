import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import f1_score


def compute_accuracy(eval_pred):
    predictions_score, labels = eval_pred
    predictions = np.argmax(predictions_score, axis=1)
    balanced = balanced_accuracy_score(y_true=labels, y_pred=predictions)
    roc_auc = roc_auc_score(labels, predictions)
    kappa = cohen_kappa_score(labels, predictions)
    f1 = f1_score(labels, predictions)
    return {"balanced_accuracy": balanced, "roc_auc": roc_auc, "kappa": kappa, "f1": f1}


def plot_training(trainer):
    training_df = pd.DataFrame(trainer.state.log_history)[["epoch", "loss", "eval_loss"]]
    train_x = training_df["epoch"].loc[~training_df.loss.isna()]
    train_y = training_df["loss"].dropna()
    eval_x = training_df["epoch"].loc[~training_df.eval_loss.isna()]
    eval_y = training_df["eval_loss"].dropna()
    plt.plot(train_x, train_y, label="Training Loss")
    plt.plot(eval_x, eval_y, label="Eval Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training Loss")
    plt.show()


def plot_eval(trainer):
    metric_df = pd.DataFrame(trainer.state.log_history)[["epoch", "eval_kappa",
                                                     "eval_f1", "eval_balanced_accuracy"]]
    metric_df = metric_df.rename(columns={"eval_kappa": "Kappa",
                          "eval_f1": "F1", "eval_balanced_accuracy":
                          "Balanced Accuracy"})
    metric_df.dropna().set_index("epoch").plot()
    plt.title("Training Metrics")
    plt.show()


def inference(tokenizer, model, device, text_batch):
    tokenized = tokenizer(text_batch, padding=True, truncation=True, return_tensors="pt").to(device)
    return model(**tokenized)


def test_inference(tokenizer, model, device, test_set, batch_size):
    model.eval()
    test_set = test_set.map(lambda d: {"predicted_scores": inference(tokenizer, model, device, d["text"]).logits},
                            batched=True, batch_size=batch_size)
    test_set = test_set.map(lambda d: {"predicted": np.argmax(d["predicted_scores"], axis=1)},
                            batched=True, batch_size=batch_size)
    return test_set
