import os
import time
import io
import collections
import itertools
import csv
import re

import numpy as np
import tensorflow as tf
import keras
from keras.callbacks import Callback
from keras.utils import OrderedEnqueuer
from sklearn.metrics import confusion_matrix
import matplotlib
import matplotlib.pyplot as plt

import config

from helpers import get_logger

logger = get_logger(__name__)


def humansize(nbytes):
    suffixes = ["B", "KB", "MB", "GB", "TB", "PB"]
    i = 0
    while nbytes >= 1024 and i < len(suffixes) - 1:
        nbytes /= 1024.0
        i += 1
    f = ("%.2f" % nbytes).rstrip("0").rstrip(".")
    return "%s %s" % (f, suffixes[i])


def schedule_decay(epoch, lr, decay=0.001):
    if epoch >= 10:
        lr = lr * 1 / (1 + decay * epoch)
    return lr


def schedule_stepped(epoch, lr, step_size=10):
    if epoch > 0:
        if epoch % step_size == 0:
            return lr / 10.0
        else:
            return lr
    else:
        return lr


def load_nlcd_stats(
    stats_mu=config.LR_STATS_MU,
    stats_sigma=config.LR_STATS_SIGMA,
    class_weights=config.LR_CLASS_WEIGHTS,
    lr_classes=config.LR_NCLASSES,
    hr_classes=config.HR_NCLASSES,
):
    stats_mu = np.loadtxt(stats_mu)
    assert lr_classes == stats_mu.shape[0]
    assert hr_classes == (stats_mu.shape[1] + 1)
    nlcd_means = np.concatenate([np.zeros((lr_classes, 1)), stats_mu], axis=1)
    nlcd_means[nlcd_means == 0] = 0.000001
    nlcd_means[:, 0] = 0
    if stats_mu == "data/nlcd_mu.txt":
        nlcd_means = do_nlcd_means_tuning(nlcd_means)

    stats_sigma = np.loadtxt(stats_sigma)
    assert lr_classes == stats_sigma.shape[0]
    assert hr_classes == (stats_sigma.shape[1] + 1)
    nlcd_vars = np.concatenate([np.zeros((lr_classes, 1)), stats_sigma], axis=1)
    nlcd_vars[nlcd_vars < 0.0001] = 0.0001

    if not class_weights:
        nlcd_class_weights = np.ones((lr_classes,))
    else:
        nlcd_class_weights = np.loadtxt(class_weights)
        assert lr_classes == nlcd_class_weights.shape[0]

    return nlcd_class_weights, nlcd_means, nlcd_vars


def do_nlcd_means_tuning(nlcd_means):
    nlcd_means[2:, 1] -= 0
    nlcd_means[3:7, 4] += 0.25
    nlcd_means = nlcd_means / np.maximum(0, nlcd_means).sum(axis=1, keepdims=True)
    nlcd_means[0, :] = 0
    nlcd_means[-1, :] = 0
    return nlcd_means


def find_key_by_str(keys, needle):
    for key in keys:
        if needle in key:
            return key
    raise ValueError("%s not found in keys" % (needle))


class LandcoverResults(keras.callbacks.Callback):
    # pylint: disable=too-many-instance-attributes,super-init-not-called,dangerous-default-value
    def __init__(self, log_dir=None, verbose=False):

        self.mb_log_keys = None
        self.epoch_log_keys = None

        self.verbose = verbose
        self.log_dir = log_dir

        self.batch_num = 0
        self.epoch_num = 0

        self.train_start_time = None
        self.mb_start_time = None
        self.epoch_start_time = None

        if self.log_dir is not None:
            self.train_mb_fn = os.path.join(log_dir, "minibatch_history.txt")
            self.train_epoch_fn = os.path.join(log_dir, "epoch_history.txt")

    def on_train_begin(self, logs={}):
        self.train_start_time = time.time()

    def on_batch_begin(self, batch, logs={}):
        self.mb_start_time = float(time.time())

    def on_batch_end(self, batch, logs={}):
        t = time.time() - self.mb_start_time

        if self.mb_log_keys is None and self.log_dir is not None:
            self.mb_log_keys = [
                key for key in list(logs.keys()) if key not in ["batch", "size"]
            ]
            f = open(self.train_mb_fn, "w")
            f.write("Batch Number,Time Elapsed")
            for key in self.mb_log_keys:
                f.write(",%s" % (key))
            f.write("\n")
            f.close()

        if self.log_dir is not None:
            f = open(self.train_mb_fn, "a")
            f.write("%d,%f" % (self.batch_num, t))
            for key in self.mb_log_keys:
                f.write(",%f" % (logs[key]))
            f.write("\n")
            f.close()

        self.batch_num += 1

    def on_epoch_begin(self, epoch, logs={}):
        self.epoch_start_time = float(time.time())

    def on_epoch_end(self, epoch, logs=None):
        t = time.time() - self.epoch_start_time
        # total_time = time.time() - self.train_start_time

        if self.epoch_log_keys is None and self.log_dir is not None:
            self.epoch_log_keys = [key for key in list(logs.keys()) if key != "epoch"]
            f = open(self.train_epoch_fn, "w")
            f.write("Epoch Number,Time Elapsed")
            for key in self.epoch_log_keys:
                f.write(",%s" % (key))
            f.write("\n")
            f.close()

        if self.log_dir is not None:
            f = open(self.train_epoch_fn, "a")
            f.write("%d,%f" % (self.epoch_num, t))
            for key in self.epoch_log_keys:
                f.write(",%f" % (logs[key]))
            f.write("\n")
            f.close()

        self.epoch_num += 1


def handle_labels(arr, key_txt):
    key_array = np.loadtxt(key_txt)
    trans_arr = arr

    for translation in key_array:
        # translation is (src label, dst label)
        scr_l, dst_l = translation
        if scr_l != dst_l:
            trans_arr[trans_arr == scr_l] = dst_l

    # translated array
    return trans_arr


def classes_in_key(key_txt):
    key_array = np.loadtxt(key_txt)
    return len(np.unique(key_array[:, 1]))


def handle_labels_with_state(arr, state, key_dict):
    key_txt = ""
    for key in key_dict:
        if re.match(r"%s" % key, state):
            key_txt = key_dict[key]
    if key_txt:
        return handle_labels(arr, key_txt)
    else:
        return arr


def to_float(arr, data_type=config.DATA_TYPE):
    if data_type == "int8":
        res = np.clip(arr / 255.0, 0.0, 1.0)
    elif data_type == "int16":
        res = np.clip(arr / 4096.0, 0.0, 1.0)
    else:
        raise ValueError("Select an appropriate data type.")
    return res


class ModelDiagnoser(Callback):
    # pylint: disable=not-context-manager,too-many-instance-attributes
    # Disable context manager warning for generator
    """TensorFlow based Callback class to plot source imagery, labels,
    predicted labels, and confusion matrix in TensorBoard.
    Based on https://stackoverflow.com/a/55856716
    """

    def __init__(
        self,
        data_generator: keras.utils.Sequence,
        batch_size: int,
        num_samples: int,
        output_dir: str,
        superres: bool = False,
    ):
        """
        Parameters
        ----------
        data_generator : keras.utils.Sequence
            Training or validtion generator. The imagery and labels that are
            plotted will come from here.
        batch_size : int
            The size of each batch.
        num_samples : int
            The number of samples (images) to plot at each epoch.
        output_dir : str
            Output directory of logs for TensorBoard.
        superres : bool
            If the model is using superres loss function.

        Returns
        -------
        ModelDiagnoser
            Initialized TensorFlow callback, ready to be used with fit method.

        """

        def read_file_colormap(file_path: str) -> collections.OrderedDict:
            """Reads a text file with color assigned to each class. i.e.:
            class, r, g, b
            1, 255, 0, 0

            Which will result in label 1 to be assigned color red.

            Parameters
            ----------
            file_path : str
                File path of the text file described above.

            Returns
            -------
            collections.OrderedDict
                OrderedDict with label plus assigned rgb color.
            """
            out_list = []
            with open(file_path) as color_map:
                csv_color_map = csv.reader(color_map)
                next(csv_color_map)
                for row in csv_color_map:
                    out_list.append((int(row[0]), (row[1], (row[2:]))))
            return collections.OrderedDict(out_list)

        def to_matplotlib_colormap(
            ordered_dict: collections.OrderedDict,
        ) -> matplotlib.colors.ListedColormap:
            """From ordered_dict generate a ListedColormap.

            Parameters
            ----------
            ordered_dict : collections.OrderedDict
                OrderedDict generated by read_file_colormap.

            Returns
            -------
            matplotlib.colors.ListedColormap
                Colormap for labels ready to use within a matplotlib figure.

            """

            def rgb(r: int, g: int, b: int) -> str:
                def clamp(x):
                    return max(0, min(int(x), 255))

                return "#{0:02x}{1:02x}{2:02x}".format(clamp(r), clamp(g), clamp(b))

            return matplotlib.colors.ListedColormap(
                [
                    rgb(*ordered_dict[i][1]) if i in ordered_dict else "#000000"
                    for i in ordered_dict
                ]
            )

        super().__init__()
        self.batch_size = batch_size
        self.num_samples = num_samples
        self.superres = superres
        self.enqueuer = OrderedEnqueuer(
            data_generator, use_multiprocessing=True, shuffle=False
        )
        self.enqueuer.start(workers=4, max_queue_size=4)
        self.writer = tf.summary.create_file_writer(output_dir)

        self.hr_classes = read_file_colormap(config.HR_COLOR)
        assert (
            len(self.hr_classes) == config.HR_NCLASSES - 1
        ), f"Wrong HR color map {config.HR_COLOR}"
        self.sr_classes = read_file_colormap(config.LR_COLOR)
        assert (
            len(self.sr_classes) == config.LR_NCLASSES - 1
        ), f"Wrong SR color map {config.LR_COLOR}"

        self.hr_classes_cmap = to_matplotlib_colormap(self.hr_classes)
        self.sr_classes_cmap = to_matplotlib_colormap(self.sr_classes)

    def plot_confusion_matrix(
        self, correct_labels: np.ndarray, predict_labels: np.ndarray
    ) -> tf.Tensor:
        """From labels and predict_labels generate a confusion matrix.

        Parameters
        ----------
        correct_labels : np.ndarray
            Given with the data_generator.
        predict_labels : np.ndarray
            Result of apllying the model on imagery given by data_generator.

        Returns
        -------
        tf.Tensor
            A tensor representing a confusion matrix image.

        """
        labels = [0] + list(self.hr_classes.keys())
        cm = confusion_matrix(
            correct_labels.reshape(-1), predict_labels.reshape(-1), labels=labels
        )
        figure = plt.figure(figsize=(10, 10))
        plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        plt.title("Confusion matrix")
        plt.colorbar()
        tick_marks = np.arange(len(labels))
        plt.xticks(tick_marks, labels, rotation=45)
        plt.yticks(tick_marks, labels)

        # Normalize the confusion matrix.
        cm = np.around(cm.astype("float") / cm.sum(axis=1)[:, np.newaxis], decimals=2)

        # Use white text if squares are dark; otherwise black.
        threshold = cm.max() / 2.0
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            color = "white" if cm[i, j] > threshold else "black"
            plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)

        plt.tight_layout()
        plt.ylabel("True label")
        plt.xlabel("Predicted label")
        return self.plot_to_image(figure)

    def plot_classification(
        self, np_array: np.ndarray, cmap: matplotlib.colors.ListedColormap
    ) -> tf.Tensor:
        """From labels generate an image with the classes mapped to the cmap (color).

        Parameters
        ----------
        np_array : np.ndarray
            Labels as given by the data_generator.
        cmap : matplotlib.colors.ListedColormap
            Mapping of class to color.

        Returns
        -------
        tf.Tensor
            A tensor representing a label plot with appropriate color.

        """
        figure = plt.figure(figsize=(10, 10))
        plt.imshow(np.squeeze(np_array), cmap=cmap, vmin=0, vmax=cmap.N)
        plt.axis("off")
        return self.plot_to_image(figure)

    @staticmethod
    def plot_to_image(figure: matplotlib.figure.Figure) -> tf.Tensor:
        """Generate a tensor from a matplotlib figure.
        From https://www.tensorflow.org/tensorboard/image_summaries#logging_arbitrary_image_data

        Parameters
        ----------
        figure : matplotlib.figure.Figure
            Figure to convert to tensor.

        Returns
        -------
        tf.Tensor
            A tensor representation of the figure.

        """
        # Save the plot to a PNG in memory.
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        # Closing the figure prevents it from being displayed directly inside
        # the notebook.
        plt.close(figure)
        buf.seek(0)
        # Convert PNG buffer to TF image
        image = tf.image.decode_png(buf.getvalue(), channels=4)
        # Add the batch dimension
        image = tf.expand_dims(image, 0)
        return image

    def on_epoch_end(self, epoch: int, logs=None):
        """Defines what is run at the end of one epoch.
        - Plots images
        - Plots label
        - Plots prediction labels
        - Plots confusion matrix of batch

        Parameters
        ----------
        epoch : int
            The epoch number.
        """

        def to_label(batch: np.ndarray) -> np.ndarray:
            """From label 1hot encoded to classified labels.

            Parameters
            ----------
            batch : np.ndarray
                One hot encoded labels (b_size, h, w, nclasses).

            Returns
            -------
            np.ndarray
                Regular labels. (b_size, h, w, 1)

            """

            label_batch = np.zeros(batch.shape[0:3])
            for i in range(batch.shape[0]):
                label_batch[i] = np.argmax(batch[i], axis=2)
            return label_batch

        output_generator = self.enqueuer.get()
        generator_output = next(output_generator)
        if self.superres:
            x_batch, y = generator_output
            y_batch_hr = y["outputs_hr"]
            y_batch_sr = y["outputs_sr"]
        else:
            x_batch, y_batch_hr = generator_output

        y_pred = self.model.predict(x_batch)
        if self.superres:
            y_pred, y_pred_sr = y_pred

        label_y_hr = to_label(y_batch_hr)
        label_y_pred = to_label(y_pred)

        with self.writer.as_default():
            for sample_index in [i for i in range(0, 3) if i <= self.batch_size - 1]:
                tf.summary.image(
                    "Epoch-{}/{}/image".format(epoch, sample_index),
                    x_batch[[sample_index], :, :, :3],
                    step=epoch,
                )
                tf.summary.image(
                    "Epoch-{}/{}/label".format(epoch, sample_index),
                    self.plot_classification(
                        label_y_hr[sample_index], self.hr_classes_cmap,
                    ),
                    step=epoch,
                )
                tf.summary.image(
                    "Epoch-{}/{}/pred".format(epoch, sample_index),
                    self.plot_classification(
                        label_y_pred[sample_index], self.hr_classes_cmap,
                    ),
                    step=epoch,
                )
                tf.summary.image(
                    f"Epoch-{epoch}/confusion_matrix",
                    self.plot_confusion_matrix(
                        label_y_hr.squeeze(), label_y_pred.squeeze(),
                    ),
                    step=epoch,
                )

                if self.superres:
                    tf.summary.image(
                        "Epoch-{}/{}/label_sr".format(epoch, sample_index),
                        self.plot_classification(
                            np.argmax(y_batch_sr[sample_index, :, :, :], axis=2),
                            self.sr_classes_cmap,
                        ),
                        step=epoch,
                    )
                    tf.summary.image(
                        "Epoch-{}/{}/pred_sr".format(epoch, sample_index),
                        self.plot_classification(
                            np.argmax(y_pred_sr[sample_index, :, :, :], axis=2),
                            self.hr_classes_cmap,
                        ),
                        step=epoch,
                    )

    def on_train_end(self, logs=None):
        """Defines what is run at the end of training.
        - Stops the data_generator
        - Closes TensorBoard writer
        """
        self.enqueuer.stop()
        self.writer.close()
