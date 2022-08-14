{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "b518b04cbfe0"
   },
   "source": [
    "##### Copyright 2020 The TensorFlow Authors."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {
    "cellView": "form",
    "execution": {
     "iopub.execute_input": "2021-08-25T17:52:21.717383Z",
     "iopub.status.busy": "2021-08-25T17:52:21.716814Z",
     "iopub.status.idle": "2021-08-25T17:52:21.719667Z",
     "shell.execute_reply": "2021-08-25T17:52:21.719136Z"
    },
    "id": "906e07f6e562"
   },
   "outputs": [],
   "source": [
    "#@title Licensed under the Apache License, Version 2.0 (the \"License\");\n",
    "# you may not use this file except in compliance with the License.\n",
    "# You may obtain a copy of the License at\n",
    "#\n",
    "# https://www.apache.org/licenses/LICENSE-2.0\n",
    "#\n",
    "# Unless required by applicable law or agreed to in writing, software\n",
    "# distributed under the License is distributed on an \"AS IS\" BASIS,\n",
    "# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.\n",
    "# See the License for the specific language governing permissions and\n",
    "# limitations under the License."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "d201e826ab29"
   },
   "source": [
    "# Writing your own callbacks"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "71699af85d2d"
   },
   "source": [
    "<table class=\"tfo-notebook-buttons\" align=\"left\">\n",
    "  <td>\n",
    "    <a target=\"_blank\" href=\"https://www.tensorflow.org/guide/keras/custom_callback\"><img src=\"https://www.tensorflow.org/images/tf_logo_32px.png\" />View on TensorFlow.org</a>\n",
    "  </td>\n",
    "  <td>\n",
    "    <a target=\"_blank\" href=\"https://colab.research.google.com/github/tensorflow/docs/blob/snapshot-keras/site/en/guide/keras/custom_callback.ipynb\"><img src=\"https://www.tensorflow.org/images/colab_logo_32px.png\" />Run in Google Colab</a>\n",
    "  </td>\n",
    "  <td>\n",
    "    <a target=\"_blank\" href=\"https://github.com/keras-team/keras-io/blob/master/guides/writing_your_own_callbacks.py\"><img src=\"https://www.tensorflow.org/images/GitHub-Mark-32px.png\" />View source on GitHub</a>\n",
    "  </td>\n",
    "  <td>\n",
    "    <a href=\"https://storage.googleapis.com/tensorflow_docs/docs/site/en/guide/keras/custom_callback.ipynb\"><img src=\"https://www.tensorflow.org/images/download_logo_32px.png\" />Download notebook</a>\n",
    "  </td>\n",
    "</table>"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "d75eb2e25f36"
   },
   "source": [
    "## Introduction\n",
    "\n",
    "A callback is a powerful tool to customize the behavior of a Keras model during\n",
    "training, evaluation, or inference. Examples include `tf.keras.callbacks.TensorBoard`\n",
    "to visualize training progress and results with TensorBoard, or\n",
    "`tf.keras.callbacks.ModelCheckpoint` to periodically save your model during training.\n",
    "\n",
    "In this guide, you will learn what a Keras callback is, what it can do, and how you can\n",
    "build your own. We provide a few demos of simple callback applications to get you\n",
    "started."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "b3600ee25c8e"
   },
   "source": [
    "## Setup"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2021-08-25T17:52:21.726870Z",
     "iopub.status.busy": "2021-08-25T17:52:21.726258Z",
     "iopub.status.idle": "2021-08-25T17:52:23.124555Z",
     "shell.execute_reply": "2021-08-25T17:52:23.123907Z"
    },
    "id": "4dadb6688663"
   },
   "outputs": [],
   "source": [
    "import tensorflow as tf\n",
    "from tensorflow import keras"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "42676f705fc8"
   },
   "source": [
    "## Keras callbacks overview\n",
    "\n",
    "All callbacks subclass the `keras.callbacks.Callback` class, and\n",
    "override a set of methods called at various stages of training, testing, and\n",
    "predicting. Callbacks are useful to get a view on internal states and statistics of\n",
    "the model during training.\n",
    "\n",
    "You can pass a list of callbacks (as the keyword argument `callbacks`) to the following\n",
    "model methods:\n",
    "\n",
    "- `keras.Model.fit()`\n",
    "- `keras.Model.evaluate()`\n",
    "- `keras.Model.predict()`"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "46945bdf5056"
   },
   "source": [
    "## An overview of callback methods\n",
    "\n",
    "### Global methods\n",
    "\n",
    "#### `on_(train|test|predict)_begin(self, logs=None)`\n",
    "\n",
    "Called at the beginning of `fit`/`evaluate`/`predict`.\n",
    "\n",
    "#### `on_(train|test|predict)_end(self, logs=None)`\n",
    "\n",
    "Called at the end of `fit`/`evaluate`/`predict`.\n",
    "\n",
    "### Batch-level methods for training/testing/predicting\n",
    "\n",
    "#### `on_(train|test|predict)_batch_begin(self, batch, logs=None)`\n",
    "\n",
    "Called right before processing a batch during training/testing/predicting.\n",
    "\n",
    "#### `on_(train|test|predict)_batch_end(self, batch, logs=None)`\n",
    "\n",
    "Called at the end of training/testing/predicting a batch. Within this method, `logs` is\n",
    "a dict containing the metrics results.\n",
    "\n",
    "### Epoch-level methods (training only)\n",
    "\n",
    "#### `on_epoch_begin(self, epoch, logs=None)`\n",
    "\n",
    "Called at the beginning of an epoch during training.\n",
    "\n",
    "#### `on_epoch_end(self, epoch, logs=None)`\n",
    "\n",
    "Called at the end of an epoch during training."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "82f2370418a1"
   },
   "source": [
    "## A basic example\n",
    "\n",
    "Let's take a look at a concrete example. To get started, let's import tensorflow and\n",
    "define a simple Sequential Keras model:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2021-08-25T17:52:23.130215Z",
     "iopub.status.busy": "2021-08-25T17:52:23.129606Z",
     "iopub.status.idle": "2021-08-25T17:52:23.131698Z",
     "shell.execute_reply": "2021-08-25T17:52:23.131292Z"
    },
    "id": "7350ea602e50"
   },
   "outputs": [],
   "source": [
    "# Define the Keras model to add callbacks to\n",
    "def get_model():\n",
    "    model = keras.Sequential()\n",
    "    model.add(keras.layers.Dense(1, input_dim=784))\n",
    "    model.compile(\n",
    "        optimizer=keras.optimizers.RMSprop(learning_rate=0.1),\n",
    "        loss=\"mean_squared_error\",\n",
    "        metrics=[\"mean_absolute_error\"],\n",
    "    )\n",
    "    return model\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "044db5f2dc6f"
   },
   "source": [
    "Then, load the MNIST data for training and testing from Keras datasets API:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2021-08-25T17:52:23.136765Z",
     "iopub.status.busy": "2021-08-25T17:52:23.136081Z",
     "iopub.status.idle": "2021-08-25T17:52:23.719915Z",
     "shell.execute_reply": "2021-08-25T17:52:23.720348Z"
    },
    "id": "f8826736a184"
   },
   "outputs": [],
   "source": [
    "# Load example MNIST data and pre-process it\n",
    "(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()\n",
    "x_train = x_train.reshape(-1, 784).astype(\"float32\") / 255.0\n",
    "x_test = x_test.reshape(-1, 784).astype(\"float32\") / 255.0\n",
    "\n",
    "# Limit the data to 1000 samples\n",
    "x_train = x_train[:1000]\n",
    "y_train = y_train[:1000]\n",
    "x_test = x_test[:1000]\n",
    "y_test = y_test[:1000]"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "b9acd50b2215"
   },
   "source": [
    "Now, define a simple custom callback that logs:\n",
    "\n",
    "- When `fit`/`evaluate`/`predict` starts & ends\n",
    "- When each epoch starts & ends\n",
    "- When each training batch starts & ends\n",
    "- When each evaluation (test) batch starts & ends\n",
    "- When each inference (prediction) batch starts & ends"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2021-08-25T17:52:23.732913Z",
     "iopub.status.busy": "2021-08-25T17:52:23.732197Z",
     "iopub.status.idle": "2021-08-25T17:52:23.733983Z",
     "shell.execute_reply": "2021-08-25T17:52:23.734412Z"
    },
    "id": "cc9888d28e79"
   },
   "outputs": [],
   "source": [
    "class CustomCallback(keras.callbacks.Callback):\n",
    "    def on_train_begin(self, logs=None):\n",
    "        keys = list(logs.keys())\n",
    "        print(\"Starting training; got log keys: {}\".format(keys))\n",
    "\n",
    "    def on_train_end(self, logs=None):\n",
    "        keys = list(logs.keys())\n",
    "        print(\"Stop training; got log keys: {}\".format(keys))\n",
    "\n",
    "    def on_epoch_begin(self, epoch, logs=None):\n",
    "        keys = list(logs.keys())\n",
    "        print(\"Start epoch {} of training; got log keys: {}\".format(epoch, keys))\n",
    "\n",
    "    def on_epoch_end(self, epoch, logs=None):\n",
    "        keys = list(logs.keys())\n",
    "        print(\"End epoch {} of training; got log keys: {}\".format(epoch, keys))\n",
    "\n",
    "    def on_test_begin(self, logs=None):\n",
    "        keys = list(logs.keys())\n",
    "        print(\"Start testing; got log keys: {}\".format(keys))\n",
    "\n",
    "    def on_test_end(self, logs=None):\n",
    "        keys = list(logs.keys())\n",
    "        print(\"Stop testing; got log keys: {}\".format(keys))\n",
    "\n",
    "    def on_predict_begin(self, logs=None):\n",
    "        keys = list(logs.keys())\n",
    "        print(\"Start predicting; got log keys: {}\".format(keys))\n",
    "\n",
    "    def on_predict_end(self, logs=None):\n",
    "        keys = list(logs.keys())\n",
    "        print(\"Stop predicting; got log keys: {}\".format(keys))\n",
    "\n",
    "    def on_train_batch_begin(self, batch, logs=None):\n",
    "        keys = list(logs.keys())\n",
    "        print(\"...Training: start of batch {}; got log keys: {}\".format(batch, keys))\n",
    "\n",
    "    def on_train_batch_end(self, batch, logs=None):\n",
    "        keys = list(logs.keys())\n",
    "        print(\"...Training: end of batch {}; got log keys: {}\".format(batch, keys))\n",
    "\n",
    "    def on_test_batch_begin(self, batch, logs=None):\n",
    "        keys = list(logs.keys())\n",
    "        print(\"...Evaluating: start of batch {}; got log keys: {}\".format(batch, keys))\n",
    "\n",
    "    def on_test_batch_end(self, batch, logs=None):\n",
    "        keys = list(logs.keys())\n",
    "        print(\"...Evaluating: end of batch {}; got log keys: {}\".format(batch, keys))\n",
    "\n",
    "    def on_predict_batch_begin(self, batch, logs=None):\n",
    "        keys = list(logs.keys())\n",
    "        print(\"...Predicting: start of batch {}; got log keys: {}\".format(batch, keys))\n",
    "\n",
    "    def on_predict_batch_end(self, batch, logs=None):\n",
    "        keys = list(logs.keys())\n",
    "        print(\"...Predicting: end of batch {}; got log keys: {}\".format(batch, keys))\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "8184bd3a76c2"
   },
   "source": [
    "Let's try it out:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2021-08-25T17:52:23.740104Z",
     "iopub.status.busy": "2021-08-25T17:52:23.739464Z",
     "iopub.status.idle": "2021-08-25T17:52:26.666045Z",
     "shell.execute_reply": "2021-08-25T17:52:26.666441Z"
    },
    "id": "75f7aa1edac6"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Starting training; got log keys: []\n",
      "Start epoch 0 of training; got log keys: []\n",
      "...Training: start of batch 0; got log keys: []\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "...Training: end of batch 0; got log keys: ['loss', 'mean_absolute_error']\n",
      "...Training: start of batch 1; got log keys: []\n",
      "...Training: end of batch 1; got log keys: ['loss', 'mean_absolute_error']\n",
      "...Training: start of batch 2; got log keys: []\n",
      "...Training: end of batch 2; got log keys: ['loss', 'mean_absolute_error']\n",
      "...Training: start of batch 3; got log keys: []\n",
      "...Training: end of batch 3; got log keys: ['loss', 'mean_absolute_error']\n",
      "Start testing; got log keys: []\n",
      "...Evaluating: start of batch 0; got log keys: []\n",
      "...Evaluating: end of batch 0; got log keys: ['loss', 'mean_absolute_error']\n",
      "...Evaluating: start of batch 1; got log keys: []\n",
      "...Evaluating: end of batch 1; got log keys: ['loss', 'mean_absolute_error']\n",
      "...Evaluating: start of batch 2; got log keys: []\n",
      "...Evaluating: end of batch 2; got log keys: ['loss', 'mean_absolute_error']\n",
      "...Evaluating: start of batch 3; got log keys: []\n",
      "...Evaluating: end of batch 3; got log keys: ['loss', 'mean_absolute_error']\n",
      "Stop testing; got log keys: ['loss', 'mean_absolute_error']\n",
      "End epoch 0 of training; got log keys: ['loss', 'mean_absolute_error', 'val_loss', 'val_mean_absolute_error']\n",
      "Stop training; got log keys: ['loss', 'mean_absolute_error', 'val_loss', 'val_mean_absolute_error']\n",
      "Start testing; got log keys: []\n",
      "...Evaluating: start of batch 0; got log keys: []\n",
      "...Evaluating: end of batch 0; got log keys: ['loss', 'mean_absolute_error']\n",
      "...Evaluating: start of batch 1; got log keys: []\n",
      "...Evaluating: end of batch 1; got log keys: ['loss', 'mean_absolute_error']\n",
      "...Evaluating: start of batch 2; got log keys: []\n",
      "...Evaluating: end of batch 2; got log keys: ['loss', 'mean_absolute_error']\n",
      "...Evaluating: start of batch 3; got log keys: []\n",
      "...Evaluating: end of batch 3; got log keys: ['loss', 'mean_absolute_error']\n",
      "...Evaluating: start of batch 4; got log keys: []\n",
      "...Evaluating: end of batch 4; got log keys: ['loss', 'mean_absolute_error']\n",
      "...Evaluating: start of batch 5; got log keys: []\n",
      "...Evaluating: end of batch 5; got log keys: ['loss', 'mean_absolute_error']\n",
      "...Evaluating: start of batch 6; got log keys: []\n",
      "...Evaluating: end of batch 6; got log keys: ['loss', 'mean_absolute_error']\n",
      "...Evaluating: start of batch 7; got log keys: []\n",
      "...Evaluating: end of batch 7; got log keys: ['loss', 'mean_absolute_error']\n",
      "Stop testing; got log keys: ['loss', 'mean_absolute_error']\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Start predicting; got log keys: []\n",
      "...Predicting: start of batch 0; got log keys: []\n",
      "...Predicting: end of batch 0; got log keys: ['outputs']\n",
      "...Predicting: start of batch 1; got log keys: []\n",
      "...Predicting: end of batch 1; got log keys: ['outputs']\n",
      "...Predicting: start of batch 2; got log keys: []\n",
      "...Predicting: end of batch 2; got log keys: ['outputs']\n",
      "...Predicting: start of batch 3; got log keys: []\n",
      "...Predicting: end of batch 3; got log keys: ['outputs']\n",
      "...Predicting: start of batch 4; got log keys: []\n",
      "...Predicting: end of batch 4; got log keys: ['outputs']\n",
      "...Predicting: start of batch 5; got log keys: []\n",
      "...Predicting: end of batch 5; got log keys: ['outputs']\n",
      "...Predicting: start of batch 6; got log keys: []\n",
      "...Predicting: end of batch 6; got log keys: ['outputs']\n",
      "...Predicting: start of batch 7; got log keys: []\n",
      "...Predicting: end of batch 7; got log keys: ['outputs']\n",
      "Stop predicting; got log keys: []\n"
     ]
    }
   ],
   "source": [
    "model = get_model()\n",
    "model.fit(\n",
    "    x_train,\n",
    "    y_train,\n",
    "    batch_size=128,\n",
    "    epochs=1,\n",
    "    verbose=0,\n",
    "    validation_split=0.5,\n",
    "    callbacks=[CustomCallback()],\n",
    ")\n",
    "\n",
    "res = model.evaluate(\n",
    "    x_test, y_test, batch_size=128, verbose=0, callbacks=[CustomCallback()]\n",
    ")\n",
    "\n",
    "res = model.predict(x_test, batch_size=128, callbacks=[CustomCallback()])"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "02113b8677eb"
   },
   "source": [
    "### Usage of `logs` dict\n",
    "The `logs` dict contains the loss value, and all the metrics at the end of a batch or\n",
    "epoch. Example includes the loss and mean absolute error."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2021-08-25T17:52:26.673729Z",
     "iopub.status.busy": "2021-08-25T17:52:26.673047Z",
     "iopub.status.idle": "2021-08-25T17:52:27.178169Z",
     "shell.execute_reply": "2021-08-25T17:52:27.178604Z"
    },
    "id": "629bc145eb84"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Up to batch 0, the average loss is   30.79.\n",
      "Up to batch 1, the average loss is  459.11.\n",
      "Up to batch 2, the average loss is  314.68.\n",
      "Up to batch 3, the average loss is  237.97.\n",
      "Up to batch 4, the average loss is  191.76.\n",
      "Up to batch 5, the average loss is  160.95.\n",
      "Up to batch 6, the average loss is  138.74.\n",
      "Up to batch 7, the average loss is  124.85.\n",
      "The average loss for epoch 0 is  124.85 and mean absolute error is    6.00.\n",
      "Up to batch 0, the average loss is    5.13.\n",
      "Up to batch 1, the average loss is    4.66.\n",
      "Up to batch 2, the average loss is    4.71.\n",
      "Up to batch 3, the average loss is    4.66.\n",
      "Up to batch 4, the average loss is    4.69.\n",
      "Up to batch 5, the average loss is    4.56.\n",
      "Up to batch 6, the average loss is    4.77.\n",
      "Up to batch 7, the average loss is    4.77.\n",
      "The average loss for epoch 1 is    4.77 and mean absolute error is    1.75.\n",
      "Up to batch 0, the average loss is    5.73.\n",
      "Up to batch 1, the average loss is    5.04.\n",
      "Up to batch 2, the average loss is    5.10.\n",
      "Up to batch 3, the average loss is    5.14.\n",
      "Up to batch 4, the average loss is    5.37.\n",
      "Up to batch 5, the average loss is    5.24.\n",
      "Up to batch 6, the average loss is    5.22.\n",
      "Up to batch 7, the average loss is    5.16.\n"
     ]
    }
   ],
   "source": [
    "class LossAndErrorPrintingCallback(keras.callbacks.Callback):\n",
    "    def on_train_batch_end(self, batch, logs=None):\n",
    "        print(\n",
    "            \"Up to batch {}, the average loss is {:7.2f}.\".format(batch, logs[\"loss\"])\n",
    "        )\n",
    "\n",
    "    def on_test_batch_end(self, batch, logs=None):\n",
    "        print(\n",
    "            \"Up to batch {}, the average loss is {:7.2f}.\".format(batch, logs[\"loss\"])\n",
    "        )\n",
    "\n",
    "    def on_epoch_end(self, epoch, logs=None):\n",
    "        print(\n",
    "            \"The average loss for epoch {} is {:7.2f} \"\n",
    "            \"and mean absolute error is {:7.2f}.\".format(\n",
    "                epoch, logs[\"loss\"], logs[\"mean_absolute_error\"]\n",
    "            )\n",
    "        )\n",
    "\n",
    "\n",
    "model = get_model()\n",
    "model.fit(\n",
    "    x_train,\n",
    "    y_train,\n",
    "    batch_size=128,\n",
    "    epochs=2,\n",
    "    verbose=0,\n",
    "    callbacks=[LossAndErrorPrintingCallback()],\n",
    ")\n",
    "\n",
    "res = model.evaluate(\n",
    "    x_test,\n",
    "    y_test,\n",
    "    batch_size=128,\n",
    "    verbose=0,\n",
    "    callbacks=[LossAndErrorPrintingCallback()],\n",
    ")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "742d62e5394a"
   },
   "source": [
    "## Usage of `self.model` attribute\n",
    "\n",
    "In addition to receiving log information when one of their methods is called,\n",
    "callbacks have access to the model associated with the current round of\n",
    "training/evaluation/inference: `self.model`.\n",
    "\n",
    "Here are of few of the things you can do with `self.model` in a callback:\n",
    "\n",
    "- Set `self.model.stop_training = True` to immediately interrupt training.\n",
    "- Mutate hyperparameters of the optimizer (available as `self.model.optimizer`),\n",
    "such as `self.model.optimizer.learning_rate`.\n",
    "- Save the model at period intervals.\n",
    "- Record the output of `model.predict()` on a few test samples at the end of each\n",
    "epoch, to use as a sanity check during training.\n",
    "- Extract visualizations of intermediate features at the end of each epoch, to monitor\n",
    "what the model is learning over time.\n",
    "- etc.\n",
    "\n",
    "Let's see this in action in a couple of examples."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "7eb29d3ed752"
   },
   "source": [
    "## Examples of Keras callback applications"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "2d1d29d99fa5"
   },
   "source": [
    "### Early stopping at minimum loss\n",
    "\n",
    "This first example shows the creation of a `Callback` that stops training when the\n",
    "minimum of loss has been reached, by setting the attribute `self.model.stop_training`\n",
    "(boolean). Optionally, you can provide an argument `patience` to specify how many\n",
    "epochs we should wait before stopping after having reached a local minimum.\n",
    "\n",
    "`tf.keras.callbacks.EarlyStopping` provides a more complete and general implementation."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2021-08-25T17:52:27.187886Z",
     "iopub.status.busy": "2021-08-25T17:52:27.187299Z",
     "iopub.status.idle": "2021-08-25T17:52:27.517300Z",
     "shell.execute_reply": "2021-08-25T17:52:27.517742Z"
    },
    "id": "5d2acd79cecd"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Up to batch 0, the average loss is   34.62.\n",
      "Up to batch 1, the average loss is  405.62.\n",
      "Up to batch 2, the average loss is  282.27.\n",
      "Up to batch 3, the average loss is  215.95.\n",
      "Up to batch 4, the average loss is  175.32.\n",
      "The average loss for epoch 0 is  175.32 and mean absolute error is    8.59.\n",
      "Up to batch 0, the average loss is    8.86.\n",
      "Up to batch 1, the average loss is    7.31.\n",
      "Up to batch 2, the average loss is    6.51.\n",
      "Up to batch 3, the average loss is    6.71.\n",
      "Up to batch 4, the average loss is    6.24.\n",
      "The average loss for epoch 1 is    6.24 and mean absolute error is    2.06.\n",
      "Up to batch 0, the average loss is    4.83.\n",
      "Up to batch 1, the average loss is    5.05.\n",
      "Up to batch 2, the average loss is    4.71.\n",
      "Up to batch 3, the average loss is    4.41.\n",
      "Up to batch 4, the average loss is    4.48.\n",
      "The average loss for epoch 2 is    4.48 and mean absolute error is    1.68.\n",
      "Up to batch 0, the average loss is    5.84.\n",
      "Up to batch 1, the average loss is    5.73.\n",
      "Up to batch 2, the average loss is    7.24.\n",
      "Up to batch 3, the average loss is   10.34.\n",
      "Up to batch 4, the average loss is   15.53.\n",
      "The average loss for epoch 3 is   15.53 and mean absolute error is    3.20.\n",
      "Restoring model weights from the end of the best epoch.\n",
      "Epoch 00004: early stopping\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "<keras.callbacks.History at 0x7fd0843bf510>"
      ]
     },
     "execution_count": 8,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "import numpy as np\n",
    "\n",
    "\n",
    "class EarlyStoppingAtMinLoss(keras.callbacks.Callback):\n",
    "    \"\"\"Stop training when the loss is at its min, i.e. the loss stops decreasing.\n",
    "\n",
    "  Arguments:\n",
    "      patience: Number of epochs to wait after min has been hit. After this\n",
    "      number of no improvement, training stops.\n",
    "  \"\"\"\n",
    "\n",
    "    def __init__(self, patience=0):\n",
    "        super(EarlyStoppingAtMinLoss, self).__init__()\n",
    "        self.patience = patience\n",
    "        # best_weights to store the weights at which the minimum loss occurs.\n",
    "        self.best_weights = None\n",
    "\n",
    "    def on_train_begin(self, logs=None):\n",
    "        # The number of epoch it has waited when loss is no longer minimum.\n",
    "        self.wait = 0\n",
    "        # The epoch the training stops at.\n",
    "        self.stopped_epoch = 0\n",
    "        # Initialize the best as infinity.\n",
    "        self.best = np.Inf\n",
    "\n",
    "    def on_epoch_end(self, epoch, logs=None):\n",
    "        current = logs.get(\"loss\")\n",
    "        if np.less(current, self.best):\n",
    "            self.best = current\n",
    "            self.wait = 0\n",
    "            # Record the best weights if current results is better (less).\n",
    "            self.best_weights = self.model.get_weights()\n",
    "        else:\n",
    "            self.wait += 1\n",
    "            if self.wait >= self.patience:\n",
    "                self.stopped_epoch = epoch\n",
    "                self.model.stop_training = True\n",
    "                print(\"Restoring model weights from the end of the best epoch.\")\n",
    "                self.model.set_weights(self.best_weights)\n",
    "\n",
    "    def on_train_end(self, logs=None):\n",
    "        if self.stopped_epoch > 0:\n",
    "            print(\"Epoch %05d: early stopping\" % (self.stopped_epoch + 1))\n",
    "\n",
    "\n",
    "model = get_model()\n",
    "model.fit(\n",
    "    x_train,\n",
    "    y_train,\n",
    "    batch_size=64,\n",
    "    steps_per_epoch=5,\n",
    "    epochs=30,\n",
    "    verbose=0,\n",
    "    callbacks=[LossAndErrorPrintingCallback(), EarlyStoppingAtMinLoss()],\n",
    ")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "939ecfbe0383"
   },
   "source": [
    "### Learning rate scheduling\n",
    "\n",
    "In this example, we show how a custom Callback can be used to dynamically change the\n",
    "learning rate of the optimizer during the course of training.\n",
    "\n",
    "See `callbacks.LearningRateScheduler` for a more general implementations."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2021-08-25T17:52:27.527142Z",
     "iopub.status.busy": "2021-08-25T17:52:27.526579Z",
     "iopub.status.idle": "2021-08-25T17:52:27.970792Z",
     "shell.execute_reply": "2021-08-25T17:52:27.970178Z"
    },
    "id": "71c752b248c0"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "Epoch 00000: Learning rate is 0.1000.\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Up to batch 0, the average loss is   26.55.\n",
      "Up to batch 1, the average loss is  435.15.\n",
      "Up to batch 2, the average loss is  298.00.\n",
      "Up to batch 3, the average loss is  225.91.\n",
      "Up to batch 4, the average loss is  182.66.\n",
      "The average loss for epoch 0 is  182.66 and mean absolute error is    8.16.\n",
      "\n",
      "Epoch 00001: Learning rate is 0.1000.\n",
      "Up to batch 0, the average loss is    7.30.\n",
      "Up to batch 1, the average loss is    6.22.\n",
      "Up to batch 2, the average loss is    6.76.\n",
      "Up to batch 3, the average loss is    6.37.\n",
      "Up to batch 4, the average loss is    5.98.\n",
      "The average loss for epoch 1 is    5.98 and mean absolute error is    2.01.\n",
      "\n",
      "Epoch 00002: Learning rate is 0.1000.\n",
      "Up to batch 0, the average loss is    4.23.\n",
      "Up to batch 1, the average loss is    4.56.\n",
      "Up to batch 2, the average loss is    4.81.\n",
      "Up to batch 3, the average loss is    4.63.\n",
      "Up to batch 4, the average loss is    4.67.\n",
      "The average loss for epoch 2 is    4.67 and mean absolute error is    1.73.\n",
      "\n",
      "Epoch 00003: Learning rate is 0.0500.\n",
      "Up to batch 0, the average loss is    6.24.\n",
      "Up to batch 1, the average loss is    5.62.\n",
      "Up to batch 2, the average loss is    5.48.\n",
      "Up to batch 3, the average loss is    5.09.\n",
      "Up to batch 4, the average loss is    4.68.\n",
      "The average loss for epoch 3 is    4.68 and mean absolute error is    1.77.\n",
      "\n",
      "Epoch 00004: Learning rate is 0.0500.\n",
      "Up to batch 0, the average loss is    3.38.\n",
      "Up to batch 1, the average loss is    3.83.\n",
      "Up to batch 2, the average loss is    3.53.\n",
      "Up to batch 3, the average loss is    3.64.\n",
      "Up to batch 4, the average loss is    3.76.\n",
      "The average loss for epoch 4 is    3.76 and mean absolute error is    1.54.\n",
      "\n",
      "Epoch 00005: Learning rate is 0.0500.\n",
      "Up to batch 0, the average loss is    3.62.\n",
      "Up to batch 1, the average loss is    3.79.\n",
      "Up to batch 2, the average loss is    3.75.\n",
      "Up to batch 3, the average loss is    3.83.\n",
      "Up to batch 4, the average loss is    4.37.\n",
      "The average loss for epoch 5 is    4.37 and mean absolute error is    1.65.\n",
      "\n",
      "Epoch 00006: Learning rate is 0.0100.\n",
      "Up to batch 0, the average loss is    6.73.\n",
      "Up to batch 1, the average loss is    6.13.\n",
      "Up to batch 2, the average loss is    5.11.\n",
      "Up to batch 3, the average loss is    4.57.\n",
      "Up to batch 4, the average loss is    4.21.\n",
      "The average loss for epoch 6 is    4.21 and mean absolute error is    1.61.\n",
      "\n",
      "Epoch 00007: Learning rate is 0.0100.\n",
      "Up to batch 0, the average loss is    3.37.\n",
      "Up to batch 1, the average loss is    3.83.\n",
      "Up to batch 2, the average loss is    3.80.\n",
      "Up to batch 3, the average loss is    3.50.\n",
      "Up to batch 4, the average loss is    3.31.\n",
      "The average loss for epoch 7 is    3.31 and mean absolute error is    1.42.\n",
      "\n",
      "Epoch 00008: Learning rate is 0.0100.\n",
      "Up to batch 0, the average loss is    5.33.\n",
      "Up to batch 1, the average loss is    4.84.\n",
      "Up to batch 2, the average loss is    4.02.\n",
      "Up to batch 3, the average loss is    3.87.\n",
      "Up to batch 4, the average loss is    3.85.\n",
      "The average loss for epoch 8 is    3.85 and mean absolute error is    1.53.\n",
      "\n",
      "Epoch 00009: Learning rate is 0.0050.\n",
      "Up to batch 0, the average loss is    1.84.\n",
      "Up to batch 1, the average loss is    2.75.\n",
      "Up to batch 2, the average loss is    3.16.\n",
      "Up to batch 3, the average loss is    3.52.\n",
      "Up to batch 4, the average loss is    3.34.\n",
      "The average loss for epoch 9 is    3.34 and mean absolute error is    1.43.\n",
      "\n",
      "Epoch 00010: Learning rate is 0.0050.\n",
      "Up to batch 0, the average loss is    2.36.\n",
      "Up to batch 1, the average loss is    2.91.\n",
      "Up to batch 2, the average loss is    2.63.\n",
      "Up to batch 3, the average loss is    2.93.\n",
      "Up to batch 4, the average loss is    3.17.\n",
      "The average loss for epoch 10 is    3.17 and mean absolute error is    1.36.\n",
      "\n",
      "Epoch 00011: Learning rate is 0.0050.\n",
      "Up to batch 0, the average loss is    3.32.\n",
      "Up to batch 1, the average loss is    3.02.\n",
      "Up to batch 2, the average loss is    2.96.\n",
      "Up to batch 3, the average loss is    2.80.\n",
      "Up to batch 4, the average loss is    2.92.\n",
      "The average loss for epoch 11 is    2.92 and mean absolute error is    1.32.\n",
      "\n",
      "Epoch 00012: Learning rate is 0.0010.\n",
      "Up to batch 0, the average loss is    4.11.\n",
      "Up to batch 1, the average loss is    3.70.\n",
      "Up to batch 2, the average loss is    3.89.\n",
      "Up to batch 3, the average loss is    3.76.\n",
      "Up to batch 4, the average loss is    3.45.\n",
      "The average loss for epoch 12 is    3.45 and mean absolute error is    1.44.\n",
      "\n",
      "Epoch 00013: Learning rate is 0.0010.\n",
      "Up to batch 0, the average loss is    3.38.\n",
      "Up to batch 1, the average loss is    3.34.\n",
      "Up to batch 2, the average loss is    3.26.\n",
      "Up to batch 3, the average loss is    3.56.\n",
      "Up to batch 4, the average loss is    3.62.\n",
      "The average loss for epoch 13 is    3.62 and mean absolute error is    1.44.\n",
      "\n",
      "Epoch 00014: Learning rate is 0.0010.\n",
      "Up to batch 0, the average loss is    2.48.\n",
      "Up to batch 1, the average loss is    2.38.\n",
      "Up to batch 2, the average loss is    2.76.\n",
      "Up to batch 3, the average loss is    2.63.\n",
      "Up to batch 4, the average loss is    2.66.\n",
      "The average loss for epoch 14 is    2.66 and mean absolute error is    1.29.\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "<keras.callbacks.History at 0x7fd08446c290>"
      ]
     },
     "execution_count": 9,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "class CustomLearningRateScheduler(keras.callbacks.Callback):\n",
    "    \"\"\"Learning rate scheduler which sets the learning rate according to schedule.\n",
    "\n",
    "  Arguments:\n",
    "      schedule: a function that takes an epoch index\n",
    "          (integer, indexed from 0) and current learning rate\n",
    "          as inputs and returns a new learning rate as output (float).\n",
    "  \"\"\"\n",
    "\n",
    "    def __init__(self, schedule):\n",
    "        super(CustomLearningRateScheduler, self).__init__()\n",
    "        self.schedule = schedule\n",
    "\n",
    "    def on_epoch_begin(self, epoch, logs=None):\n",
    "        if not hasattr(self.model.optimizer, \"lr\"):\n",
    "            raise ValueError('Optimizer must have a \"lr\" attribute.')\n",
    "        # Get the current learning rate from model's optimizer.\n",
    "        lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))\n",
    "        # Call schedule function to get the scheduled learning rate.\n",
    "        scheduled_lr = self.schedule(epoch, lr)\n",
    "        # Set the value back to the optimizer before this epoch starts\n",
    "        tf.keras.backend.set_value(self.model.optimizer.lr, scheduled_lr)\n",
    "        print(\"\\nEpoch %05d: Learning rate is %6.4f.\" % (epoch, scheduled_lr))\n",
    "\n",
    "\n",
    "LR_SCHEDULE = [\n",
    "    # (epoch to start, learning rate) tuples\n",
    "    (3, 0.05),\n",
    "    (6, 0.01),\n",
    "    (9, 0.005),\n",
    "    (12, 0.001),\n",
    "]\n",
    "\n",
    "\n",
    "def lr_schedule(epoch, lr):\n",
    "    \"\"\"Helper function to retrieve the scheduled learning rate based on epoch.\"\"\"\n",
    "    if epoch < LR_SCHEDULE[0][0] or epoch > LR_SCHEDULE[-1][0]:\n",
    "        return lr\n",
    "    for i in range(len(LR_SCHEDULE)):\n",
    "        if epoch == LR_SCHEDULE[i][0]:\n",
    "            return LR_SCHEDULE[i][1]\n",
    "    return lr\n",
    "\n",
    "\n",
    "model = get_model()\n",
    "model.fit(\n",
    "    x_train,\n",
    "    y_train,\n",
    "    batch_size=64,\n",
    "    steps_per_epoch=5,\n",
    "    epochs=15,\n",
    "    verbose=0,\n",
    "    callbacks=[\n",
    "        LossAndErrorPrintingCallback(),\n",
    "        CustomLearningRateScheduler(lr_schedule),\n",
    "    ],\n",
    ")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "c9be225b57f1"
   },
   "source": [
    "### Built-in Keras callbacks\n",
    "Be sure to check out the existing Keras callbacks by\n",
    "reading the [API docs](https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/).\n",
    "Applications include logging to CSV, saving\n",
    "the model, visualizing metrics in TensorBoard, and a lot more!"
   ]
  }
 ],
 "metadata": {
  "colab": {
   "collapsed_sections": [],
   "name": "custom_callback.ipynb",
   "toc_visible": true
  },
  "kernelspec": {
   "display_name": "Python 3",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 0
}
