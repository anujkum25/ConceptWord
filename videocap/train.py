import time
import os
import pprint
import tensorflow as tf
from videocap.datasets.lsmdc import DatasetLSMDC
from videocap.datasets import data_util
from videocap.util import log

from videocap.models.caption_model import CAPGenerator, CAPTrainer
from videocap.models.fib_model import FIBGenerator, FIBTrainer
from videocap.datasets.batch_queue import BatchQueue
from videocap.configuration import ModelConfig, TrainConfig
import json

# For debug purporses
import hickle as hkl
import numpy as np
pp = pprint.PrettyPrinter(indent=2)


MODELS = {
    'CAP': CAPGenerator,
    'FIB': FIBGenerator,
}
MODEL_TRAINERS = {
    'CAP': CAPTrainer,
    'FIB': FIBTrainer,
}

def main(argv):
    model_config = ModelConfig()
    train_config = TrainConfig()

    train_dataset = DatasetLSMDC(dataset_name='train',
                                 image_feature_net=model_config.image_feature_net,
                                 layer=model_config.layer,
                                 max_length=model_config.video_steps,
                                 max_n_videos=None,
                                 data_type=train_config.train_tag,
                                 attr_length=model_config.attr_length)
    validation_dataset = DatasetLSMDC(dataset_name='test',
                                      image_feature_net=model_config.image_feature_net,
                                      layer=model_config.layer,
                                      max_length=model_config.video_steps,
                                      max_n_videos=None,
                                      data_type=train_config.train_tag,
                                      attr_length=model_config.attr_length)
    train_dataset.build_word_vocabulary()
    validation_dataset.share_word_vocabulary_from(train_dataset)

    train_iter = train_dataset.batch_iter(train_config.num_epochs, model_config.batch_size)
    train_queue = BatchQueue(train_iter, name='train')
    val_iter = validation_dataset.batch_iter(20*train_config.num_epochs, model_config.batch_size, shuffle=False)
    val_queue = BatchQueue(val_iter, name='test')
    train_queue.start_threads()
    val_queue.start_threads()

    g = tf.Graph()
    with g.as_default():
        global session, model, trainer
        tf_config = tf.ConfigProto()
        tf_config.gpu_options.allow_growth = True
        session = tf.Session(graph=g, config=tf_config)

        model = MODELS[train_config.train_tag](model_config, train_dataset.word_matrix,
                                               train_dataset.index_1000)

        log.info("Build the model...")
        model.build_model(**model.get_placeholder())
        trainer = MODEL_TRAINERS[train_config.train_tag](train_config, model, session)

        session.run(tf.global_variables_initializer())
        steps_in_epoch = int(np.ceil(len(train_dataset) / model.batch_size))

        for step in range(train_config.max_steps):
            step_result = trainer.run_single_step(
                queue=train_queue, is_train=True)

            if step_result['current_step'] % train_config.steps_per_logging == 0:
                step_result['steps_in_epoch'] = steps_in_epoch
                trainer.log_step_message(**step_result)

            if step_result['current_step'] % train_config.steps_per_evaluate == 0:
                trainer.evaluate(queue=val_queue, dataset=validation_dataset)

        train_queue.thread_close()
        val_queue.thread_close()

if __name__ == '__main__':
    tf.app.run(main=main)
