# Copyright 2017-2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.
from __future__ import absolute_import

import os
import pytest

from sagemaker.tensorflow import TensorFlow
from sagemaker.utils import sagemaker_timestamp

DISTRIBUTION_ENABLED = {'mpi': {'enabled': True}}

_ECR_TF_CONTAINER_IMAGE = "964029418868.dkr.ecr.us-west-2.amazonaws.com/sagemaker-horovod-distributed-training:1.11.0-cpu-py3"
_ECR_TF_CONTAINER_IMAGE_GPU = "964029418868.dkr.ecr.us-west-2.amazonaws.com/sagemaker-horovod-distributed-training:1.11.0-gpu-py3"


def _get_train_test_data(data_path, sagemaker_session):
    prefix = 'tf_mnist/{}'.format(sagemaker_timestamp())
    train_data_path = os.path.join(data_path, 'train')
    key_prefix = prefix + '/train'
    train_input = sagemaker_session.upload_data(path=train_data_path, key_prefix=key_prefix)
    test_path = os.path.join(data_path, 'test')
    test_input = sagemaker_session.upload_data(path=test_path, key_prefix=prefix + '/test')

    return test_input, train_input


def train(script, instance_type, sagemaker_local_session, docker_image, training_data_path, source_dir,
          train_instance_count, base_job_name):
    estimator = TensorFlow(entry_point=script,
                           role='SageMakerRole',
                           train_instance_count=train_instance_count,
                           train_instance_type=instance_type,
                           sagemaker_session=sagemaker_local_session,
                           image_name=docker_image,
                           base_job_name=base_job_name,
                           source_dir=source_dir,
                           distributions=DISTRIBUTION_ENABLED,
                           framework_version='1.11',
                           script_mode=True)

    estimator.fit(training_data_path, wait=False)


@pytest.mark.parametrize('script,src_dir, instance_count', [
    ('train_mnist_hvd.py', "horovod_mnist", 1),
])
def test_horovod_local(sagemaker_local_session, script, src_dir, instance_count):
    source_dir = os.path.join(os.path.dirname(__file__), '..', 'data', src_dir)
    data_path = os.path.join(source_dir, 'data')

    train(script=script,
          instance_type="local",
          sagemaker_local_session=sagemaker_local_session,
          docker_image="sagemaker-horovod-distributed-training:1.11.0-cpu-py3",
          training_data_path='file://{}'.format(data_path),
          source_dir=source_dir,
          train_instance_count=instance_count,
          base_job_name="tf-hvd-local")


@pytest.mark.parametrize('script,src_dir, instance_count,instance_type', [
    ("train_mnist_hvd.py", "horovod_mnist", 1, "ml.c4.xlarge"),
    ("train_mnist_hvd.py", "horovod_mnist", 2, "ml.c4.xlarge"),
    ("train_mnist_hvd.py", "horovod_mnist", 5, "ml.c4.xlarge"),
])
def test_horovod_sagemaker(sagemaker_session, script, src_dir, instance_count, instance_type):
    source_dir = os.path.join(os.path.dirname(__file__), '..', 'data', src_dir)
    data_path = os.path.join(source_dir, 'data')

    test_input, train_input = _get_train_test_data(data_path, sagemaker_session)

    train(script=script,
          instance_type=instance_type,
          sagemaker_local_session=sagemaker_session,
          docker_image=_ECR_TF_CONTAINER_IMAGE,
          training_data_path={'train': train_input, 'test': test_input},
          source_dir=source_dir,
          train_instance_count=instance_count,
          base_job_name="tf-hvd-{}x".format(script[:-3].replace("_","-")))


@pytest.mark.parametrize('script,src_dir, instance_count,instance_type', [
    ("train_mnist_hvd.py", "horovod_mnist", 1, "ml.p3.2xlarge"),
    ("train_mnist_hvd.py", "horovod_mnist", 2, "ml.p3.2xlarge"),
    ("train_mnist_hvd.py", "horovod_mnist", 5, "ml.p3.2xlarge"),
])
def test_horovod_sagemaker_gpu(sagemaker_session, script, src_dir, instance_count, instance_type):
    source_dir = os.path.join(os.path.dirname(__file__), '..', 'data', src_dir)
    data_path = os.path.join(source_dir, 'data')

    test_input, train_input = _get_train_test_data(data_path, sagemaker_session)

    train(script=script,
          instance_type=instance_type,
          sagemaker_local_session=sagemaker_session,
          docker_image=_ECR_TF_CONTAINER_IMAGE_GPU,
          training_data_path={'train': train_input, 'test': test_input},
          source_dir=source_dir,
          train_instance_count=instance_count,
          base_job_name="tf-hvd-{}x-gpu".format(instance_count))
