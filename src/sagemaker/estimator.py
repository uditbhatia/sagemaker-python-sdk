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
from __future__ import print_function, absolute_import

import json
import logging
import os
import warnings
from abc import ABCMeta
from abc import abstractmethod
from six import with_metaclass

from sagemaker.analytics import TrainingJobAnalytics
from sagemaker.fw_utils import (create_image_uri, tar_and_upload_dir, parse_s3_url, UploadedCode,
                                validate_source_dir)
from sagemaker.job import _Job
from sagemaker.local import LocalSession
from sagemaker.model import Model
from sagemaker.model import (SCRIPT_PARAM_NAME, DIR_PARAM_NAME, CLOUDWATCH_METRICS_PARAM_NAME,
                             CONTAINER_LOG_LEVEL_PARAM_NAME, JOB_NAME_PARAM_NAME, SAGEMAKER_REGION_PARAM_NAME)
from sagemaker.predictor import RealTimePredictor
from sagemaker.session import Session
from sagemaker.session import s3_input
from sagemaker.transformer import Transformer
from sagemaker.utils import base_name_from_image, name_from_base, name_from_image, get_config_value


class EstimatorBase(with_metaclass(ABCMeta, object)):
    """Handle end-to-end Amazon SageMaker training and deployment tasks.

    For introduction to model training and deployment, see
    http://docs.aws.amazon.com/sagemaker/latest/dg/how-it-works-training.html

    Subclasses must define a way to determine what image to use for training,
    what hyperparameters to use, and how to create an appropriate predictor instance.
    """

    def __init__(self, role, train_instance_count, train_instance_type,
                 train_volume_size=30, train_volume_kms_key=None, train_max_run=24 * 60 * 60, input_mode='File',
                 output_path=None, output_kms_key=None, base_job_name=None, sagemaker_session=None, tags=None,
                 subnets=None, security_group_ids=None):
        """Initialize an ``EstimatorBase`` instance.

        Args:
            role (str): An AWS IAM role (either name or full ARN). The Amazon SageMaker training jobs and APIs
                that create Amazon SageMaker endpoints use this role to access training data and model artifacts.
                After the endpoint is created, the inference code might use the IAM role,
                if it needs to access an AWS resource.
            train_instance_count (int): Number of Amazon EC2 instances to use for training.
            train_instance_type (str): Type of EC2 instance to use for training, for example, 'ml.c4.xlarge'.
            train_volume_size (int): Size in GB of the EBS volume to use for storing input data
                during training (default: 30). Must be large enough to store training data if File Mode is used
                (which is the default).
            train_volume_kms_key (str): Optional. KMS key ID for encrypting EBS volume attached to the
                training instance (default: None).
            train_max_run (int): Timeout in seconds for training (default: 24 * 60 * 60).
                After this amount of time Amazon SageMaker terminates the job regardless of its current status.
            input_mode (str): The input mode that the algorithm supports (default: 'File'). Valid modes:
                'File' - Amazon SageMaker copies the training dataset from the S3 location to a local directory.
                'Pipe' - Amazon SageMaker streams data directly from S3 to the container via a Unix-named pipe.
            output_path (str): S3 location for saving the trainig result (model artifacts and output files).
                If not specified, results are stored to a default bucket. If the bucket with the specific name
                does not exist, the estimator creates the bucket during the
                :meth:`~sagemaker.estimator.EstimatorBase.fit` method execution.
            output_kms_key (str): Optional. KMS key ID for encrypting the training output (default: None).
            base_job_name (str): Prefix for training job name when the :meth:`~sagemaker.estimator.EstimatorBase.fit`
                method launches. If not specified, the estimator generates a default job name, based on
                the training image name and current timestamp.
            sagemaker_session (sagemaker.session.Session): Session object which manages interactions with
                Amazon SageMaker APIs and any other AWS services needed. If not specified, the estimator creates one
                using the default AWS configuration chain.
            tags (list[dict]): List of tags for labeling a training job. For more, see
                https://docs.aws.amazon.com/sagemaker/latest/dg/API_Tag.html.
            subnets (list[str]): List of subnet ids. If not specified training job will be created without VPC config.
            security_group_ids (list[str]): List of security group ids. If not specified training job will be created
                without VPC config.
        """
        self.role = role
        self.train_instance_count = train_instance_count
        self.train_instance_type = train_instance_type
        self.train_volume_size = train_volume_size
        self.train_volume_kms_key = train_volume_kms_key
        self.train_max_run = train_max_run
        self.input_mode = input_mode
        self.tags = tags

        if self.train_instance_type in ('local', 'local_gpu'):
            if self.train_instance_type == 'local_gpu' and self.train_instance_count > 1:
                raise RuntimeError("Distributed Training in Local GPU is not supported")
            self.sagemaker_session = sagemaker_session or LocalSession()
        else:
            self.sagemaker_session = sagemaker_session or Session()

        self.base_job_name = base_job_name
        self._current_job_name = None
        self.output_path = output_path
        self.output_kms_key = output_kms_key
        self.latest_training_job = None

        # VPC configurations
        self.subnets = subnets
        self.security_group_ids = security_group_ids

    @abstractmethod
    def train_image(self):
        """Return the Docker image to use for training.

        The  :meth:`~sagemaker.estimator.EstimatorBase.fit` method, which does the model training, calls this method to
        find the image to use for model training.

        Returns:
            str: The URI of the Docker image.
        """
        pass

    @abstractmethod
    def hyperparameters(self):
        """Return the hyperparameters as a dictionary to use for training.

        The  :meth:`~sagemaker.estimator.EstimatorBase.fit` method, which trains the model, calls this method to
        find the hyperparameters.

        Returns:
            dict[str, str]: The hyperparameters.
        """
        pass

    def _prepare_for_training(self, job_name=None):
        """Set any values in the estimator that need to be set before training.

        Args:
            * job_name (str): Name of the training job to be created. If not specified, one is generated,
                using the base name given to the constructor if applicable.
        """
        if job_name is not None:
            self._current_job_name = job_name
        else:
            # honor supplied base_job_name or generate it
            base_name = self.base_job_name or base_name_from_image(self.train_image())
            self._current_job_name = name_from_base(base_name)

        # if output_path was specified we use it otherwise initialize here.
        # For Local Mode with local_code=True we don't need an explicit output_path
        if self.output_path is None:
            local_code = get_config_value('local.local_code', self.sagemaker_session.config)
            if self.sagemaker_session.local_mode and local_code:
                self.output_path = ''
            else:
                self.output_path = 's3://{}/'.format(self.sagemaker_session.default_bucket())

    def fit(self, inputs, wait=True, logs=True, job_name=None):
        """Train a model using the input training dataset.

        The API calls the Amazon SageMaker CreateTrainingJob API to start model training.
        The API uses configuration you provided to create the estimator and the
        specified input training data to send the CreatingTrainingJob request to Amazon SageMaker.

        This is a synchronous operation. After the model training successfully completes,
        you can call the ``deploy()`` method to host the model using the Amazon SageMaker hosting services.

        Args:
            inputs (str or dict or sagemaker.session.s3_input): Information about the training data.
                This can be one of three types:

                * (str) the S3 location where training data is saved.

                * (dict[str, str] or dict[str, sagemaker.session.s3_input]) If using multiple channels for
                    training data, you can specify a dict mapping channel names
                    to strings or :func:`~sagemaker.session.s3_input` objects.
                * (sagemaker.session.s3_input) - channel configuration for S3 data sources that can provide
                    additional information as well as the path to the training dataset.
                    See :func:`sagemaker.session.s3_input` for full details.
            wait (bool): Whether the call should wait until the job completes (default: True).
            logs (bool): Whether to show the logs produced by the job.
                Only meaningful when wait is True (default: True).
            job_name (str): Training job name. If not specified, the estimator generates a default job name,
                based on the training image name and current timestamp.
        """
        self._prepare_for_training(job_name=job_name)

        self.latest_training_job = _TrainingJob.start_new(self, inputs)
        if wait:
            self.latest_training_job.wait(logs=logs)

    @classmethod
    def _from_training_job(cls, init_params, hyperparameters, image, sagemaker_session):
        """Create an Estimator from existing training job data.

        Args:
            init_params (dict): The init_params the training job was created with.
            hyperparameters (dict):  The hyperparameters the training job was created with.
            image (str): Container image (if any) the training job was created with
            sagemaker_session (sagemaker.session.Session): A sagemaker Session to pass to the estimator.

        Returns: An instance of the calling Estimator Class.

        """
        raise NotImplementedError()

    @classmethod
    def attach(cls, training_job_name, sagemaker_session=None):
        """Attach to an existing training job.

        Create an Estimator bound to an existing training job, each subclass is responsible to implement
        ``_prepare_init_params_from_job_description()`` as this method delegates the actual conversion of a training
        job description to the arguments that the class constructor expects. After attaching, if the training job has a
        Complete status, it can be ``deploy()`` ed to create a SageMaker Endpoint and return a ``Predictor``.

        If the training job is in progress, attach will block and display log messages
        from the training job, until the training job completes.

        Args:
            training_job_name (str): The name of the training job to attach to.
            sagemaker_session (sagemaker.session.Session): Session object which manages interactions with
                Amazon SageMaker APIs and any other AWS services needed. If not specified, the estimator creates one
                using the default AWS configuration chain.

        Examples:
            >>> my_estimator.fit(wait=False)
            >>> training_job_name = my_estimator.latest_training_job.name
            Later on:
            >>> attached_estimator = Estimator.attach(training_job_name)
            >>> attached_estimator.deploy()

        Returns:
            Instance of the calling ``Estimator`` Class with the attached training job.
        """
        sagemaker_session = sagemaker_session or Session()

        job_details = sagemaker_session.sagemaker_client.describe_training_job(TrainingJobName=training_job_name)
        init_params = cls._prepare_init_params_from_job_description(job_details)

        estimator = cls(sagemaker_session=sagemaker_session, **init_params)
        estimator.latest_training_job = _TrainingJob(sagemaker_session=sagemaker_session,
                                                     training_job_name=init_params['base_job_name'])
        estimator.latest_training_job.wait()
        return estimator

    def deploy(self, initial_instance_count, instance_type, endpoint_name=None, **kwargs):
        """Deploy the trained model to an Amazon SageMaker endpoint and return a ``sagemaker.RealTimePredictor`` object.

        More information:
        http://docs.aws.amazon.com/sagemaker/latest/dg/how-it-works-training.html

        Args:
            initial_instance_count (int): Minimum number of EC2 instances to deploy to an endpoint for prediction.
            instance_type (str): Type of EC2 instance to deploy to an endpoint for prediction,
                for example, 'ml.c4.xlarge'.
            endpoint_name (str): Name to use for creating an Amazon SageMaker endpoint. If not specified, the name of
                the training job is used.
            **kwargs: Passed to invocation of ``create_model()``. Implementations may customize
                ``create_model()`` to accept ``**kwargs`` to customize model creation during deploy.
                For more, see the implementation docs.

        Returns:
            sagemaker.predictor.RealTimePredictor: A predictor that provides a ``predict()`` method,
                which can be used to send requests to the Amazon SageMaker endpoint and obtain inferences.
        """
        self._ensure_latest_training_job()
        endpoint_name = endpoint_name or self.latest_training_job.name
        self.deploy_instance_type = instance_type
        return self.create_model(**kwargs).deploy(
            instance_type=instance_type,
            initial_instance_count=initial_instance_count,
            endpoint_name=endpoint_name)

    @property
    def model_data(self):
        """str: The model location in S3. Only set if Estimator has been ``fit()``."""
        return self.sagemaker_session.sagemaker_client.describe_training_job(
            TrainingJobName=self.latest_training_job.name)['ModelArtifacts']['S3ModelArtifacts']

    @abstractmethod
    def create_model(self, **kwargs):
        """Create a SageMaker ``Model`` object that can be deployed to an ``Endpoint``.

        Args:
            **kwargs: Keyword arguments used by the implemented method for creating the ``Model``.

        Returns:
            sagemaker.model.Model: A SageMaker ``Model`` object. See :func:`~sagemaker.model.Model` for full details.
        """
        pass

    @classmethod
    def _prepare_init_params_from_job_description(cls, job_details):
        """Convert the job description to init params that can be handled by the class constructor

        Args:
            job_details: the returned job details from a describe_training_job API call.

        Returns:
             dictionary: The transformed init_params

        """
        init_params = dict()

        init_params['role'] = job_details['RoleArn']
        init_params['train_instance_count'] = job_details['ResourceConfig']['InstanceCount']
        init_params['train_instance_type'] = job_details['ResourceConfig']['InstanceType']
        init_params['train_volume_size'] = job_details['ResourceConfig']['VolumeSizeInGB']
        init_params['train_max_run'] = job_details['StoppingCondition']['MaxRuntimeInSeconds']
        init_params['input_mode'] = job_details['AlgorithmSpecification']['TrainingInputMode']
        init_params['base_job_name'] = job_details['TrainingJobName']
        init_params['output_path'] = job_details['OutputDataConfig']['S3OutputPath']
        init_params['output_kms_key'] = job_details['OutputDataConfig']['KmsKeyId']

        init_params['hyperparameters'] = job_details['HyperParameters']
        init_params['image'] = job_details['AlgorithmSpecification']['TrainingImage']

        return init_params

    def delete_endpoint(self):
        """Delete an Amazon SageMaker ``Endpoint``.

        Raises:
            ValueError: If the endpoint does not exist.
        """
        self._ensure_latest_training_job(error_message='Endpoint was not created yet')
        self.sagemaker_session.delete_endpoint(self.latest_training_job.name)

    def transformer(self, instance_count, instance_type, strategy=None, assemble_with=None, output_path=None,
                    output_kms_key=None, accept=None, env=None, max_concurrent_transforms=None,
                    max_payload=None, tags=None, role=None):
        """Return a ``Transformer`` that uses a SageMaker Model based on the training job. It reuses the
        SageMaker Session and base job name used by the Estimator.

        Args:
            instance_count (int): Number of EC2 instances to use.
            instance_type (str): Type of EC2 instance to use, for example, 'ml.c4.xlarge'.
            strategy (str): The strategy used to decide how to batch records in a single request (default: None).
                Valid values: 'MULTI_RECORD' and 'SINGLE_RECORD'.
            assemble_with (str): How the output is assembled (default: None). Valid values: 'Line' or 'None'.
            output_path (str): S3 location for saving the transform result. If not specified, results are stored to
                a default bucket.
            output_kms_key (str): Optional. KMS key ID for encrypting the transform output (default: None).
            accept (str): The content type accepted by the endpoint deployed during the transform job.
            env (dict): Environment variables to be set for use during the transform job (default: None).
            max_concurrent_transforms (int): The maximum number of HTTP requests to be made to
                each individual transform container at one time.
            max_payload (int): Maximum size of the payload in a single HTTP request to the container in MB.
            tags (list[dict]): List of tags for labeling a transform job. If none specified, then the tags used for
                the training job are used for the transform job.
            role (str): The ``ExecutionRoleArn`` IAM Role ARN for the ``Model``, which is also used during
                transform jobs. If not specified, the role from the Estimator will be used.
        """
        self._ensure_latest_training_job()

        model_name = self.sagemaker_session.create_model_from_job(self.latest_training_job.name, role=role)
        tags = tags or self.tags

        return Transformer(model_name, instance_count, instance_type, strategy=strategy, assemble_with=assemble_with,
                           output_path=output_path, output_kms_key=output_kms_key, accept=accept,
                           max_concurrent_transforms=max_concurrent_transforms, max_payload=max_payload,
                           env=env, tags=tags, base_transform_job_name=self.base_job_name,
                           sagemaker_session=self.sagemaker_session)

    @property
    def training_job_analytics(self):
        """Return a ``TrainingJobAnalytics`` object for the current training job.
        """
        if self._current_job_name is None:
            raise ValueError('Estimator is not associated with a TrainingJob')
        return TrainingJobAnalytics(self._current_job_name, sagemaker_session=self.sagemaker_session)

    def _ensure_latest_training_job(self, error_message='Estimator is not associated with a training job'):
        if self.latest_training_job is None:
            raise ValueError(error_message)


class _TrainingJob(_Job):
    def __init__(self, sagemaker_session, training_job_name):
        super(_TrainingJob, self).__init__(sagemaker_session, training_job_name)

    @classmethod
    def start_new(cls, estimator, inputs):
        """Create a new Amazon SageMaker training job from the estimator.

        Args:
            estimator (sagemaker.estimator.EstimatorBase): Estimator object created by the user.
            inputs (str): Parameters used when called  :meth:`~sagemaker.estimator.EstimatorBase.fit`.

        Returns:
            sagemaker.estimator._TrainingJob: Constructed object that captures all information about the started
            training job.
        """

        local_mode = estimator.sagemaker_session.local_mode

        # Allow file:// input only in local mode
        if isinstance(inputs, str) and inputs.startswith('file://'):
            if not local_mode:
                raise ValueError('File URIs are supported in local mode only. Please use a S3 URI instead.')

        config = _Job._load_config(inputs, estimator)

        if estimator.hyperparameters() is not None:
            hyperparameters = {str(k): str(v) for (k, v) in estimator.hyperparameters().items()}

        estimator.sagemaker_session.train(image=estimator.train_image(), input_mode=estimator.input_mode,
                                          input_config=config['input_config'], role=config['role'],
                                          job_name=estimator._current_job_name, output_config=config['output_config'],
                                          resource_config=config['resource_config'], vpc_config=config['vpc_config'],
                                          hyperparameters=hyperparameters, stop_condition=config['stop_condition'],
                                          tags=estimator.tags)

        return cls(estimator.sagemaker_session, estimator._current_job_name)

    def wait(self, logs=True):
        if logs:
            self.sagemaker_session.logs_for_job(self.job_name, wait=True)
        else:
            self.sagemaker_session.wait_for_job(self.job_name)


class Estimator(EstimatorBase):
    """
    A generic Estimator to train using any supplied algorithm. This class is designed for use with
    algorithms that don't have their own, custom class.
    """

    def __init__(self, image_name, role, train_instance_count, train_instance_type,
                 train_volume_size=30, train_volume_kms_key=None, train_max_run=24 * 60 * 60,
                 input_mode='File', output_path=None, output_kms_key=None, base_job_name=None,
                 sagemaker_session=None, hyperparameters=None, tags=None, subnets=None, security_group_ids=None):
        """Initialize an ``Estimator`` instance.

        Args:
            image_name (str): The container image to use for training.
            role (str): An AWS IAM role (either name or full ARN). The Amazon SageMaker training jobs and APIs
                that create Amazon SageMaker endpoints use this role to access training data and model artifacts.
                After the endpoint is created, the inference code might use the IAM role,
                if it needs to access an AWS resource.
            train_instance_count (int): Number of Amazon EC2 instances to use for training.
            train_instance_type (str): Type of EC2 instance to use for training, for example, 'ml.c4.xlarge'.
            train_volume_size (int): Size in GB of the EBS volume to use for storing input data
                during training (default: 30). Must be large enough to store training data if File Mode is used
                (which is the default).
            train_volume_kms_key (str): Optional. KMS key ID for encrypting EBS volume attached to the
                training instance (default: None).
            train_max_run (int): Timeout in seconds for training (default: 24 * 60 * 60).
                After this amount of time Amazon SageMaker terminates the job regardless of its current status.
            input_mode (str): The input mode that the algorithm supports (default: 'File'). Valid modes:

                * 'File' - Amazon SageMaker copies the training dataset from the S3 location to a local directory.
                * 'Pipe' - Amazon SageMaker streams data directly from S3 to the container via a Unix-named pipe.

            output_path (str): S3 location for saving the trainig result (model artifacts and output files).
                If not specified, results are stored to a default bucket. If the bucket with the specific name
                does not exist, the estimator creates the bucket during the
                :meth:`~sagemaker.estimator.EstimatorBase.fit` method execution.
            output_kms_key (str): Optional. KMS key ID for encrypting the training output (default: None).
            base_job_name (str): Prefix for training job name when the :meth:`~sagemaker.estimator.EstimatorBase.fit`
                method launches. If not specified, the estimator generates a default job name, based on
                the training image name and current timestamp.
            sagemaker_session (sagemaker.session.Session): Session object which manages interactions with
                Amazon SageMaker APIs and any other AWS services needed. If not specified, the estimator creates one
                using the default AWS configuration chain.
            hyperparameters (dict): Dictionary containing the hyperparameters to initialize this estimator with.
            tags (list[dict]): List of tags for labeling a training job. For more, see
                https://docs.aws.amazon.com/sagemaker/latest/dg/API_Tag.html.
            subnets (list[str]): List of subnet ids. If not specified training job will be created without VPC config.
            security_group_ids (list[str]): List of security group ids. If not specified training job will be created
                without VPC config.
        """
        self.image_name = image_name
        self.hyperparam_dict = hyperparameters.copy() if hyperparameters else {}
        super(Estimator, self).__init__(role, train_instance_count, train_instance_type,
                                        train_volume_size, train_volume_kms_key, train_max_run, input_mode,
                                        output_path, output_kms_key, base_job_name, sagemaker_session,
                                        tags, subnets, security_group_ids)

    def train_image(self):
        """
        Returns the docker image to use for training.

        The fit() method, that does the model training, calls this method to find the image to use for model training.
        """
        return self.image_name

    def set_hyperparameters(self, **kwargs):
        for k, v in kwargs.items():
            self.hyperparam_dict[k] = v

    def hyperparameters(self):
        """Returns the hyperparameters as a dictionary to use for training.

       The fit() method, that does the model training, calls this method to find the hyperparameters you specified.
        """
        return self.hyperparam_dict

    def create_model(self, role=None, image=None, predictor_cls=None, serializer=None, deserializer=None,
                     content_type=None, accept=None, **kwargs):
        """
        Create a model to deploy.

        Args:
            role (str): The ``ExecutionRoleArn`` IAM Role ARN for the ``Model``, which is also used during
                transform jobs. If not specified, the role from the Estimator will be used.
            image (str): An container image to use for deploying the model. Defaults to the image used for training.
            predictor_cls (RealTimePredictor): The predictor class to use when deploying the model.
            serializer (callable): Should accept a single argument, the input data, and return a sequence
                of bytes. May provide a content_type attribute that defines the endpoint request content type
            deserializer (callable): Should accept two arguments, the result data and the response content type,
                and return a sequence of bytes. May provide a content_type attribute that defines th endpoint
                response Accept content type.
            content_type (str): The invocation ContentType, overriding any content_type from the serializer
            accept (str): The invocation Accept, overriding any accept from the deserializer.

            The serializer, deserializer, content_type, and accept arguments are only used to define a default
            RealTimePredictor. They are ignored if an explicit predictor class is passed in. Other arguments
            are passed through to the Model class.

        Returns: a Model ready for deployment.
        """
        if predictor_cls is None:
            def predict_wrapper(endpoint, session):
                return RealTimePredictor(endpoint, session, serializer, deserializer, content_type, accept)
            predictor_cls = predict_wrapper

        role = role or self.role

        return Model(self.model_data, image or self.train_image(), role, sagemaker_session=self.sagemaker_session,
                     predictor_cls=predictor_cls, **kwargs)

    @classmethod
    def _prepare_init_params_from_job_description(cls, job_details):
        """Convert the job description to init params that can be handled by the class constructor

        Args:
            job_details: the returned job details from a describe_training_job API call.

        Returns:
             dictionary: The transformed init_params

        """
        init_params = super(Estimator, cls)._prepare_init_params_from_job_description(job_details)

        init_params['image_name'] = init_params.pop('image')
        return init_params


class Framework(EstimatorBase):
    """Base class that cannot be instantiated directly.

    Subclasses define functionality pertaining to specific ML frameworks,
    such as training/deployment images and predictor instances.
    """

    def __init__(self, entry_point, source_dir=None, hyperparameters=None, enable_cloudwatch_metrics=False,
                 container_log_level=logging.INFO, code_location=None, image_name=None, **kwargs):
        """Base class initializer. Subclasses which override ``__init__`` should invoke ``super()``

        Args:
            entry_point (str): Path (absolute or relative) to the Python source file which should be executed
                as the entry point to training. This should be compatible with either Python 2.7 or Python 3.5.
            source_dir (str): Path (absolute or relative) to a directory with any other training
                source code dependencies aside from tne entry point file (default: None). Structure within this
                directory are preserved when training on Amazon SageMaker.
            hyperparameters (dict): Hyperparameters that will be used for training (default: None).
                The hyperparameters are made accessible as a dict[str, str] to the training code on SageMaker.
                For convenience, this accepts other types for keys and values, but ``str()`` will be called
                to convert them before training.
            enable_cloudwatch_metrics (bool): [DEPRECATED] Now there are cloudwatch metrics emitted by all SageMaker
                training jobs. This will be ignored for now and removed in a further release.
            container_log_level (int): Log level to use within the container (default: logging.INFO).
                Valid values are defined in the Python logging module.
            code_location (str): Name of the S3 bucket where custom code is uploaded (default: None).
                If not specified, default bucket created by ``sagemaker.session.Session`` is used.
            **kwargs: Additional kwargs passed to the ``EstimatorBase`` constructor.
            image_name (str): An alternate image name to use instead of the official Sagemaker image
                for the framework. This is useful to run one of the Sagemaker supported frameworks
                with an image containing custom dependencies.
        """
        super(Framework, self).__init__(**kwargs)
        self.source_dir = source_dir
        self.entry_point = entry_point
        if enable_cloudwatch_metrics:
            warnings.warn('enable_cloudwatch_metrics is now deprecated and will be removed in the future.',
                          DeprecationWarning)
        self.enable_cloudwatch_metrics = False
        self.container_log_level = container_log_level
        self._hyperparameters = hyperparameters or {}
        self.code_location = code_location
        self.image_name = image_name

    def _prepare_for_training(self, job_name=None):
        """Set hyperparameters needed for training. This method will also validate ``source_dir``.

        Args:
            * job_name (str): Name of the training job to be created. If not specified, one is generated,
                using the base name given to the constructor if applicable.
        """
        super(Framework, self)._prepare_for_training(job_name=job_name)

        # validate source dir will raise a ValueError if there is something wrong with the
        # source directory. We are intentionally not handling it because this is a critical error.
        if self.source_dir and not self.source_dir.lower().startswith('s3://'):
            validate_source_dir(self.entry_point, self.source_dir)

        # if we are in local mode with local_code=True. We want the container to just
        # mount the source dir instead of uploading to S3.
        local_code = get_config_value('local.local_code', self.sagemaker_session.config)
        if self.sagemaker_session.local_mode and local_code:
            # if there is no source dir, use the directory containing the entry point.
            if self.source_dir is None:
                self.source_dir = os.path.dirname(self.entry_point)
            self.entry_point = os.path.basename(self.entry_point)

            code_dir = 'file://' + self.source_dir
            script = self.entry_point
        else:
            self.uploaded_code = self._stage_user_code_in_s3()
            code_dir = self.uploaded_code.s3_prefix
            script = self.uploaded_code.script_name

        # Modify hyperparameters in-place to point to the right code directory and script URIs
        self._hyperparameters[DIR_PARAM_NAME] = code_dir
        self._hyperparameters[SCRIPT_PARAM_NAME] = script
        self._hyperparameters[CLOUDWATCH_METRICS_PARAM_NAME] = self.enable_cloudwatch_metrics
        self._hyperparameters[CONTAINER_LOG_LEVEL_PARAM_NAME] = self.container_log_level
        self._hyperparameters[JOB_NAME_PARAM_NAME] = self._current_job_name
        self._hyperparameters[SAGEMAKER_REGION_PARAM_NAME] = self.sagemaker_session.boto_region_name

    def _stage_user_code_in_s3(self):
        """Upload the user training script to s3 and return the location.

        Returns: s3 uri

        """
        if self.code_location is None:
            code_bucket = self.sagemaker_session.default_bucket()
            code_s3_prefix = '{}/source'.format(self._current_job_name)
        else:
            code_bucket, key_prefix = parse_s3_url(self.code_location)
            code_s3_prefix = '/'.join(filter(None, [key_prefix, self._current_job_name, 'source']))

        return tar_and_upload_dir(session=self.sagemaker_session.boto_session,
                                  bucket=code_bucket,
                                  s3_key_prefix=code_s3_prefix,
                                  script=self.entry_point,
                                  directory=self.source_dir)

    def _model_source_dir(self):
        """Get the appropriate value to pass as source_dir to model constructor on deploying

        Returns:
            str: Either a local or an S3 path pointing to the source_dir to be used for code by the model to be deployed
        """
        return self.source_dir if self.sagemaker_session.local_mode else self.uploaded_code.s3_prefix

    def hyperparameters(self):
        """Return the hyperparameters as a dictionary to use for training.

        The  :meth:`~sagemaker.estimator.EstimatorBase.fit` method, which trains the model, calls this method
        to find the hyperparameters.

        Returns:
            dict[str, str]: The hyperparameters.
        """
        return self._json_encode_hyperparameters(self._hyperparameters)

    @classmethod
    def _prepare_init_params_from_job_description(cls, job_details):
        """Convert the job description to init params that can be handled by the class constructor

        Args:
            job_details: the returned job details from a describe_training_job API call.

        Returns:
             dictionary: The transformed init_params

        """
        init_params = super(Framework, cls)._prepare_init_params_from_job_description(job_details)

        init_params['entry_point'] = json.loads(init_params['hyperparameters'].get(SCRIPT_PARAM_NAME))
        init_params['source_dir'] = json.loads(init_params['hyperparameters'].get(DIR_PARAM_NAME))
        init_params['enable_cloudwatch_metrics'] = json.loads(
            init_params['hyperparameters'].get(CLOUDWATCH_METRICS_PARAM_NAME))
        init_params['container_log_level'] = json.loads(
            init_params['hyperparameters'].get(CONTAINER_LOG_LEVEL_PARAM_NAME))

        hyperparameters = {}
        for k, v in init_params['hyperparameters'].items():
            # Tuning jobs add this special hyperparameter which is not JSON serialized
            if k == '_tuning_objective_metric':
                if v.startswith('"') and v.endswith('"'):
                    v = v.strip('"')
                hyperparameters[k] = v
            else:
                hyperparameters[k] = json.loads(v)

        init_params['hyperparameters'] = hyperparameters

        return init_params

    def train_image(self):
        """Return the Docker image to use for training.

        The  :meth:`~sagemaker.estimator.EstimatorBase.fit` method, which does the model training,
        calls this method to find the image to use for model training.

        Returns:
            str: The URI of the Docker image.
        """
        if self.image_name:
            return self.image_name
        else:
            return create_image_uri(self.sagemaker_session.boto_region_name, self.__framework_name__,
                                    self.train_instance_type, self.framework_version, py_version=self.py_version)

    @classmethod
    def attach(cls, training_job_name, sagemaker_session=None):
        """Attach to an existing training job.

        Create an Estimator bound to an existing training job, each subclass is responsible to implement
        ``_prepare_init_params_from_job_description()`` as this method delegates the actual conversion of a training
        job description to the arguments that the class constructor expects. After attaching, if the training job has a
        Complete status, it can be ``deploy()`` ed to create a SageMaker Endpoint and return a ``Predictor``.

        If the training job is in progress, attach will block and display log messages
        from the training job, until the training job completes.

        Args:
            training_job_name (str): The name of the training job to attach to.
            sagemaker_session (sagemaker.session.Session): Session object which manages interactions with
                Amazon SageMaker APIs and any other AWS services needed. If not specified, the estimator creates one
                using the default AWS configuration chain.

        Examples:
            >>> my_estimator.fit(wait=False)
            >>> training_job_name = my_estimator.latest_training_job.name
            Later on:
            >>> attached_estimator = Estimator.attach(training_job_name)
            >>> attached_estimator.deploy()

        Returns:
            Instance of the calling ``Estimator`` Class with the attached training job.
        """
        estimator = super(Framework, cls).attach(training_job_name, sagemaker_session)
        estimator.uploaded_code = UploadedCode(estimator.source_dir, estimator.entry_point)
        return estimator

    @staticmethod
    def _json_encode_hyperparameters(hyperparameters):
        return {str(k): json.dumps(v) for (k, v) in hyperparameters.items()}

    @classmethod
    def _update_init_params(cls, hp, tf_arguments):
        updated_params = {}
        for argument in tf_arguments:
            value = hp.pop(argument, None)
            if value is not None:
                value = json.loads(value)
                updated_params[argument] = value
        return updated_params

    def transformer(self, instance_count, instance_type, strategy=None, assemble_with=None, output_path=None,
                    output_kms_key=None, accept=None, env=None, max_concurrent_transforms=None,
                    max_payload=None, tags=None, role=None, model_server_workers=None):
        """Return a ``Transformer`` that uses a SageMaker Model based on the training job. It reuses the
        SageMaker Session and base job name used by the Estimator.

        Args:
            instance_count (int): Number of EC2 instances to use.
            instance_type (str): Type of EC2 instance to use, for example, 'ml.c4.xlarge'.
            strategy (str): The strategy used to decide how to batch records in a single request (default: None).
                Valid values: 'MULTI_RECORD' and 'SINGLE_RECORD'.
            assemble_with (str): How the output is assembled (default: None). Valid values: 'Line' or 'None'.
            output_path (str): S3 location for saving the transform result. If not specified, results are stored to
                a default bucket.
            output_kms_key (str): Optional. KMS key ID for encrypting the transform output (default: None).
            accept (str): The content type accepted by the endpoint deployed during the transform job.
            env (dict): Environment variables to be set for use during the transform job (default: None).
            max_concurrent_transforms (int): The maximum number of HTTP requests to be made to
                each individual transform container at one time.
            max_payload (int): Maximum size of the payload in a single HTTP request to the container in MB.
            tags (list[dict]): List of tags for labeling a transform job. If none specified, then the tags used for
                the training job are used for the transform job.
            role (str): The ``ExecutionRoleArn`` IAM Role ARN for the ``Model``, which is also used during
                transform jobs. If not specified, the role from the Estimator will be used.
            model_server_workers (int): Optional. The number of worker processes used by the inference server.
                If None, server will use one worker per vCPU.
        """
        self._ensure_latest_training_job()
        role = role or self.role

        model = self.create_model(role=role, model_server_workers=model_server_workers)

        container_def = model.prepare_container_def(instance_type)
        model_name = model.name or name_from_image(container_def['Image'])
        self.sagemaker_session.create_model(model_name, role, container_def)

        transform_env = model.env.copy()
        if env is not None:
            transform_env.update(env)

        tags = tags or self.tags
        return Transformer(model_name, instance_count, instance_type, strategy=strategy, assemble_with=assemble_with,
                           output_path=output_path, output_kms_key=output_kms_key, accept=accept,
                           max_concurrent_transforms=max_concurrent_transforms, max_payload=max_payload,
                           env=transform_env, tags=tags, base_transform_job_name=self.base_job_name,
                           sagemaker_session=self.sagemaker_session)


def _s3_uri_prefix(channel_name, s3_data):
    if isinstance(s3_data, s3_input):
        s3_uri = s3_data.config['DataSource']['S3DataSource']['S3Uri']
    else:
        s3_uri = s3_data
    if not s3_uri.startswith('s3://'):
        raise ValueError('Expecting an s3 uri. Got {}'.format(s3_uri))
    return {channel_name: s3_uri[5:]}


# E.g. 's3://bucket/data' would return 'bucket/data'.
# Also accepts other valid input types, e.g. dict and s3_input.
def _s3_uri_without_prefix_from_input(input_data):
    # Unpack an input_config object from a dict if a dict was passed in.
    if isinstance(input_data, dict):
        response = {}
        for channel_name, channel_s3_uri in input_data.items():
            response.update(_s3_uri_prefix(channel_name, channel_s3_uri))
        return response
    elif isinstance(input_data, str):
        return _s3_uri_prefix('training', input_data)
    elif isinstance(input_data, s3_input):
        return _s3_uri_prefix('training', input_data)
    else:
        raise ValueError('Unrecognized type for S3 input data config - not str or s3_input: {}'.format(input_data))
