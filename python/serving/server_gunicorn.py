#
# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Launcher for the prediction_executor based encoder server.

Uses the servingframework to create a request server which
performs the logic for requests in separate processes and uses a local TFserving
instance to handle the model.
"""

from collections.abc import Sequence
import copy
import os
import pathlib
from typing import Any

from absl import app
from absl import flags
from absl import logging
import jsonschema
import yaml
import transformers

from data_accessors import data_accessor_const
from serving.serving_framework import inline_prediction_executor
from serving.serving_framework import server_gunicorn
from serving.serving_framework.triton import server_health_check
from serving.serving_framework.triton import triton_streaming_server_model_runner
from serving import predictor
from serving import predictor_const

LOCAL_MODEL_PATH_FLAG = flags.DEFINE_string(
    'local_model_path',
    None,
    'The local model path for configuration purposes.',
    required=False,
)
HF_MODEL_FLAG = flags.DEFINE_string(
    'hf_model',
    None,
    'The HF model to use for the server.',
    required=False,
)


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  if 'AIP_HTTP_PORT' not in os.environ:
    raise ValueError(
        'The environment variable AIP_HTTP_PORT needs to be specified.'
    )
  http_port = int(os.environ.get('AIP_HTTP_PORT'))
  options = {
      'bind': f'0.0.0.0:{http_port}',
      'workers': 3,
      'timeout': 120,
  }
  model_rest_port = int(os.environ.get('MODEL_REST_PORT'))
  health_checker = server_health_check.TritonServerHealthCheck(model_rest_port)

  # Get schema validator.
  local_path = os.path.dirname(__file__)
  with open(
      os.path.join(local_path, 'vertex_schemata', 'request.yaml'), 'r'
  ) as f:
    instance_validator = jsonschema.Draft202012Validator(yaml.safe_load(f))

  # Construct conversation to prompt converter callable.
  if (LOCAL_MODEL_PATH_FLAG.value is None) == (HF_MODEL_FLAG.value is None):
    raise ValueError(
        'Exactly one of --local_model_path or --hf_model needs to be specified.'
    )
  if LOCAL_MODEL_PATH_FLAG.value is not None:
    processor = transformers.AutoProcessor.from_pretrained(
        pathlib.Path(LOCAL_MODEL_PATH_FLAG.value),
    )
  else:
    processor = transformers.AutoProcessor.from_pretrained(
        HF_MODEL_FLAG.value,
        token=os.environ.get('HF_TOKEN', None),
    )

  def to_prompt(
      conversation: list[dict[str, Any]], params: dict[str, Any]
  ) -> str:
    """Generates a prompt from a conversation without modifying the original."""
    type_map = {
        img_type: 'image' for img_type in predictor_const.IMAGE_INPUT_TYPES
    } | {
        predictor_const.TEXT_INPUT_TYPE: 'text',
    }
    patch_key = data_accessor_const.InstanceJsonKeys.PATCH_COORDINATES
    revised = [copy.copy(message) for message in conversation]
    for message in revised:
      if 'content' in message:
        if isinstance(message['content'], str):
          continue
        revised_content = []
        for entry in message['content']:
          entry = copy.copy(entry)
          revised_content.append(entry)
          if entry['type'] in predictor_const.IMAGE_INPUT_TYPES:
            image_structure = entry[entry['type']]
            if patch_key in image_structure:
              revised_content.extend(
                  [entry] * (len(image_structure[patch_key]) - 1)
              )
          entry['type'] = type_map.get(entry['type'], entry['type'])
        message['content'] = revised_content

    return processor.apply_chat_template(revised, tokenize=False, **params)

  predictor_instance = predictor.MedGemmaPredictor(
      prompt_converter=to_prompt,
      instance_validator=instance_validator,
      )
  executor = inline_prediction_executor.InlinePredictionExecutor(
      predictor_instance.predict,
      triton_streaming_server_model_runner.TritonStreamingServerModelRunner,
  )
  logging.info('Launching gunicorn application.')
  server_gunicorn.PredictionApplication(
      executor,
      health_check=health_checker,
      options=options,
      instance_input=False,
      additional_routes={
          '/v1/chat/completions': executor,
      }
  ).run()


if __name__ == '__main__':
  app.run(main)
