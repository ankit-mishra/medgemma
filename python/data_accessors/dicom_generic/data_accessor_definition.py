# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Request dataclasses for DICOM generic data accessor."""

import dataclasses
import json
from typing import Any, List, Mapping

from ez_wsi_dicomweb import credential_factory as credential_factory_module
from ez_wsi_dicomweb import dicom_web_interface
from ez_wsi_dicomweb.ml_toolkit import dicom_path

from data_accessors import data_accessor_const
from data_accessors import data_accessor_errors
from data_accessors.utils import json_validation_utils
from data_accessors.utils import patch_coordinate as patch_coordinate_module

_InstanceJsonKeys = data_accessor_const.InstanceJsonKeys
_PRESENT = 'PRESENT'


@dataclasses.dataclass(frozen=True)
class DicomGenericImage:
  credential_factory: credential_factory_module.AbstractCredentialFactory
  instance_path: str
  base_request: Mapping[str, Any]
  patch_coordinates: List[patch_coordinate_module.PatchCoordinate]
  dicom_instances_metadata: List[dicom_web_interface.DicomObject]


def _generate_instance_metadata_error_string(
    metadata: Mapping[str, Any], *keys: str
) -> str:
  """returns instance metadata as a error string."""
  result = {}
  for key in keys:
    if key not in metadata:
      continue
    if key == _InstanceJsonKeys.BEARER_TOKEN:
      value = metadata[key]
      # If bearer token is present, and defined strip
      if isinstance(value, str) and value:
        result[key] = _PRESENT
        continue
    # otherwise just associate key and value.
    result[key] = metadata[key]
  return json.dumps(result, sort_keys=True)


def json_to_generic_dicom_image(
    credential_factory: credential_factory_module.AbstractCredentialFactory,
    instance: Mapping[str, Any],
    default_patch_width: int,
    default_patch_height: int,
    require_patch_dim_match_default_dim: bool,
    dicom_instances_metadata: List[dicom_web_interface.DicomObject],
) -> DicomGenericImage:
  """Converts json to DicomGenericImage."""
  try:
    patch_coordinates = patch_coordinate_module.parse_patch_coordinates(
        instance.get(_InstanceJsonKeys.PATCH_COORDINATES, []),
        default_patch_width,
        default_patch_height,
        require_patch_dim_match_default_dim,
    )
  except patch_coordinate_module.InvalidCoordinateError as exp:
    instance_error_msg = _generate_instance_metadata_error_string(
        instance,
        _InstanceJsonKeys.PATCH_COORDINATES,
    )
    raise data_accessor_errors.InvalidRequestFieldError(
        f'Invalid patch coordinate; {exp}; {instance_error_msg}'
    ) from exp

  if _InstanceJsonKeys.DICOM_SOURCE in instance:
    instance_path = instance.get(_InstanceJsonKeys.DICOM_SOURCE)
    if isinstance(instance_path, list):
      if not instance_path:
        raise data_accessor_errors.InvalidRequestFieldError(
            'dicom_source is an empty list.'
        )
      if len(instance_path) > 1:
        raise data_accessor_errors.InvalidRequestFieldError(
            'Endpoint does not support definitions with multiple DICOMweb URIs'
            ' in a dicom_source.'
        )
      instance_path = instance_path[0]
  elif _InstanceJsonKeys.DICOM_WEB_URI in instance:
    # Legacy support for decoding DICOM_WEB_URI used in MedSigLip Endpoint.
    instance_path = instance.get(_InstanceJsonKeys.DICOM_WEB_URI)
  else:
    raise data_accessor_errors.InvalidRequestFieldError(
        'DICOM path not defined.'
    )
  try:
    instance_path = json_validation_utils.validate_not_empty_str(instance_path)
  except json_validation_utils.ValidationError as exp:
    raise data_accessor_errors.InvalidRequestFieldError(
        'Invalid DICOM path.'
    ) from exp
  try:
    if dicom_path.FromString(instance_path).type != dicom_path.Type.INSTANCE:
      raise data_accessor_errors.InvalidRequestFieldError(
          f'Unsupported DICOM source "{instance_path}". Required to define a'
          ' DICOM SOP Instance.'
      )
  except ValueError as exp:
    raise data_accessor_errors.InvalidRequestFieldError(
        f'Invalid DICOMweb uri "{instance_path}".'
    ) from exp
  if len(dicom_instances_metadata) > 1:
    raise data_accessor_errors.InvalidRequestFieldError(
        'Multiple instances defined for DICOM path.'
    )
  try:
    return DicomGenericImage(
        credential_factory=credential_factory,
        instance_path=instance_path,
        base_request=instance,
        patch_coordinates=patch_coordinates,
        dicom_instances_metadata=dicom_instances_metadata,
    )
  except json_validation_utils.ValidationError as exp:
    error_msg = _generate_instance_metadata_error_string(
        instance,
        _InstanceJsonKeys.DICOM_WEB_URI,
        _InstanceJsonKeys.DICOM_SOURCE,
        _InstanceJsonKeys.BEARER_TOKEN,
        _InstanceJsonKeys.EXTENSIONS,
    )
    raise data_accessor_errors.InvalidRequestFieldError(
        f'DICOM instance JSON formatting is invalid; {error_msg}'
    ) from exp
