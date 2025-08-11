# Copyright The FMS Model Optimizer Authors
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

"""
Base QModel Class
"""

# Standard
# pylint: disable=keyword-arg-before-vararg
import logging

# Third Party
import torch

logger = logging.getLogger(__name__)


class Qmodel:  # do not inherit nn.Module, or self.model will not show up in __dict__
    """
    A wrapper for fms_mo model, mainly for user API simplification purpose.
    Everything is the same as the original model, but we can add new member functions.
    Make sure the naming will be unique enough so that we won't override any existing functions
    in Huggingface models.
    """

    def __init__(self, original_model, qcfg=None, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.org_attr = dir(original_model)
        self.model = original_model
        if qcfg:
            self.qcfg = qcfg

    def __getattr__(self, name: str):
        if name in self.org_attr:
            logger.info(f"Trying to access self.{name}, forward to self.model.{name}")
            return getattr(self.model, name)
            # NOTE: this self.model is in __dict__, so it will not trigger __getattr__
            # recursively.!!

    def __call__(self, *args, **kwargs):
        logger.info(
            "Make this object callable, but actually just calling self.model.__call__()"
        )
        return self.model(*args, **kwargs)

    def __repr__(self):
        OKCYAN = "\033[96m"
        ENDC = "\033[0m"
        rep_txt = f"{OKCYAN}FMSMO_Qmodel_wrapper({ENDC}\n{self.model.__repr__()}{OKCYAN}){ENDC}"
        return rep_txt

    def to(self, tar_dev: torch.device):
        """
        Demonstrate that we can override a function in original model
        it should not call __getattr__(), i.e. will not see the printout from that func

        Args:
            tar_dev (torch.device): A new device

        Returns:
            Qmodel: Moved model to tar_dev
        """
        logger.info(
            f"Override a function in original model. moving the model to a new device {tar_dev}"
        )
        return self.model.to(tar_dev)

    def save_model_in_pt_fmt(
        self, filename: str = "model.pt", exam_inp: torch.Tensor = None
    ):
        """
        Save entire model to file

        Args:
            filename (str, optional): File path to save model. Defaults to "model.pt".
            exam_inp (torch.Tensor, optional): Example input for model. Defaults to None.
        """
        # NOTE self.qcfg has a lot of info already, like transformers_version
        # NOTE cannot save wrapped self, can only save self.model...
        save_dict = {"model": self.model}
        if exam_inp:
            save_dict["exam_inp"] = exam_inp
        torch.save(save_dict, filename)
        logger.info(f"{filename} saved successfully.")

    def save_statedict_in_pt_fmt(
        self,
        filename: str = "model.pt",
    ):
        """
        Save the model state dict to file

        Args:
            filename (str, optional): File path to save model state dict. Defaults to "model.pt".
        """
        torch.save(self.model.state_dict(), filename)
        logger.info(f"model.state_dict() is saved to {filename} successfully.")

    def run_gptq(self):
        """
        Check model is supported by AutoGPTQ first
        """
        return
