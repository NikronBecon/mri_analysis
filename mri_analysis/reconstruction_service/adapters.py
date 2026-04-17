from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Protocol

from mri_analysis.shared.schemas import ReconstructionRequest


class ReconstructionAdapter(Protocol):
    def run(self, input_path: Path, workdir: Path, request: ReconstructionRequest) -> Path:
        ...


@dataclass
class StubReconstructionAdapter:
    name: str = "stub"

    def run(self, input_path: Path, workdir: Path, request: ReconstructionRequest) -> Path:
        output_path = workdir / "reconstructed.dicom"
        payload = input_path.read_bytes()
        content = (
            b"STUB-DICOM\n"
            + f"correlation_id={request.correlation_id}\n".encode("utf-8")
            + f"source_format={request.input_format}\n".encode("utf-8")
            + payload
        )
        output_path.write_bytes(content)
        return output_path


@dataclass
class MonaiUnetReconstructionAdapter:
    checkpoint_path: Path
    device_name: str = "cpu"
    name: str = "monai_unet"

    def __post_init__(self) -> None:
        import torch
        from monai.networks.nets import BasicUNet

        self.device = self._resolve_device(torch, self.device_name)
        self.model = BasicUNet(
            spatial_dims=2,
            in_channels=1,
            out_channels=1,
            features=[32, 64, 128, 256, 512, 32],
        ).to(self.device)

        try:
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=True)
        except TypeError:
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device)

        state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def run(self, input_path: Path, workdir: Path, request: ReconstructionRequest) -> Path:
        if request.input_format == "h5":
            volume, source_dataset = self._reconstruct_h5_volume(input_path)
        elif request.input_format == "dicom":
            volume, source_dataset = self._load_dicom_volume(input_path)
        else:
            raise ValueError(f"Unsupported reconstruction input format: {request.input_format}")

        output_path = workdir / "reconstructed.dcm"
        self._write_dicom(output_path, volume, request, source_dataset)
        return output_path

    @staticmethod
    def _resolve_device(torch_module, requested: str):
        normalized = requested.strip().lower()
        if normalized in {"gpu", "cuda"}:
            normalized = "cuda:0"
        if normalized.startswith("cuda") and torch_module.cuda.is_available():
            return torch_module.device(normalized)
        if normalized == "mps" and hasattr(torch_module.backends, "mps") and torch_module.backends.mps.is_available():
            return torch_module.device("mps")
        return torch_module.device("cpu")

    def _reconstruct_h5_volume(self, input_path: Path):
        import numpy as np
        import torch
        from monai.apps.reconstruction.complex_utils import complex_abs, convert_to_tensor_complex
        from monai.apps.reconstruction.fastmri_reader import FastMRIReader
        from monai.apps.reconstruction.mri_utils import root_sum_of_squares
        from monai.data.fft_utils import ifftn_centered

        reader = FastMRIReader()
        raw_data = reader.read(str(input_path))
        kspace_np, _ = reader.get_data(raw_data)

        if np.iscomplexobj(kspace_np):
            kspace_tensor = convert_to_tensor_complex(kspace_np, dtype=torch.float32)
        elif kspace_np.shape[-1] == 2:
            kspace_tensor = torch.as_tensor(kspace_np, dtype=torch.float32)
        else:
            raise ValueError(
                f"Unsupported k-space representation: shape={kspace_np.shape}, dtype={kspace_np.dtype}. "
                "Expected complex-valued data or a final size-2 real/imag dimension."
            )

        zero_filled_complex = ifftn_centered(kspace_tensor, spatial_dims=2, is_complex=True)
        zero_filled_abs = complex_abs(zero_filled_complex)
        zero_filled = root_sum_of_squares(zero_filled_abs, spatial_dim=1).to(torch.float32)

        slice_means = zero_filled.mean(dim=(1, 2), keepdim=True)
        slice_stds = zero_filled.std(dim=(1, 2), keepdim=True).clamp_min(1e-8)
        zero_filled_norm = ((zero_filled - slice_means) / slice_stds).clamp(-6.0, 6.0)

        reconstructed_slices = []
        with torch.no_grad():
            for slice_idx in range(zero_filled_norm.shape[0]):
                model_input = zero_filled_norm[slice_idx].unsqueeze(0).unsqueeze(0).to(self.device)
                model_output = self.model(model_input).detach().cpu().squeeze(0).squeeze(0)
                model_output = model_output * slice_stds[slice_idx].item() + slice_means[slice_idx].item()
                reconstructed_slices.append(model_output.numpy())

        volume = np.stack(reconstructed_slices, axis=0)
        return volume.astype(np.float32), None

    @staticmethod
    def _load_dicom_volume(input_path: Path):
        import numpy as np
        from pydicom import dcmread

        dataset = dcmread(str(input_path))
        if "PixelData" not in dataset:
            raise ValueError("Input DICOM does not contain pixel data")

        pixel_array = dataset.pixel_array
        if pixel_array.ndim == 2:
            volume = pixel_array[np.newaxis, :, :]
        elif pixel_array.ndim == 3:
            volume = pixel_array
        else:
            raise ValueError(f"Unsupported DICOM pixel array shape: {pixel_array.shape}")

        return volume.astype(np.float32), dataset

    @staticmethod
    def _to_uint16(volume):
        import numpy as np

        finite_volume = np.nan_to_num(volume.astype(np.float32), copy=False)
        minimum = float(finite_volume.min())
        maximum = float(finite_volume.max())
        if maximum <= minimum:
            return np.zeros_like(finite_volume, dtype=np.uint16)
        scaled = (finite_volume - minimum) / (maximum - minimum)
        return np.clip(np.round(scaled * 65535.0), 0, 65535).astype(np.uint16)

    def _write_dicom(self, output_path: Path, volume, request: ReconstructionRequest, source_dataset) -> None:
        from pydicom.dataset import FileDataset, FileMetaDataset
        from pydicom.uid import ExplicitVRLittleEndian, MRImageStorage, PYDICOM_IMPLEMENTATION_UID, generate_uid

        pixel_data = self._to_uint16(volume)
        rows = int(pixel_data.shape[-2])
        columns = int(pixel_data.shape[-1])
        number_of_frames = int(pixel_data.shape[0])

        file_meta = FileMetaDataset()
        file_meta.MediaStorageSOPClassUID = MRImageStorage
        file_meta.MediaStorageSOPInstanceUID = generate_uid()
        file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
        file_meta.ImplementationClassUID = PYDICOM_IMPLEMENTATION_UID

        dataset = FileDataset(str(output_path), {}, file_meta=file_meta, preamble=b"\0" * 128)
        dataset.is_little_endian = True
        dataset.is_implicit_VR = False
        dataset.SOPClassUID = file_meta.MediaStorageSOPClassUID
        dataset.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
        dataset.Modality = "MR"
        dataset.ImageType = ["DERIVED", "SECONDARY", "RECONSTRUCTION"]
        dataset.Rows = rows
        dataset.Columns = columns
        dataset.NumberOfFrames = str(number_of_frames)
        dataset.SamplesPerPixel = 1
        dataset.PhotometricInterpretation = "MONOCHROME2"
        dataset.BitsAllocated = 16
        dataset.BitsStored = 16
        dataset.HighBit = 15
        dataset.PixelRepresentation = 0
        dataset.PixelSpacing = [1.0, 1.0]
        dataset.SliceThickness = 1.0
        dataset.InstanceNumber = 1

        now = datetime.utcnow()
        dataset.ContentDate = now.strftime("%Y%m%d")
        dataset.ContentTime = now.strftime("%H%M%S.%f")
        dataset.StudyDate = now.strftime("%Y%m%d")
        dataset.StudyTime = now.strftime("%H%M%S")

        if source_dataset is not None:
            for field in ("PatientName", "PatientID", "StudyInstanceUID", "SeriesInstanceUID", "AccessionNumber"):
                if hasattr(source_dataset, field):
                    setattr(dataset, field, getattr(source_dataset, field))
        if not getattr(dataset, "PatientName", None):
            dataset.PatientName = "Anonymous"
        if not getattr(dataset, "PatientID", None):
            dataset.PatientID = request.correlation_id
        if not getattr(dataset, "StudyInstanceUID", None):
            dataset.StudyInstanceUID = generate_uid()
        dataset.SeriesInstanceUID = generate_uid()

        dataset.SeriesDescription = "AI MRI Reconstruction"
        dataset.ProtocolName = "AI MRI Reconstruction"
        dataset.ImageComments = f"Generated by {self.name}"
        dataset.PixelData = pixel_data.tobytes()
        dataset.save_as(str(output_path), write_like_original=False)


def build_reconstruction_adapter_from_env() -> ReconstructionAdapter:
    adapter_name = os.getenv("RECONSTRUCTION_ADAPTER", "stub").strip().lower()
    if adapter_name == "stub":
        return StubReconstructionAdapter()
    if adapter_name == "monai_unet":
        checkpoint_path = Path(os.getenv("RECONSTRUCTION_MODEL_PATH", "/app/demo_checkpoint/unet_mri_reconstruction.pt"))
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Reconstruction checkpoint not found: {checkpoint_path}")
        return MonaiUnetReconstructionAdapter(
            checkpoint_path=checkpoint_path,
            device_name=os.getenv("INFERENCE_DEVICE", "cpu"),
        )
    raise ValueError(f"Unsupported RECONSTRUCTION_ADAPTER value: {adapter_name}")
