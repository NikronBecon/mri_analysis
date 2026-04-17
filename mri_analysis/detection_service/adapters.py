from __future__ import annotations

from contextlib import nullcontext
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Tuple

from mri_analysis.shared.schemas import DetectionRequest, Finding, FindingLocation, FindingsDocument


class DetectionAdapter(Protocol):
    def run(self, input_path: Path, workdir: Path, request: DetectionRequest) -> Tuple[Path, Path]:
        ...


@dataclass
class StubDetectionAdapter:
    name: str = "stub"

    def run(self, input_path: Path, workdir: Path, request: DetectionRequest) -> Tuple[Path, Path]:
        annotated_path = workdir / "annotated.dicom"
        findings_path = workdir / "findings.json"

        payload = input_path.read_bytes()
        annotated_path.write_bytes(
            b"ANNOTATED-STUB-DICOM\n"
            + f"correlation_id={request.correlation_id}\n".encode("utf-8")
            + payload
        )

        findings = FindingsDocument(
            job_id=request.job_id,
            study_id=request.job_id or request.correlation_id,
            source_file=input_path.name,
            findings=[
                Finding(
                    pathology_type="stub_lesion",
                    confidence=0.93,
                    location=FindingLocation(description="left temporal lobe"),
                    explanation="Stub output for pipeline integration until a real pathology model is plugged in.",
                )
            ],
            artifacts={},
        )
        findings_path.write_text(json.dumps(findings.model_dump(), indent=2), encoding="utf-8")
        return annotated_path, findings_path


class SegResNetDetectionAdapter:
    """Production adapter using SegResNet (MONAI) for brain tumor segmentation.

    Expects a 4-channel NIfTI input (T1c, T1, T2, FLAIR).
    Outputs an annotated NIfTI segmentation mask and a findings JSON.
    """

    REGION_MAP = {
        0: ("tumor_core", "Necrotic/non-enhancing tumor core and enhancing regions"),
        1: ("whole_tumor", "Complete tumor extent including edema"),
        2: ("enhancing_tumor", "Gadolinium-enhancing active tumor regions"),
    }
    name = "segresnet"

    def __init__(self, bundle_dir: str | Path, device: str = "cuda:0"):
        import torch
        from monai.inferers import SlidingWindowInferer
        from monai.networks.nets import SegResNet

        normalized_device = device.strip().lower()
        if normalized_device in {"gpu", "cuda"}:
            normalized_device = "cuda:0"
        self.device = torch.device(normalized_device if torch.cuda.is_available() and normalized_device.startswith("cuda") else "cpu")
        self.bundle_dir = Path(bundle_dir)
        self.model_path = self.bundle_dir / "models" / "model.pt"

        self.model = SegResNet(
            blocks_down=[1, 2, 2, 4],
            blocks_up=[1, 1, 1],
            init_filters=16,
            in_channels=4,
            out_channels=3,
            dropout_prob=0.2,
        )
        ckpt = torch.load(
            self.model_path,
            map_location=self.device,
            weights_only=True,
        )
        self.model.load_state_dict(ckpt)
        self.model.to(self.device).eval()

        self.inferer = SlidingWindowInferer(
            roi_size=[240, 240, 160], sw_batch_size=1, overlap=0.5,
        )

    def run(self, input_path: Path, workdir: Path, request: DetectionRequest) -> Tuple[Path, Path]:
        import nibabel as nib
        import numpy as np
        from pydicom import dcmread
        import torch

        if input_path.suffixes[-2:] == [".nii", ".gz"] or input_path.suffix == ".nii":
            image, reference = self._load_nifti_volume(input_path)
            annotated_path_builder = lambda label_map: self._write_nifti_annotation(input_path, workdir, label_map, nib)
            source_kind = "nifti"
        else:
            dicom_dataset = dcmread(str(input_path))
            image, reference = self._load_dicom_volume(dicom_dataset)
            annotated_path_builder = lambda label_map: self._write_annotated_dicom(workdir, dicom_dataset, reference["base_volume"], label_map)
            source_kind = "dicom"

        image = torch.from_numpy(image).unsqueeze(0).to(self.device)

        # Inference
        autocast_context = torch.amp.autocast("cuda") if self.device.type == "cuda" else nullcontext()
        with torch.no_grad(), autocast_context:
            logits = self.inferer(image, self.model)

        pred_probs = torch.sigmoid(logits)[0].cpu()
        pred_binary = (pred_probs > 0.5).float()

        # Build label map: WT=2, TC=1, ET=4 (BraTS convention)
        pred_np = pred_binary.numpy()
        label_map = np.zeros(pred_np.shape[1:], dtype=np.uint8)
        label_map[pred_np[1] > 0] = 2
        label_map[pred_np[0] > 0] = 1
        label_map[pred_np[2] > 0] = 4

        annotated_path = annotated_path_builder(label_map)

        # Build findings
        findings_list = []
        for ch_idx, (name, explanation) in self.REGION_MAP.items():
            mask = pred_binary[ch_idx]
            n_voxels = int(mask.sum())
            if n_voxels == 0:
                continue
            coords = torch.nonzero(mask)
            centroid = coords.float().mean(dim=0).tolist()
            confidence = float(pred_probs[ch_idx][mask > 0].mean())

            findings_list.append(Finding(
                pathology_type=name,
                confidence=round(confidence, 4),
                location=FindingLocation(
                    description=(
                        f"centroid=({centroid[0]:.0f},{centroid[1]:.0f},{centroid[2]:.0f}), "
                        f"volume={n_voxels / 1000:.2f}mL"
                    ),
                ),
                explanation=explanation,
            ))

        doc = FindingsDocument(
            job_id=request.job_id,
            study_id=request.job_id or request.correlation_id,
            source_file=input_path.name,
            findings=findings_list,
            artifacts={
                "annotated_output": annotated_path.name,
                "input_kind": source_kind,
                "channel_strategy": "native_4ch" if source_kind == "nifti" else "repeated_single_volume_to_4ch",
                "model_path": str(self.model_path),
            },
        )
        findings_path = workdir / "findings.json"
        findings_path.write_text(json.dumps(doc.model_dump(), indent=2), encoding="utf-8")

        return annotated_path, findings_path

    @staticmethod
    def _normalize_channels(image):
        import numpy as np

        normalized = image.astype(np.float32, copy=True)
        for channel_idx in range(normalized.shape[0]):
            channel = normalized[channel_idx]
            mask = channel != 0
            sample = channel[mask] if mask.any() else channel.reshape(-1)
            mean = float(sample.mean())
            std = float(sample.std())
            if std < 1e-8:
                std = 1.0
            normalized[channel_idx] = (channel - mean) / std
        return normalized

    def _load_nifti_volume(self, input_path: Path):
        import nibabel as nib
        import numpy as np

        nifti = nib.load(str(input_path))
        image = np.asarray(nifti.get_fdata(), dtype=np.float32)
        if image.ndim != 4:
            raise ValueError(
                "SegResNetDetectionAdapter expects a 4-channel NIfTI volume with shape (H, W, D, 4) "
                f"or (4, H, W, D). Received shape: {image.shape}"
            )
        if image.shape[0] == 4:
            channel_first = image
        elif image.shape[-1] == 4:
            channel_first = np.moveaxis(image, -1, 0)
        else:
            raise ValueError(
                "SegResNetDetectionAdapter expects exactly 4 channels in the NIfTI input. "
                f"Received shape: {image.shape}"
            )
        return self._normalize_channels(channel_first), {"affine": nifti.affine, "header": nifti.header}

    def _load_dicom_volume(self, dicom_dataset):
        import numpy as np

        if "PixelData" not in dicom_dataset:
            raise ValueError("Input DICOM does not contain pixel data")
        pixel_array = np.asarray(dicom_dataset.pixel_array, dtype=np.float32)
        if pixel_array.ndim == 2:
            pixel_array = pixel_array[np.newaxis, :, :]
        elif pixel_array.ndim != 3:
            raise ValueError(f"Unsupported DICOM pixel array shape: {pixel_array.shape}")

        volume_hwd = np.transpose(pixel_array, (1, 2, 0))
        image = np.repeat(volume_hwd[np.newaxis, ...], 4, axis=0)
        return self._normalize_channels(image), {"base_volume": pixel_array}

    @staticmethod
    def _to_uint16(volume):
        import numpy as np

        volume = np.nan_to_num(volume.astype(np.float32), copy=False)
        vmin = float(volume.min())
        vmax = float(volume.max())
        if vmax <= vmin:
            return np.zeros_like(volume, dtype=np.uint16)
        scaled = (volume - vmin) / (vmax - vmin)
        return np.clip(np.round(scaled * 65535.0), 0, 65535).astype(np.uint16)

    def _write_nifti_annotation(self, input_path: Path, workdir: Path, label_map, nib_module):
        reference = nib_module.load(str(input_path))
        annotated_path = workdir / "annotated_segmentation.nii.gz"
        seg_nii = nib_module.Nifti1Image(label_map, affine=reference.affine, header=reference.header)
        nib_module.save(seg_nii, str(annotated_path))
        return annotated_path

    def _write_annotated_dicom(self, workdir: Path, source_dataset, base_volume, label_map):
        import numpy as np
        from pydicom.dataset import FileDataset, FileMetaDataset
        from pydicom.uid import ExplicitVRLittleEndian, MRImageStorage, PYDICOM_IMPLEMENTATION_UID, generate_uid

        base_uint16 = self._to_uint16(base_volume)
        label_dhw = np.transpose(label_map, (2, 0, 1))

        overlay = base_uint16.copy()
        overlay[label_dhw == 1] = 50000
        overlay[label_dhw == 2] = 58000
        overlay[label_dhw == 4] = 65535

        file_meta = FileMetaDataset()
        file_meta.MediaStorageSOPClassUID = MRImageStorage
        file_meta.MediaStorageSOPInstanceUID = generate_uid()
        file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
        file_meta.ImplementationClassUID = PYDICOM_IMPLEMENTATION_UID

        annotated_path = workdir / "annotated.dcm"
        dataset = FileDataset(str(annotated_path), {}, file_meta=file_meta, preamble=b"\0" * 128)
        dataset.is_little_endian = True
        dataset.is_implicit_VR = False
        dataset.SOPClassUID = file_meta.MediaStorageSOPClassUID
        dataset.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
        dataset.Modality = "MR"
        dataset.ImageType = ["DERIVED", "SECONDARY", "AI_ANNOTATION"]
        dataset.Rows = int(overlay.shape[-2])
        dataset.Columns = int(overlay.shape[-1])
        dataset.NumberOfFrames = str(int(overlay.shape[0]))
        dataset.SamplesPerPixel = 1
        dataset.PhotometricInterpretation = "MONOCHROME2"
        dataset.BitsAllocated = 16
        dataset.BitsStored = 16
        dataset.HighBit = 15
        dataset.PixelRepresentation = 0
        dataset.PixelSpacing = getattr(source_dataset, "PixelSpacing", [1.0, 1.0])
        dataset.SliceThickness = getattr(source_dataset, "SliceThickness", 1.0)
        dataset.PatientName = getattr(source_dataset, "PatientName", "Anonymous")
        dataset.PatientID = getattr(source_dataset, "PatientID", "unknown")
        dataset.StudyInstanceUID = getattr(source_dataset, "StudyInstanceUID", generate_uid())
        dataset.SeriesInstanceUID = generate_uid()
        dataset.SeriesDescription = "AI Brain Tumor Annotation"
        dataset.ProtocolName = "AI Brain Tumor Annotation"
        dataset.ImageComments = (
            "Generated by SegResNetDetectionAdapter; labels burned into grayscale intensity. "
            "Single reconstructed volume duplicated to 4 channels for compatibility."
        )
        dataset.PixelData = overlay.tobytes()
        dataset.save_as(str(annotated_path), write_like_original=False)
        return annotated_path


def build_detection_adapter_from_env() -> DetectionAdapter:
    adapter_name = os.getenv("DETECTION_ADAPTER", "stub").strip().lower()
    if adapter_name == "stub":
        return StubDetectionAdapter()
    if adapter_name == "segresnet":
        bundle_dir = Path(os.getenv("DETECTION_BUNDLE_DIR", "/app/model_bundles/brats_mri_segmentation"))
        model_path = bundle_dir / "models" / "model.pt"
        if not model_path.is_file():
            raise FileNotFoundError(
                f"Detection bundle checkpoint not found: {model_path}. "
                "Place the SegResNet bundle at DETECTION_BUNDLE_DIR/models/model.pt."
            )
        return SegResNetDetectionAdapter(
            bundle_dir=bundle_dir,
            device=os.getenv("INFERENCE_DEVICE", "cpu"),
        )
    raise ValueError(f"Unsupported DETECTION_ADAPTER value: {adapter_name}")
