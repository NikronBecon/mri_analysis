from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Tuple

import numpy as np

from mri_analysis.shared.schemas import DetectionRequest, Finding, FindingLocation, FindingsDocument


class DetectionAdapter(Protocol):
    def run(self, input_path: Path, workdir: Path, request: DetectionRequest) -> Tuple[Path, Path]:
        ...


@dataclass
class StubDetectionAdapter:
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

    def __init__(self, bundle_dir: str | Path, device: str = "cuda:0"):
        import torch
        from monai.inferers import SlidingWindowInferer
        from monai.networks.nets import SegResNet

        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.bundle_dir = Path(bundle_dir)

        self.model = SegResNet(
            blocks_down=[1, 2, 2, 4],
            blocks_up=[1, 1, 1],
            init_filters=16,
            in_channels=4,
            out_channels=3,
            dropout_prob=0.2,
        )
        ckpt = torch.load(
            self.bundle_dir / "models" / "model.pt",
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
        import torch
        from monai.transforms import Compose, EnsureChannelFirst, LoadImage, NormalizeIntensity

        # Load & preprocess
        preprocess = Compose([
            LoadImage(image_only=True),
            EnsureChannelFirst(),
            NormalizeIntensity(nonzero=True, channel_wise=True),
        ])
        image = preprocess(str(input_path))
        image = image.unsqueeze(0).to(self.device)

        # Inference
        with torch.no_grad(), torch.amp.autocast("cuda"):
            logits = self.inferer(image, self.model)

        pred_probs = torch.sigmoid(logits)[0].cpu()
        pred_binary = (pred_probs > 0.5).float()

        # Build label map: WT=2, TC=1, ET=4 (BraTS convention)
        pred_np = pred_binary.numpy()
        label_map = np.zeros(pred_np.shape[1:], dtype=np.uint8)
        label_map[pred_np[1] > 0] = 2
        label_map[pred_np[0] > 0] = 1
        label_map[pred_np[2] > 0] = 4

        # Save annotated NIfTI
        ref_img = nib.load(str(input_path))
        seg_nii = nib.Nifti1Image(label_map, affine=ref_img.affine)
        annotated_path = workdir / "annotated_segmentation.nii.gz"
        nib.save(seg_nii, str(annotated_path))

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
            artifacts={"segmentation_nifti": annotated_path.name},
        )
        findings_path = workdir / "findings.json"
        findings_path.write_text(json.dumps(doc.model_dump(), indent=2), encoding="utf-8")

        return annotated_path, findings_path

