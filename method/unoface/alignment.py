import importlib.util
import io
import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from torchvision.transforms import functional as TF


def reference_exclusions(manifest_path=None):

    if not manifest_path:
        return set()
    manifest = json.loads(Path(manifest_path).read_text())
    paths = set()
    for image in manifest["images"]:
        path = Path(image["path"]).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Missing reference input: {path}")
        paths.add(str(path))
    return paths


class ReferenceAlignment:
    def __init__(self, root, detector):
        self.root = str(Path(root).resolve())
        self.detector_path = str(Path(detector).resolve())
        if not Path(self.detector_path).is_file():
            raise FileNotFoundError(f"Missing reference detector: {self.detector_path}")
        self.fallback_paths = set()
        self.pid = None

    def _initialize(self):
        dependencies = Path(self.root) / "deps"
        if dependencies.is_dir():
            sys.path.insert(0, str(dependencies))
        import cv2
        import onnxruntime as ort

        cv2.setNumThreads(1)
        modules = []
        for name in ("scrfd", "face_align"):
            path = Path(self.root) / "upstream" / f"{name}.py"
            if not path.is_file():
                raise FileNotFoundError(
                    f"Missing alignment source {path}; run python scripts/prepare_alignment.py"
                )
            spec = importlib.util.spec_from_file_location("_reference_" + name, path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            modules.append(module)
        options = ort.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        options.log_severity_level = 3
        session = ort.InferenceSession(
            self.detector_path, sess_options=options, providers=["CPUExecutionProvider"]
        )
        self.detector = modules[0].SCRFD(model_file=self.detector_path, session=session)
        self.detector.prepare(ctx_id=-1, input_size=(320, 320), det_thresh=0.5)
        self.crop = modules[1].norm_crop
        self.cv2 = cv2
        self.pid = os.getpid()

    def __getstate__(self):
        return {
            k: v
            for k, v in self.__dict__.items()
            if k not in ("detector", "crop", "cv2", "pid")
        } | {"pid": None}

    def __call__(self, image):
        if (
            not isinstance(image, bytes)
            and str(Path(image).resolve()) in self.fallback_paths
        ):
            with Image.open(image) as decoded:
                resized = TF.resize(decoded.convert("RGB"), [112, 112])
                return TF.normalize(TF.to_tensor(resized), [0.5] * 3, [0.5] * 3)
        if self.pid != os.getpid():
            self._initialize()
        cv2 = self.cv2

        encoded_input = image if isinstance(image, bytes) else Path(image).read_bytes()
        image = cv2.imdecode(
            np.frombuffer(encoded_input, dtype=np.uint8), cv2.IMREAD_COLOR
        )
        if image is None:
            raise ValueError("Cannot decode reference input")
        boxes, landmarks = self.detector.detect(
            image, input_size=(320, 320), max_num=1, metric="default"
        )
        if len(boxes) == 0:
            boxes, landmarks = self.detector.detect(
                image, input_size=(640, 640), max_num=1, metric="default"
            )
        if len(boxes) == 0:
            pad = int(max(image.shape[:2]) * 0.3)
            padded = cv2.copyMakeBorder(
                image, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0
            )
            boxes, landmarks = self.detector.detect(
                padded, input_size=(160, 160), max_num=1, metric="default"
            )
            if landmarks is not None:
                landmarks -= pad
        if len(boxes) == 0 or landmarks is None:
            raise ValueError("No face detected for the aligned reference input")
        cropped = self.crop(image, landmarks[0], image_size=112)

        ok, encoded = cv2.imencode(".jpg", cropped, [cv2.IMWRITE_JPEG_QUALITY, 100])
        if not ok:
            raise RuntimeError("Aligned image encoding failed")
        decoded = Image.open(io.BytesIO(encoded.tobytes())).convert("RGB")
        return TF.normalize(TF.to_tensor(decoded), [0.5] * 3, [0.5] * 3)


def reference_alignment(checkpoint, root, detector=None):
    protocol = checkpoint.get("preprocessing", "")
    if not protocol:
        return None
    if not protocol.startswith("five-landmark aligned 112x112 RGB"):
        raise ValueError(f"Unsupported reference preprocessing: {protocol}")
    if not detector:
        raise ValueError(
            "Specify --reference-detector with the SCRFD det_500m.onnx file"
        )
    return ReferenceAlignment(root, detector)
