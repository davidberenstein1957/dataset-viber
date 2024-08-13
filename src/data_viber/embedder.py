# Copyright 2024-present, David Berenstein, Inc.
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

from pathlib import Path

import torch
import torch.nn.functional as F
from optimum.modeling_base import OptimizedModel
from optimum.onnxruntime import ORTModelForFeatureExtraction, ORTOptimizer, ORTQuantizer
from optimum.onnxruntime.configuration import AutoQuantizationConfig, OptimizationConfig
from transformers import AutoTokenizer, Pipeline

from data_viber._constants import DEFAULT_EMBEDDING_MODEL


class Embedder:
    def __init__(
        self,
        model_id=DEFAULT_EMBEDDING_MODEL,
        use_onnx=True,
        device="cpu",
    ):
        self.device = device
        self.model_id = model_id
        # set onnx path to cache dir in main home directory
        self.onnx_path = Path("~/.cache/onnx").expanduser()
        self.use_onnx = use_onnx
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        self.load_model()
        self.save_model()
        self.create_pipeline()
        print(f"Model loaded on {self.device}")
        if self.use_onnx:
            print("Optimizing and quantizing model...")
            self.optimize_model()
            print("Optimization complete.")
            self.load_optimized_model()
            self.quantize_model()
            self.print_model_sizes()

    def load_model(self):
        if self.use_onnx:
            self.model: OptimizedModel = ORTModelForFeatureExtraction.from_pretrained(
                self.model_id, export=True
            )
        else:
            self.model = ORTModelForFeatureExtraction.from_pretrained(
                self.model_id, device=self.device
            )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)

    def save_model(self):
        self.model.save_pretrained(self.onnx_path)
        self.tokenizer.save_pretrained(self.onnx_path)

    @staticmethod
    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )

    class SentenceEmbeddingPipeline(Pipeline):
        def _sanitize_parameters(self, **kwargs):
            return {}, {}, {}

        def preprocess(self, inputs):
            return self.tokenizer(
                inputs, padding=True, truncation=True, return_tensors="pt"
            )

        def _forward(self, model_inputs):
            outputs = self.model(**model_inputs)
            return {
                "outputs": outputs,
                "attention_mask": model_inputs["attention_mask"],
            }

        def postprocess(self, model_outputs):
            sentence_embeddings = Embedder.mean_pooling(
                model_outputs["outputs"], model_outputs["attention_mask"]
            )
            return F.normalize(sentence_embeddings, p=2, dim=1)

    def create_pipeline(self):
        self.pipeline = self.SentenceEmbeddingPipeline(
            model=self.model, tokenizer=self.tokenizer, device=self.device
        )

    def encode(self, text, convert_to_numpy=False):
        prediction = self.pipeline(text)
        if convert_to_numpy:
            return [pred.cpu().detach().numpy()[0] for pred in prediction]
        else:
            return prediction

    def optimize_model(self):
        if not self.use_onnx:
            raise ValueError(
                "Model must be in ONNX format to optimize. Set use_onnx=True when initializing."
            )
        optimizer = ORTOptimizer.from_pretrained(self.model)
        optimization_config = OptimizationConfig(
            optimization_level=99,
            fp16=True,
            optimize_for_gpu=False if self.device == "cpu" else True,
        )
        optimizer.optimize(
            save_dir=self.onnx_path,
            file_suffix=f"optimized_{self.device}",
            optimization_config=optimization_config,
        )

    def load_optimized_model(self):
        if not self.use_onnx:
            raise ValueError(
                "Model must be in ONNX format to load optimized version. Set use_onnx=True when initializing."
            )
        self.model = ORTModelForFeatureExtraction.from_pretrained(
            self.onnx_path,
            file_name=f"model_optimized_{self.device}.onnx",
            device=self.device,
        )
        self.create_pipeline()

    def quantize_model(self):
        if not self.use_onnx:
            raise ValueError(
                "Model must be in ONNX format to quantize. Set use_onnx=True when initializing."
            )
        dynamic_quantizer = ORTQuantizer.from_pretrained(self.model)
        dqconfig = AutoQuantizationConfig.avx512_vnni(
            is_static=False, per_channel=False
        )
        dynamic_quantizer.quantize(
            save_dir=self.onnx_path,
            quantization_config=dqconfig,
            file_suffix=f"quantized_{self.device}",
        )

    def load_quantized_model(self):
        if not self.use_onnx:
            raise ValueError(
                "Model must be in ONNX format to load quantized version. Set use_onnx=True when initializing."
            )
        self.model = ORTModelForFeatureExtraction.from_pretrained(
            self.onnx_path,
            file_name=f"model_optimized_quantized_{self.device}.onnx",
            device=self.device,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.onnx_path)
        self.create_pipeline()

    def print_model_sizes(self):
        if not self.use_onnx:
            raise ValueError(
                "Model must be in ONNX format to print sizes. Set use_onnx=True when initializing."
            )
        size = (self.onnx_path / "model.onnx").stat().st_size / (1024 * 1024)
        optimized_size = (self.onnx_path / "model_optimized.onnx").stat().st_size / (
            1024 * 1024
        )
        quantized_size = (
            self.onnx_path / "model_optimized_quantized.onnx"
        ).stat().st_size / (1024 * 1024)
        print(f"Model file size: {size:.2f} MB")
        print(f"Optimized Model file size: {optimized_size:.2f} MB")
        print(f"Quantized Model file size: {quantized_size:.2f} MB")
