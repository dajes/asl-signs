import os

os.environ['CUDA_VISIBLE_DEVICES'] = ''

import onnxsim
import torch
from tensorflow.python.ops.image_ops_impl import ResizeMethod

import constants
from dataset.basic import BasicDataset
from modeling.transformer import Attention, Mlp
from modeling.llama import Attention as AttentionLlama, SwiGLUFFNFused
from utils import seed_everything

CLIP_CONTEXT = 10000
model_path = [
    # os.path.join(constants.MODELS_PATH, 'asl_0711/asl_0711.ckpt'),
    # os.path.join(constants.MODELS_PATH, 'asl_0714/asl_0714.ckpt'),
    # os.path.join(constants.MODELS_PATH, 'asl_0715/asl_0715.ckpt'),
    os.path.join(constants.MODELS_PATH, 'asl_0739/last.ckpt'),
]
onnx_path = os.path.splitext(model_path[-1])[0] + '.onnx'
sm_path = os.path.splitext(model_path[-1])[0] + '.pb'
tf_path = os.path.splitext(model_path[-1])[0] + '.tf'
tflite_path = os.path.splitext(model_path[-1])[0] + '.tflite'


class Inference(torch.nn.Module):
    def __init__(self, model, pad_or_trim=True):
        super().__init__()
        self.model = model
        linear = self.model.tail
        self.relevant_ids = BasicDataset.rids
        self.n_coords = linear.in_features // len(self.relevant_ids)
        assert linear.in_features == self.n_coords * len(self.relevant_ids)
        self.max_len = min(self.model.max_len, CLIP_CONTEXT)
        self.pad_or_trim = pad_or_trim
        self.center_around = BasicDataset.get_center_around()

    def compat(self):
        for module in self.model.modules():
            if isinstance(module, (Attention, AttentionLlama)):
                module.flash = False
            if isinstance(module, (Mlp, SwiGLUFFNFused)):
                module.export = True

    def forward(self, x):
        x = x.reshape(1, x.shape[0], x.shape[1] * x.shape[2])
        x[torch.isnan(x)] = 0
        return self.model(x, 1, )


class Ensemble(torch.nn.Module):
    def __init__(self, models, method='mean'):
        super().__init__()
        self.models = torch.nn.ModuleList(models)
        self.method = method
        self.relevant_ids = models[0].relevant_ids
        self.n_coords = models[0].n_coords
        self.max_len = models[0].max_len
        self.pad_or_trim = models[0].pad_or_trim
        self.center_around = models[0].center_around

    def compat(self):
        for m in self.models:
            m.compat()

    def forward(self, x):
        preds = torch.stack([m(x) for m in self.models])
        if self.method == 'mean':
            ensembled = preds.mean(0)
        elif self.method == 'probmean':
            preds = preds.softmax(-1)
            ensembled = preds.mean(0)
            ensembled = ensembled / ensembled.sum(-1, keepdims=True)
        elif self.method == 'ranking':
            ensembled = (preds.argsort(-1) + torch.arange(preds.shape[0])[:, None, None] / preds.shape[0]).mean(0)
            ensembled = ensembled / ensembled.sum(-1, keepdims=True)
        else:
            raise ValueError(f'Unknown method {repr(self.method)}')

        return ensembled


if __name__ == '__main__':
    import numpy as np
    import onnx
    import tensorflow as tf
    from onnx_tf.backend import prepare
    from module import Module

    seed_everything()
    inferences = []
    for model_path_ in model_path:
        module = Module.load_from_checkpoint(
            model_path_, 'cpu', strict=False
        )
        inference = Inference(module.model_ema.module).eval()
        inferences.append(inference)
    inference = Ensemble(inferences, 'mean').eval()
    inputs = torch.rand((50, 543, 3))
    _inputs = inputs.clone()
    _inputs = _inputs[:, inference.relevant_ids, :inference.n_coords]

    data_subset = _inputs[:, inference.center_around]
    m = ~torch.isnan(data_subset).any(dim=2)
    _inputs -= data_subset[m].mean(0, keepdims=True)[None]

    with torch.no_grad():
        truth = inference(_inputs)
        inference.compat()
        truth2 = inference(_inputs)
    error = np.abs(truth2.numpy() - truth.numpy()).max()
    print('Compatibility check:')
    print('Max error:', error)
    print('Cosine similarity:', np.dot(truth2.numpy(), truth[0].numpy()) /
          (np.linalg.norm(truth2.numpy()) * np.linalg.norm(truth.numpy())))

    print('[1 / 9] Exporting ONNX model...')
    torch.onnx.export(
        model=inference,
        args=_inputs,
        f=onnx_path,
        opset_version=12,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "input"}},
        verbose=False
    )

    print('[2 / 9] Simplifying ONNX model...')
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    model_onnx, check = onnxsim.simplify(onnx_model)
    assert check
    onnx.save(model_onnx, onnx_path)

    print('[3 / 9] Preparing ONNX model...')
    tf_rep = prepare(onnx_model)
    print('[4 / 9] Exporting ONNX model...')
    tf_rep.export_graph(sm_path)


    class ASLInferModel(tf.Module):
        def __init__(self, path):
            super().__init__()
            self.model = tf.saved_model.load(path)
            self.model.trainable = False
            self.max_len = inference.max_len
            self.pad_or_trim = inference.pad_or_trim
            self.relevant_ids = tf.constant(inference.relevant_ids)
            self.center_around = tf.constant(inference.center_around)
            self.n_relevant = len(inference.relevant_ids)
            self.n_coords = inference.n_coords

        @tf.function(
            input_signature=[
                tf.TensorSpec(shape=[None, 543, 3], dtype=tf.float32, name="inputs")
            ]
        )
        def call(self, input):
            input = tf.gather(input, self.relevant_ids, axis=1)
            input = input[:, :, :self.n_coords]
            if self.pad_or_trim:
                # resize sequence length of max_len
                if tf.shape(input)[0] > self.max_len:
                    input = \
                        tf.image.resize(input[None], (self.max_len, self.n_relevant), ResizeMethod.NEAREST_NEIGHBOR)[0]

            data_subset = tf.gather(input, self.center_around, axis=1)
            input = input - tf.reduce_mean(
                data_subset[~tf.math.reduce_any(tf.math.is_nan(data_subset), axis=2)], axis=0, keepdims=True)

            output_tensors = {
                "outputs": self.model(
                    **{"input": input}
                )["output"][0, :]
            }
            return output_tensors


    print('[5 / 9] Loading TF model...')
    mytfmodel = ASLInferModel(sm_path)
    print('[6 / 9] Exporting TF model...')
    tf.saved_model.save(
        mytfmodel, tf_path, signatures={"serving_default": mytfmodel.call}
    )
    print('[7 / 9] Checking TF model...')
    r = mytfmodel.call(inputs.numpy())['outputs'].numpy()

    error = np.abs(r - truth.numpy()).max()
    print('Tf model check:')
    print('Max error:', error)
    print('Cosine similarity:', np.dot(r, truth[0].numpy()) / (np.linalg.norm(r) * np.linalg.norm(truth.numpy())))

    print('[8 / 9] Loading TFLite model...')
    converter = tf.lite.TFLiteConverter.from_saved_model(tf_path)

    # seed_everything()
    # ds = BasicDataset.from_csv(os.path.join(constants.DATASET_PATH, 'asl-signs', 'train.csv'), 3, 1)
    # train_ds, val_ds = ds.random_split(19 / 21)

    # converter.optimizations = [tf.lite.Optimize.DEFAULT]
    # converter.target_spec.supported_types = [tf.float16]
    print('[9 / 9] Converting TFLite model...')
    tflite_model = converter.convert()

    with open(tflite_path, "wb") as f:
        f.write(tflite_model)

    import zipfile

    with zipfile.ZipFile(
            os.path.join(os.path.dirname(model_path[-1]), 'submission.zip'), 'w', compression=zipfile.ZIP_LZMA,
            compresslevel=9
    ) as zip:
        zip.write(tflite_path, 'model.tflite')
