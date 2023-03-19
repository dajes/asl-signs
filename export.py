import os

import onnxsim
import torch
from tensorflow.python.ops.image_ops_impl import ResizeMethod

import constants
from dataset.basic import BasicDataset

CLIP_CONTEXT = 10000
model_path = os.path.join(constants.MODELS_PATH, 'asl_0321/asl_0321.ckpt')
onnx_path = os.path.splitext(model_path)[0] + '.onnx'
sm_path = os.path.splitext(model_path)[0] + '.pb'
tf_path = os.path.splitext(model_path)[0] + '.tf'
tflite_path = os.path.splitext(model_path)[0] + '.tflite'


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

    def forward(self, x):
        x = x[:, self.relevant_ids, :self.n_coords]
        x = x.reshape(1, x.shape[0], x.shape[1] * x.shape[2])
        x[torch.isnan(x)] = 0
        return self.model(x)


if __name__ == '__main__':
    import numpy as np
    import onnx
    import tensorflow as tf
    from onnx_tf.backend import prepare
    from module import Module

    module = Module.load_from_checkpoint(
        model_path, 'cpu'
    )
    inference = Inference(module.model).eval()
    inputs = torch.rand((50, 543, 3))

    with torch.no_grad():
        truth = inference(inputs)
    torch.onnx.export(
        model=inference,
        args=inputs,
        f=onnx_path,
        opset_version=12,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "input"}},
    )

    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    model_onnx, check = onnxsim.simplify(onnx_model)
    assert check
    onnx.save(model_onnx, onnx_path)

    tf_rep = prepare(onnx_model)
    tf_rep.export_graph(sm_path)


    class ASLInferModel(tf.Module):
        def __init__(self, path):
            super().__init__()
            self.model = tf.saved_model.load(path)
            self.model.trainable = False
            self.max_len = inference.max_len
            self.pad_or_trim = inference.pad_or_trim

        @tf.function(
            input_signature=[
                tf.TensorSpec(shape=[None, 543, 3], dtype=tf.float32, name="inputs")
            ]
        )
        def call(self, input):
            if self.pad_or_trim:
                # pad or trim x to have sequence length of max_len
                if tf.shape(input)[0] > self.max_len:
                    input = tf.image.resize(input[None], (self.max_len, 543), ResizeMethod.NEAREST_NEIGHBOR)[0]

            output_tensors = {
                "outputs": self.model(
                    **{"input": input}
                )["output"][0, :]
            }
            return output_tensors


    mytfmodel = ASLInferModel(sm_path)
    tf.saved_model.save(
        mytfmodel, tf_path, signatures={"serving_default": mytfmodel.call}
    )
    r = mytfmodel.call(inputs.numpy())['outputs'].numpy()

    error = np.abs(r - truth.numpy()).max()
    print('Max error:', error)
    print('Cosine similarity:', np.dot(r, truth[0].numpy()) / (np.linalg.norm(r) * np.linalg.norm(truth.numpy())))

    converter = tf.lite.TFLiteConverter.from_saved_model(tf_path)
    tflite_model = converter.convert()

    with open(tflite_path, "wb") as f:
        f.write(tflite_model)

    import zipfile

    with zipfile.ZipFile(
            os.path.join(os.path.dirname(model_path), 'submission.zip'), 'w', compression=zipfile.ZIP_LZMA,
            compresslevel=9
    ) as zip:
        zip.write(tflite_path, 'model.tflite')
