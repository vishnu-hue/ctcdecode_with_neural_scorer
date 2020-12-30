import torch
import onnx
import onnxruntime
from onnxruntime.quantization import quantize_dynamic, QuantType
import numpy as np
from transformers import GPT2Tokenizer, GPT2LMHeadModel
tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2')
model = GPT2LMHeadModel.from_pretrained('distilgpt2')
inputs = dict()
inputs["input_ids"] = tokenizer("Hello,", return_tensors="pt")["input_ids"]
print(inputs)
outputs = model(**inputs)
loss = outputs[0]
logits = outputs[1]
print(outputs[1][0].size())
input_name = ["input_ids"]
input = inputs["input_ids"]
print(input)
output_name = ["output_0"]
print(input.shape)
batch_size = 1
torch.onnx.export(model, input, "gpt2.onnx", input_names = input_name, output_names = output_name, opset_version=11, dynamic_axes={'input_ids' : {1 : 'batch_size'}})
onnx_model_path = "gpt2.onnx"
quantized_model_path = "quant_gpt2.onnx"
onnx_opt_model = onnx.load(onnx_model_path)
quantize_dynamic(onnx_model_path,
                quantized_model_path,
                weight_type=QuantType.QInt8)
vocab = tokenizer.get_vocab()
print(len(vocab))
#cpu_model = onnx.load("gpt2.onnx")
cpu_model = onnxruntime.InferenceSession("quant_gpt2.onnx")
# Inputs are provided through numpy array
model_inputs = tokenizer.encode_plus("The capital")
input = dict()
input["input_ids"] = model_inputs["input_ids"]
print(input)
inputs_onnx = {k: np.atleast_2d(v) for k, v in input.items()}
print(inputs_onnx)
# Run the model (None = get all the outputs)
sequence= cpu_model.run(None, inputs_onnx)
#sequence = cpu_model.generate(model_inputs['input_ids'])
# Print information about outputs
print(sequence[0][0][0][5])
print(sequence[0][0][1][5])
print(sequence[0].shape)
print(f"Sequence output: {sequence[1][-1].shape}")
print(cpu_model.get_outputs()[1].name)
