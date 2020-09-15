from coremltools.converters.mil import register_torch_op
from coremltools.converters.mil.frontend.torch.ops import _get_inputs
from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.frontend.torch.torch_op_registry import _TORCH_OPS_REGISTRY
import numpy as np


@register_torch_op
def constant_pad_nd(context, node):
    inputs = _get_inputs(context, node, expected=3)
    new_pad = inputs[1].val.reshape((-1, 2))[::-1].reshape(-1).tolist()
    new_pad = [0]*(2*len(inputs[0].shape)-len(new_pad)) + new_pad
    padded = mb.pad(x=inputs[0], pad=np.array(
        new_pad), mode="constant", constant_val=float(0), name=node.name)
    context.add(padded)
