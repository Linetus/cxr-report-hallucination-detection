import torch
print("torch", torch.__version__, "| CUDA", torch.cuda.is_available(), "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-")
import bitsandbytes as bnb
print("bitsandbytes", bnb.__version__)
from llava.model.builder import load_pretrained_model
from llava.conversation import conv_templates
from llava.mm_utils import tokenizer_image_token, KeywordsStoppingCriteria
from llava.constants import IMAGE_TOKEN_INDEX
print("IMPORTS LLAVA OK")
print("conv modes disponibles:", [k for k in conv_templates.keys()][:8])
