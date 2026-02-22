import transformers
print(transformers.__version__)
from transformers import modeling_rope_utils
# 아래 코드가 에러 없이 통과해야 함
print(modeling_rope_utils.RoPEParameters) 
