from typing import Optional
import sentencepiece
from sentencepiece import SentencePieceProcessor,SentencePieceTrainer

#@title
#Tokenizer[model path]

class Tokenizer:

  def __init__(self, model_path):
    self.processor = SentencePieceProcessor(model_file=str(model_path))
    self.bos_id = self.processor.bos_id()
    self.eos_id = self.processor.eos_id()
    self.pad_id = self.processor.pad_id()
  
  @property
  def vocab_size(self) -> int:
    return self.processor.vocab_size()


#encode
  def encode(
        self,
        string: str,
        bos: bool = False,
        eos: bool = False,
        max_length: int = -1,
        pad: bool = False,
        device: Optional[torch.device] = None
    ) -> torch.Tensor:
    tokens = self.processor.encode(string)
    if bos:
        tokens = [self.bos_id] + tokens
    if eos:
        tokens = tokens + [self.eos_id]
    if max_length > 0:
        tokens = tokens[:max_length]
    if pad and len(tokens) < max_length:
        tokens += [self.pad_id] * (max_length - len(tokens))

    return torch.tensor(tokens, dtype=torch.long, device=device)


#decode
  def decode(self,tokens:torch.Tensor) -> str:
    return self.processor.decode(tokens.tolist())
  
#train[take input txt bro][have export it on path]
  @staticmethod
  def train(input: str, destination: str, vocab_size=2000) -> None:
    model_prefix = os.path.join(destination, "tokenizer")
    SentencePieceTrainer.Train(input=input, model_prefix=model_prefix, vocab_size=vocab_size)

#@title
# Tokenizer.train(input="/content/data/input.txt",destination="/content/tokenizer")

# #@title
# enc = Tokenizer("/content/tokenizer/tokenizer.model")

