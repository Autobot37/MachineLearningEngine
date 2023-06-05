from transformers import PreTrainedModel, PretrainedConfig
from transformers import AutoModel, AutoConfig
from settings import ModelConfig
from defaultmodel import plModel
class HFConfig(PretrainedConfig):
    model_type="GPT"
    def __init__(self,config,**kwargs):
        super().__init__(**kwargs)
        self.set_attributes(config)
    
    def set_attributes(self,obj):
        attributes = vars(obj)
        for attr, value in attributes.items():
            setattr(self, attr, value)


class HFModel(PreTrainedModel):
    config_class = HFConfig
    def __init__(self, config):
        super().__init__(config)
        self.model = plModel(config)
    
    def forward(self,x):
        return self.model(x)


# config = MyConfig(config)
# model = MyModel(config)
# model.save_pretrained('./my_model_dir')

