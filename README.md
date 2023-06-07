# LargeLanguageModel Assist

It is both a workable trainable code and custom snippets for different types of configurations to train LLM.

## About

Mainly it consists of following components -:

##  Datasets
Consists of lightning data Module for more controllable dataset and huggigngface datasets for advanced functionality.

consists of sharding for distributed usage

## Model
consist of default gpt2 model.
we can also load huggignface model.

additionally porting from lightning to huggingface transformers is also being done.for their advanced parametric efficient fine-tuning usage.

##Trainer
consists of lightning trainer.so if you have linux env you can use advanced deepspeed zero for many billion parameters training.

consists of Custom trainer with grad accumulation distributed data parallelism learning rate scheduler and mixed precision.

as above hf model can also be trained with accelerator and peft.

## Profiler
chrome tracing.




 
## Usage
Clone library ,in terminal type "ipconfig" put your ipv4 address instead of master_addr , login with your weight and bias account [comment out 474th line main.py]and run main aka:
```bash
git clone https://github.com/Autobot37/MachineLearningEngine
cd LLM
torchrun --nproc_per_node=1 --nnodes=1 --node_rank=0 --master_addr=123.456.123.456 --master_port=8008 main.py
```

requirements
```bash
lightning
torch
python 3.10
sentencepiece
```


## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

## License

[MIT](https://choosealicense.com/licenses/mit/)
