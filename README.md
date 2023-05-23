# MachineLearningEngine
Modular Snippets for flexible Machine Learning Workflow.

Practicaly Machine Learning workflow requires some specefi components.
1- Dataset building -> Here i am focused on distributed sharded training or pytorch fully sharded data parallel.or i simple terms divind big dataset across multiplce devices to
process it in parallel.There are many innovations there like memmap(direct mapping to files) and representing data in a way that require less transferring.

2-Model Part - practically you rarely have to build model from scratch . you have to modify some or ensemble them.This section requires to fit model in device. because sometimes it doesnt.
model parallelsim is a concept and also quantization and torch.compile things may reduce size of model.

3-Training/Finetuning - this is part where it all meets agian distributed data parallel by pytorch help or tricks like gradient accumulation or deepspeed optimizations helps.

4-Profiling/Logging - Plotting and tracking data and device.

Now things are available huggingface and bare pytorch and lightning. But here my goal is pytorch like simplicity.not one liner but a custom framework to moldify.
