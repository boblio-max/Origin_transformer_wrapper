import transformers as trs
import torch as th
import peft as p
from torch.utils.data import Dataset as d
import json as j
import random as rnd

class OriginDataset(d):
    def __init__(self, data, tokenizer, max_seq_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        example = self.data[index]
        instruction = example["instruction"]
        output = example["output"]
        formatted_text = "### Instruction\n" + instruction + "\n\n### Response\n" + output

        full_tokens = self.tokenizer(
            formatted_text,
            max_length=self.max_seq_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        input_ids = full_tokens["input_ids"].squeeze(0)
        attention_mask = full_tokens["attention_mask"].squeeze(0)

        labels = input_ids.clone()

        instruction_text = "### Instruction\n" + instruction + "\n\n### Response\n"
        instruction_tokens = self.tokenizer(
            instruction_text,
            max_length=self.max_seq_length,
            truncation=True,
            return_tensors="pt",
            add_special_tokens=False
        )
        instruction_len = instruction_tokens["input_ids"].size(1)
        labels[:instruction_len] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

class transformer:
    def __init__(self):
        pass
    def init(self, data_path, model_name, output_dir):
        self.data_path = data_path
        self.dataset = None
        self.model_name = model_name
        self.output_dir = output_dir
        self.max_seq  = 1
        self.batch_size = 1
        self.epoch = 1
        self.lr = 1e-3
        self.train_data = None
        self.test_data = None
        self.model = None
        self.tokenizer = None
        self.training_args = None
        self.trainer = None
    def config(self, max_seq, batch_size, epoch, lr):
        self.max_seq  = max_seq
        self.batch_size = batch_size
        self.epoch = epoch
        self.lr = lr
        
    def load_data(self):
        with open(self.data_path, "r", encoding="utf-8") as f:
            self.dataset = j.load(f)
            
    def shuffle_data(self):
        rnd.shuffle(self.dataset)

    def split_data(self):
        x = int(len(self.dataset)*0.80)
        self.train_data = self.dataset[:x]
        self.test_data = self.dataset[x:]
    
    def load_tokens(self):
        self.tokenizer = trs.AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
    def load_model(self):
        self.model = trs.AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype=th.float16)
    
    def merge_model(self, lora_config):
        if lora_config is None:
            raise Exception("Lora Config is not generated. Run lora_config = lora(param)")
        else:
            self.model = p.get_peft_model(self.model, lora_config)
    
    def training_args(self, gradient_accumulation, log_dir, log_steps, save_steps, save_t_limit, load_best_model, report=None):
        self.training_args = trs.TrainingArguments(
            output_dir= self.output_dir,
            per_device_train_batch_size=self.batch_size,
            gradient_accumulation_steps=gradient_accumulation,
            num_train_epochs=self.epoch,
            learning_rate=self.lr,
            logging_dir=log_dir,
            logging_steps=log_steps,
            save_steps=save_steps,
            save_total_limit=save_t_limit,
            load_best_model_at_end=load_best_model,
            report_to=report, # Disable reporting to services like W&B
        )

    def init_trainer(self):
        self.trainer = trs.Trainer(
            model = self.model,
            args = self.training_args,
            train_dataset=OriginDataset(self.train_data, self.tokenizer, self.max_seq),
            eval_dataset=OriginDataset(self.test_data, self.tokenizer, self.max_seq),
            tokenizer=self.tokenizer,
        )
    
    def train(self):
        print("Starting training...")
        self.trainer.train()
        self.model.save_pretrained(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)
        print(f"Training complete. Model saved to {self.output_dir}")

    def load_adapter(self, adapter_path):
        self.model = p.PeftModel.from_pretrained(self.model, adapter_path)
        self.model.to("cuda")
        self.model.eval()

    def generate(self, instruction, max_new_tokens=100, temperature=0.1):
        prompt = "### Instruction\n" + instruction + "\n\n### Response\n"
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        with th.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True
            )
        result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        parts = result.split("### Response\n")
        return parts[1].strip() if len(parts) > 1 else result.strip()
