import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint

from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from torch.utils.data import DataLoader, RandomSampler
from transformers.modeling_outputs import CausalLMOutputWithPast

def forward(
        self,
        input_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        task_ids=None,
        **kwargs,
    ):
    '''inspected from peft code; actually running model.base_model.forward -> model.model.forward'''
    peft_config = self.active_peft_config
    if not peft_config.is_prompt_learning:
        if self.base_model.config.model_type == "mpt":
            if inputs_embeds is not None:
                raise AssertionError("forward in MPTForCausalLM does not support inputs_embeds")
            return self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                **kwargs,
            )

        return self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            labels=labels,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )

    batch_size = _get_batch_size(input_ids, inputs_embeds)
    if attention_mask is not None:
        # concat prompt attention mask
        prefix_attention_mask = torch.ones(batch_size, peft_config.num_virtual_tokens).to(attention_mask.device)
        attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)

    if kwargs.get("position_ids", None) is not None:
        warnings.warn("Position ids are not supported for parameter efficient tuning. Ignoring position ids.")
        kwargs["position_ids"] = None
    if kwargs.get("token_type_ids", None) is not None:
        warnings.warn("Token type ids are not supported for parameter efficient tuning. Ignoring token type ids")
        kwargs["token_type_ids"] = None
    kwargs.update(
        {
            "attention_mask": attention_mask,
            "labels": labels,
            "output_attentions": output_attentions,
            "output_hidden_states": output_hidden_states,
            "return_dict": return_dict,
        }
    )

    if peft_config.peft_type == PeftType.PREFIX_TUNING:
        past_key_values = self.get_prompt(batch_size)
        return self.base_model(
            input_ids=input_ids, inputs_embeds=inputs_embeds, past_key_values=past_key_values, **kwargs
        )
    else:
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        # concat prompt labels
        if labels is not None:
            prefix_labels = torch.full((batch_size, peft_config.num_virtual_tokens), -100).to(labels.device)
            kwargs["labels"] = torch.cat((prefix_labels, labels), dim=1)
        prompts = self.get_prompt(batch_size=batch_size, task_ids=task_ids)
        prompts = prompts.to(inputs_embeds.dtype)
        inputs_embeds = torch.cat((prompts, inputs_embeds), dim=1)
        return self.base_model(inputs_embeds=inputs_embeds, **kwargs)


# @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
# @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
def base_model_forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
) -> Union[Tuple, CausalLMOutputWithPast]:
    r"""
    Args:
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

    Returns:

    Example:

    ```python
    >>> from transformers import AutoTokenizer, LlamaForCausalLM

    >>> model = LlamaForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
    >>> tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)

    >>> prompt = "Hey, are you conscious? Can you talk to me?"
    >>> inputs = tokenizer(prompt, return_tensors="pt")

    >>> # Generate
    >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
    >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
    ```"""

    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
    outputs = self.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )

    hidden_states = outputs[0]
    if self.config.pretraining_tp > 1:
        lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
        logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
        logits = torch.cat(logits, dim=-1)
    else:
        logits = self.lm_head(hidden_states)
    logits = logits.float()

    loss = None
    if labels is not None:
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
#             loss_fct = CrossEntropyLoss()
        loss_fct = CrossEntropyLoss(reduction='none')
        shift_logits = shift_logits.view(-1, self.config.vocab_size)
        shift_labels = shift_labels.view(-1)
        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)
        loss = loss_fct(shift_logits, shift_labels)

    if not return_dict:
        output = (logits,) + outputs[1:]
        return (loss,) + output if loss is not None else output

    return CausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )

def _get_train_sampler(self, sampler=None) -> Optional[torch.utils.data.Sampler]:
        print("_get_train_sampler is called")
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None

        # Build the sampler.
#         if self.args.group_by_length:
#             if is_datasets_available() and isinstance(self.train_dataset, datasets.Dataset):
#                 lengths = (
#                     self.train_dataset[self.args.length_column_name]
#                     if self.args.length_column_name in self.train_dataset.column_names
#                     else None
#                 )
#             else:
#                 lengths = None
#             model_input_name = self.tokenizer.model_input_names[0] if self.tokenizer is not None else None
#             return LengthGroupedSampler(
#                 self.args.train_batch_size * self.args.gradient_accumulation_steps,
#                 dataset=self.train_dataset,
#                 lengths=lengths,
#                 model_input_name=model_input_name,
#             )
#
#         else:
#             return RandomSampler(self.train_dataset)
        return self.train_dataset.pruning_sampler()

def get_train_dataloader(self) -> DataLoader:
    """
    Returns the training [`~torch.utils.data.DataLoader`].

    Will use no sampler if `train_dataset` does not implement `__len__`, a random sampler (adapted to distributed
    training if necessary) otherwise.

    Subclass and override this method if you want to inject some custom behavior.
    """
    print("get_train_dataloader is called")
    if self.train_dataset is None:
        raise ValueError("Trainer: training requires a train_dataset.")

    train_dataset = self.train_dataset
    data_collator = self.data_collator
    if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
        train_dataset = self._remove_unused_columns(train_dataset, description="training")
    else:
        data_collator = self._get_collator_with_removed_columns(data_collator, description="training")

    dataloader_params = {
        "batch_size": self._train_batch_size,
        "collate_fn": data_collator,
        "num_workers": self.args.dataloader_num_workers,
        "pin_memory": self.args.dataloader_pin_memory,
    }

    if not isinstance(train_dataset, torch.utils.data.IterableDataset):
        dataloader_params["sampler"] = self._get_train_sampler()
        dataloader_params["drop_last"] = self.args.dataloader_drop_last
        dataloader_params["worker_init_fn"] = seed_worker

    return self.accelerator.prepare(DataLoader(train_dataset, **dataloader_params))
