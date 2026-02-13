
import os
import sys
import torch
import torch.nn as nn
from unittest.mock import MagicMock, patch
from omegaconf import OmegaConf

# Mock distributed
sys.modules["torch.distributed"] = MagicMock()
sys.modules["torch.distributed.fsdp"] = MagicMock()
sys.modules["verl.utils.hdfs_io"] = MagicMock()
sys.modules["verl.utils.fs"] = MagicMock()

# Mock imports that might fail in non-GPU or partial env
# We use patch.dict to mock modules during the import of the worker
with patch.dict(sys.modules, {
    "verl.utils.torch_functional": MagicMock(),
    "verl.single_controller.base.decorator": MagicMock(),
    "verl.utils.hf_tokenizer": MagicMock(),
    "verl.workers.sharding_manager.fsdp_ulysses": MagicMock(),
    "verl.utils.flops_counter": MagicMock(),
    "codetiming": MagicMock(),
    "verl.models.registry": MagicMock(),
    "peft": MagicMock(),  # Mock peft itself to avoid ImportErrors if not installed
}):
    # Now import the worker class (pointing to the file we modified)
    sys.path.append(os.getcwd())
    try:
        from verl.workers.fsdp_workers import ActorRolloutRefWorker
    except ImportError:
        # Fallback if imports fail completely
        print("Could not import ActorRolloutRefWorker due to missing dependencies.")
        ActorRolloutRefWorker = MagicMock()

# Mock config
config = OmegaConf.create({
    "actor": {
        "fsdp_config": {"param_offload": False, "grad_offload": False, "optimizer_offload": False},
        "ppo_mini_batch_size": 1,
        "ppo_micro_batch_size": 1,
        "ulysses_sequence_parallel_size": 1,
        # LORA CONFIG TO TEST
        "lora": {
            "r": 16,
            "lora_alpha": 32,
            "target_modules": ["linear"],
            "lora_dropout": 0.05,
            "bias": "none"
        }
    },
    "rollout": {"n": 1, "log_prob_micro_batch_size": 1, "name": "vllm", "tensor_model_parallel_size": 1},
    "ref": {"fsdp_config": {}, "log_prob_micro_batch_size": 1},
    "model": {"path": "dummy_path", "enable_gradient_checkpointing": True}
})

class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 10)
        self.config = MagicMock()
        self.config.model_type = "llama"
        self.config.tie_word_embeddings = False
        
    def gradient_checkpointing_enable(self, **kwargs):
        pass

def test_lora_integration():
    print("Testing LoRA Integration in ActorRolloutRefWorker...")
    
    # We need to mock get_peft_model and LoraConfig inside the module
    with patch("transformers.AutoConfig.from_pretrained") as mock_config, \
         patch("transformers.AutoModelForCausalLM.from_pretrained") as mock_model_cls, \
         patch("verl.workers.fsdp_workers.get_peft_model") as mock_get_peft, \
         patch("verl.workers.fsdp_workers.LoraConfig") as mock_lora_config, \
         patch("torch.distributed.fsdp.FullyShardedDataParallel") as mock_fsdp:
            
        mock_model = DummyModel()
        mock_model_cls.return_value = mock_model
        
        # Instantiate worker (might trigger init logic)
        # We'll just call the specific method we modified if possible, 
        # but the modifications are in _build_model_optimizer which is called during init_model.
        # So let's instantiate and manually call _build_model_optimizer
        
        # Mock Worker init to skip distributed setup
        with patch("verl.workers.fsdp_workers.Worker.__init__"):
            worker = ActorRolloutRefWorker(config, role='actor')
            # Bypass other inits
            worker.device_mesh = MagicMock()
            worker.ulysses_sequence_parallel_size = 1
            worker._is_actor = True
            worker.rank = 0 # Simulate rank 0 for printing
            
            # Call the method
            worker._build_model_optimizer(
                model_path="dummy",
                fsdp_config={},
                optim_config=MagicMock(),
                override_model_config={}
            )
        
        # VERIFICATION
        print("\nVerifying LoraConfig initialization...")
        if mock_lora_config.called:
            _, kwargs = mock_lora_config.call_args
            print("✅ LoraConfig called with:", kwargs)
            if kwargs.get('r') == 16:
                print("   -> Rank correct (16)")
            else:
                print(f"   -> Rank Mismatch: {kwargs.get('r')}")
        else:
            print("❌ LoraConfig was NOT called!")
            
        print("\nVerifying get_peft_model call...")
        if mock_get_peft.called:
             print("✅ get_peft_model was called.")
        else:
             print("❌ get_peft_model was NOT called!")

if __name__ == "__main__":
    try:
        test_lora_integration()
    except Exception as e:
        print(f"\nTest Script Error: {e}")
