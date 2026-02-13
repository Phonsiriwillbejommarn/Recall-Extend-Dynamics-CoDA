
import torch
from verl.trainer.ppo.core_algos import compute_red_weight, compute_accuracy_weight

def test_compute_red_weight():
    print("Testing compute_red_weight...")
    # Case 1: Initial state (no change) -> weight should be 1.0 (clamped min) or G?
    # delta_h = |1 - 1/1| = 0. delta_sft = 0. w = 0/0 -> usually 1.0 handled by epsilon and clamp
    w = compute_red_weight(torch.tensor(1.0), torch.tensor(1.0), torch.tensor(1.0), torch.tensor(1.0), G=5.0)
    print(f"  Init (1.0->1.0): {w.item()} (Expected ~1.0)")
    assert 0.99 <= w.item() <= 1.01

    # Case 2: RL entropy drops (learning), SFT constant
    # h_rl: 1.0 -> 0.5 (delta = |1.0 - 0.5| = 0.5)
    # h_sft: 1.0 -> 1.0 (delta = 0)
    # w = 0 / 0.5 = 0 -> clamped to 1.0
    w = compute_red_weight(torch.tensor(1.0), torch.tensor(0.5), torch.tensor(1.0), torch.tensor(1.0), G=5.0)
    print(f"  RL drops (1.0->0.5), SFT cons: {w.item()} (Expected 1.0)")
    assert w.item() == 1.0

    # Case 3: RL constant, SFT drops (learning)
    # h_rl: 1.0 -> 1.0 (delta = 0)
    # h_sft: 1.0 -> 0.5 (delta = 0.5)
    # w = 0.5 / 1e-8 = huge -> clamped to G (5.0)
    w = compute_red_weight(torch.tensor(1.0), torch.tensor(1.0), torch.tensor(1.0), torch.tensor(0.5), G=5.0)
    print(f"  RL cons, SFT drops: {w.item()} (Expected 5.0)")
    assert w.item() == 5.0

    # Case 4: Both drop equal amount (Absolute difference)
    # h_rl: 2.0 -> 1.5 (delta = 0.5)
    # h_sft: 1.0 -> 0.8 (delta = 0.2)
    # w = 0.2 / 0.5 = 0.4 -> clamped to 1.0
    # Note: New logic uses ABSOLUTE change.
    w = compute_red_weight(torch.tensor(2.0), torch.tensor(1.5), torch.tensor(1.0), torch.tensor(0.8), G=5.0)
    print(f"  RL(2.0->1.5), SFT(1.0->0.8): {w.item()} (Expected 1.0)")
    assert 0.99 <= w.item() <= 1.01

def test_compute_accuracy_weight():
    print("\nTesting compute_accuracy_weight...")
    G = 5.0
    # Acc = 0.0 -> Factor G
    w = compute_accuracy_weight(0.0, G)
    print(f"  Acc 0.0: {w} (Expected {G})")
    assert abs(w - G) < 1e-5

    # Acc = 0.5 -> Factor 1.0
    w = compute_accuracy_weight(0.5, G)
    print(f"  Acc 0.5: {w} (Expected 1.0)")
    assert abs(w - 1.0) < 1e-5

    # Acc = 1.0 -> Factor 1/G
    w = compute_accuracy_weight(1.0, G)
    print(f"  Acc 1.0: {w} (Expected {1/G})")
    assert abs(w - 1/G) < 1e-5

if __name__ == "__main__":
    test_compute_red_weight()
    test_compute_accuracy_weight()
    print("\n✅ All tests passed!")
