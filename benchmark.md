# Grasping metrics for different pipeline

Success rate and run time.

# Dataset structure

Number of samples. Number of objects. Label distribution.

# Grasp Generator

Generate specific grasp samples given the partial point cloud of the object.

Input: partial point cloud in BPS format, palm pose (R+t) and finger conf.

Output: palm pose (R+t) and finger conf.

How currently it's evaluated:

1. visualize the generated grasps and the grasps from simulation -> similar distribution
2. Evaluate the generated grasps in simulation -> 61% success rate

## CVAE

Latent space: [5,] mu and [5,] logvar
What's the requirement for the block used to predict latent space?

## Metrics

1. Loss function
weighted sum of KL loss + pose L2 loss (rot + transl) + joint conf L2 loss

2. Number of parameters

3. Flops (total number of floating point operations required for a single forward pass)

### PointNet/ PointNet ++

- [ ] real time capability?
From paper: PointNet is able to process more than one million points per second for point cloud classification
kit_Patches_pcd031.pcd --> one example from KIT dataset has 7k point clouds -> 7ms run time!

However, according to PointNet++ paper, the runtime of PointNet++ is 4-8 times slower than PointNet.

- [ ] New dataloader with raw point cloud data
- [ ] PointNet pretrained or training from scratch? It's mainly MLPs so train from scratch.
- [ ] Find pytorch implementation
- [ ] train: fixed number of points 4096 like basis point set. This is how PointNet paper is implemented for training.

### Transformer

# Grasp Evaluator

## Metrics

Loss function:
"""Computes the binary cross entropy loss between predicted success-label and true success"""
        bce_loss_val = self.bce_weight * self.BCE_loss(pred_success_p, self.FFHEvaluator.gt_label)

# Benchmark

| Metrics  | FFHNet Generator  |   |   |   |
|----------|-------------------|---|---|---|
| Speed    |  10ms for 1000 grasps   |   |   |   |
| params   |  14.0 million             |   |   |   |
|          |                   |   |   |   |

| Metrics  | FFHNet Evaluator  |   |   |   |
|----------|-------------------|---|---|---|
| Speed    |  20ms for 1000 grasps  |   |   |   |
| params   |   10.6 million             |   |   |   |
|          |                   |   |   |   |
