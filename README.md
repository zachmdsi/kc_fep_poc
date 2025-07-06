# Kolmogorov Complexity and Free Energy in Active Inference

This project explores a simple question: can an intelligent agent compress its own sensory history more efficiently than a standard lossless compressor, and if so, do the saved bits line up with a lower free-energy score? Each saved bit would provide direct evidence that the agent’s model captures structure in the data and that the free-energy principle makes a falsifiable prediction.

Kolmogorov complexity is the length of the shortest computer program that can reproduce a given data stream. Although we cannot compute this length exactly, we can approximate an upper bound using a parametric model. When the bound becomes smaller, it means the description of the sensory history has become shorter.

Free energy is a score that combines two ideas. The first is how well the model predicts the data we observe. The second is how simple the model itself remains. A lower score means the model explains the data efficiently without unnecessary complication. When the description of the data gets shorter, the free-energy score should drop as well. That connection allows us to test the free-energy principle in a concrete way.

The experimental design is straightforward. Each episode of interaction produces three numbers:

- The length of the agent’s description of its sensory history.
- The length produced by a strong off-the-shelf lossless compressor.
- The agent’s free-energy score.

These numbers let us trace learning progress episode by episode.

The difference between the compressor’s length and the agent’s length is the compression gap. Tracking that gap over time reveals whether the agent is learning to represent the world more concisely than a universal compressor. We can also track the relation between the compression gap and the free-energy score, which we call the efficiency trend. When the agent becomes more skilled at compression, the free-energy score should fall proportionally.

Three predictions follow from this setup. First, the compression gap should never shrink for long once learning begins. A steady decline would suggest the model is failing to capture structure. Second, the better the compression gets, the more the free-energy score should drop—the two numbers should move together. Third, after enough experience in a stable environment, the agent’s description of the data should be shorter than the one produced by the compressor.

Confirming these trends would support the idea that active inference not only guides actions but also shortens the algorithmic description of experience. If those trends disappear, we learn that the link may be weaker than theorised. Either outcome delivers a clear, measurable test of the free-energy principle.

## Quick start

1. Create a `config.yml` file describing the experiment. A minimal example:

   ```yaml
   episodes: 3
   p: 0.5
   steps: 128
   csv_path: metrics.csv
   ```

2. Run the experiment:

   ```bash
   kc-fep-run
   ```

This writes `metrics.csv` and prints a table summarising each episode.
