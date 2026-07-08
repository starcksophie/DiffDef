# Diff-Def: Diffusion-Generated Deformation Fields for Conditional Atlases

Anatomical atlases are essential tools in population studies and medical research, particularly for investigating sub-populations with specific characteristics (e.g., age, disease status). We propose a novel method for generating conditional atlases using latent diffusion models, which model deformation fields to transform a general population atlas into a sub-population-specific atlas. Our approach enhances interpretability, maintains structural integrity, and minimizes hallucinations in generated images. This method outperforms existing atlas generation techniques in generating realistic and anatomically accurate atlases, as demonstrated on brain MR images from the UK Biobank dataset.

## Data

The code uses magnetic resonance imaging (MRI) data from the UK Biobank dataset. Due to data sharing agreements, the original UK Biobank data cannot be redistributed.

## Requirements

- Python 3.11.3
- Install required packages using the `requirements.txt` file:
  ```bash
  pip install -r requirements.txt
  ```

## Citation
```
  @article{starck2024diff,
  title={Diff-Def: Diffusion-Generated Deformation Fields for Conditional Atlases},
  author={Starck, Sophie and Sideri-Lampretsa, Vasiliki and Kainz, Bernhard and Menten, Martin and Mueller, Tamara and Rueckert, Daniel},
  journal={arXiv preprint arXiv:2403.16776},
  year={2024}
}
```

## Contact

For any questions you can contact Sophie Starck (sophie.starck@tum.de) or Vasiliki Sideri-Lampretsa (vasiliki.sideri-lampretsa@tum.de).
