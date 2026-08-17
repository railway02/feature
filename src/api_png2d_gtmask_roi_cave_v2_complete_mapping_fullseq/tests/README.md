Run from the package root:

```bash
python -m unittest discover -s tests -v
```

The package was also exercised with a synthetic end-to-end pipeline containing:

- paired `Image.nii.gz + Segmentation.nii.gz`;
- standalone `Post-Segmentation.nii.gz`;
- Pre+Post and Post-only series;
- strict source-phase closure;
- on-the-fly local cropping;
- sharded fake-CAVE extraction;
- full featurebank validation;
- series table generation.
