# Demo phantom

`mni152_phantom.npz` is a 224 x 224 axial slice (1 mm, MNI z = -28 mm: temporal poles,
orbitofrontal cortex, pons and cerebellum) with

- `image`: T2*-like magnitude built from the ICBM 152 2009a nonlinear symmetric T1 template and its
  GM / WM probability maps (CSF bright, GM > WM, bone dark, scalp fat bright);
- `b0`: field map in Hz at 3 T from a susceptibility model of the head (tissue -9.4 ppm against
  air; frontal and sphenoid sinuses, nasal cavity and mastoid cells placed against the brain
  surface, with the face and neck below the skull base), computed with the dipole kernel,
  linearly shimmed over the brain, smoothed by 2 mm and scaled by 0.8 so that the EPI
  displacement of the default example (224 lines, 0.25 ms) stays invertible everywhere;
- `head` (brain plus skull/scalp shell, i.e. the signal support), `brain`: masks.

`mni152_phantom_preview.png` shows the template slice, the phantom, the field map and the cavity layout.

`build_mni_phantom.py` regenerates it for any slice position (`--z`) and shim order; it needs
`nilearn`, which ships the template. `../make_synthetic_example.py` turns the phantom into the
rotated-view EPI input of `main.py`.

The template is distributed under its original terms: Copyright (C) 1993-2009 Louis Collins,
McConnell Brain Imaging Centre, Montreal Neurological Institute, McGill University. Permission to
use, copy, modify, and distribute this software and its documentation for any purpose and without
fee is hereby granted, provided that the above copyright notice appear in all copies. The authors
and McGill University make no representations about the suitability of this software for any
purpose. It is provided "as is" without express or implied warranty. Fonov V. et al., NeuroImage
54(1):313-327, 2011; Fonov V. et al., NeuroImage 47(S1):S102, 2009.
