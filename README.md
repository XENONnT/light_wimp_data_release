# XENONnT Light WIMP data release

Data release and tools for data re-interpretation for [XENONnT's first search for light dark matter](https://arxiv.org/abs/2409.17868).

XENON collaboration, 2024

Contact: Lanqing Yuan (yuanlq@uchicago.edu) and Shenyang Shi (ss6109@columbia.edu)

## Scope 

 * This release contains data from the analysis, and the final results decribed in the paper [First Search for Light Dark Matter in the Neutrino Fog with XENONnT](https://arxiv.org/abs/2409.17868).
 * A tool for recasting our results, to get limits on customized new physics, or to reinterpret our results by getting limits on a considered dark matter model but with a different yield model. 

## Citation

Please cite the paper by
```
@article{XENON:2024hup,
    author = "Aprile, E. and others",
    collaboration = "XENON",
    title = "{First Search for Light Dark Matter in the Neutrino Fog with XENONnT}",
    eprint = "2409.17868",
    archivePrefix = "arXiv",
    primaryClass = "hep-ex",
    month = "9",
    year = "2024"
}
```
and then cite this package from Zenodo. 

## Installation

Run `pip install -e ./` to install the essential dependencies.

## Contents

This package is structued as follows:

  * `notebooks` contains pedagogical notebooks helping you recasting our results.
  * `lightwimp_data_release/data` contains the templates and signal spectrum to be used in recasting our results, with the following catagories:
    * Mono-enegetic simulations for each possible light yield and charge yield combination.
    * Background used in the analysis.
      * `ac`: Accndental coincidence background.
      * `cevns`: Solar $^8\mathrm{B}$ $\mathrm{CE}\nu\mathrm{NS}$ background.
      * `rg`: Radiogenic neutron background.
      * `er`: Electronic recoil background.
    * Signals used in the analysis:
      * `wimp_si`: Spin-independent WIMP signal.
      * `wimp_si_n_1/2/m2`: Momemtum dependent dark matter.
      * `mirror_dm`: Mirror dark matter (dark oxygen).
  * `lightwimp_data_release/limits` contains the data points in FIG 3 of the [paper](https://arxiv.org/abs/2409.17868).

## Caveat for recasting

The recasting tool introduced in `notebooks` allows users to reinterpret our data in typically two ways:
  * (Typically for theorists): Input a different signal model and get constraints. 
  * (Typically for experimentalists): Input a different yield model, and apply it to the already considered signal models (eg. a WIMP of SI interaction) to get new constraints.

There are two apprxoimations we introduced on the basis of [standard XENON statistical inference procedures](https://arxiv.org/abs/2406.13638) applied in this [paper](https://arxiv.org/abs/2409.17868):
  * The templates are produced by a fast interpolation method, with details described in [notebook 1](https://github.com/XENONnT/light_wimp_data_release/tree/master/notebooks). This gives negligible bias on the final limit.
  * For computational concern, we only showed users the limit obtained by assuming [asymptotic Neyman threshold](https://arxiv.org/abs/2406.13638), which might make the limit over-convservative by ~30%. However, by following the procedure described in [standard XENON statistical inference](https://arxiv.org/abs/2406.13638), with the contents provided by this package, the user can in principle perform inference with toyMC-based Neyman construction to resolve this bias. We didn't show how to do it here because of the nontrivial computation burden.

The recasting method presented in the noteboks has been extensively tested for accuracy and bias, which leads to the conclusions in the bullet points above. Below we show a test we did:

![wimp_si_benchmark_recast](https://github.com/user-attachments/assets/9eeee703-0791-4a4e-b69b-2f0e73f61779)

In this test for WIMP of spin-independent interaction, we use the recasting method to generate templates and perform fast inference with asymptotic Neyman threshold approxiamtion (blue). As a comparison, we also use the official templates used in the paper and perforom the same fast inference (orange). For both results, we calculate the ratio between them and the rigorously-derived result in paper (before power constrained limit). One can see that the "Recasted" and "Original" are differentiated by less than 5%, which suggests the high accuracy of fast template interpolation method. The limit ratio between the fast inference method and rigorous method, which is around 100% for lower mass but increases up to ~135% for higher mass, characterizes the systematic bias for using asymptotic Neyman thresholds. Fortunately, such bias is usually toward the more conservative direction.

Users of this package should keep in mind that, **the limit you obtained following this notebooks might be ~30% over-conservative**. 
