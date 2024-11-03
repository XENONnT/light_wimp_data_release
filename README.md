# XENONnT Light WIMP data release

XENON collaboration, 2024

Contact: Lanqing Yuan (yuanlq@uchicago.edu) and [Shenyang Shi](ss6109@columbia.edu)

## Scope 

 * This release contains data from the analysis decribed in the paper [First Search for Light Dark Matter in the Neutrino Fog with XENONnT](https://arxiv.org/abs/2409.17868).
 * A tool for recasting our results, to get limits on customized new physics, or to reinterpret our results by getting limits on a considered dark matter model but with a different yield model. 


## Installation

Run `pip install -e ./` to install the essential dependencies.

## Contents

This data release are structued as follows:

  * `lightwimp_data_release/data` contains the templates and signal spectrum, with the following catagories:
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
