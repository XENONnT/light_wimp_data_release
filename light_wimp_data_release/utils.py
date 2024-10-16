import numpy as np
import os
from pathlib import Path
from warnings import warn
from tqdm.notebook import tqdm
import importlib.resources
from scipy.interpolate import RegularGridInterpolator
from inference_interface import template_to_multihist
from scipy.interpolate import interp1d
from appletree.utils import load_json, integrate_midpoint, cumulative_integrate_midpoint

Ek_MIN = 0.51  # keV
Ek_MAX = 5.0  # keV
SAMPLE_SIZE = int(2e4)
BASIS_PATH = str(
    importlib.resources.files("light_wimp_data_release.data.orthonormal_basis")
)
JSON_PATH = importlib.resources.files("light_wimp_data_release.data") / "signal"
JSON_PATH = Path(JSON_PATH)

BASIS_Ek = np.concatenate(
    [[0.51, 0.6, 0.7, 0.8, 0.9], np.arange(1, 2, 0.25), np.arange(2, 5.5, 0.5)]
)
LY_SWEEP = np.arange(1, 11, 1)
QY_SWEEP = np.arange(1, 11, 1)

def get_yield(t, lower, median, upper, y_max=9, y_min=1):
    """
    Get yield based on yield parameter tly or tqy:
        yield = median + (upper - median) * t if t >= 0
        yield = median + (lower - median) * t if t < 0    
    Truncate the yield to be within y_max and y_min
    :param t: yield parameter tly or tqy
    :param lower: lower yield model dict with keys: coordinate_system, map (unit: quanta/keV)
    :param median: median yield model dict with keys: coordinate_system, map (unit: quanta/keV)
    :param upper: upper yield model dict with keys: coordinate_system, map (unit: quanta/keV)
    :param y_max: maximum yield value (unit: quanta/keV)
    :param y_min: minimum yield value (unit: quanta/keV)
    :return: yield model dict with keys: coordinate_system, map (unit: quanta/keV)
    """
    # Check if the coordinate system is the same
    assert median['coordinate_system'] == lower['coordinate_system']
    assert median['coordinate_system'] == upper['coordinate_system']
    
    d_down = np.array(median['map']) - np.array(lower['map'])
    d_up = np.array(upper['map']) - np.array(median['map'])
    
    if t >= 0:
        new_map = median['map'] + d_up * t
    elif t < 0:
        new_map = median['map'] + d_down * t

    # Cap the yield by brutal force
    new_map[new_map>=y_max] = y_max
    new_map[new_map<=y_min] = y_min

    return {
        "coordinate_system": list(median['coordinate_system']),
        "map": list(new_map)
    }

class Template:
    """
    Build templates for a specific signal model and yield model with basis mono-energetic simulations
    """

    def __init__(self):

        self.interpolators = {
            "sr0": self.interpolator("sr0"),
            "sr1": self.interpolator("sr1"),
        }

    def required_keys(self, type):
        if type == "signal":
            return [
                "coordinate_system",  # pcf or cdf distrubution, normalized to 1
                "coordinate_name",  # pdf or cdf
                "map",  # recoil energies in keV
                "rate",  # total event rate normalization events/tonne/year liquid xenon
            ]
        elif type == "yield":
            return [
                "coordinate_system",  # pcf or cdf distrubution, normalized to 1
                "map",  # recoil energies in keV
            ]

    def assert_keys_in_dict(self, dictionary, required_keys):
        for key in required_keys:
            assert key in dictionary, f"Missing required key: {key}"

    def build_template(self, signal_spectrum, yield_model: dict = None):
        """
        Build a return signal template for sr0 and sr1
        :return:
        """
        # Initalize signal model
        signal_spectrum = self.format_custom_signal_spectrum(signal_spectrum)

        # Initialize yield model
        if yield_model is None:
            self.yield_model = self.default_yield_model()
        else:
            self.yield_model = self.format_custom_yield_model(yield_model)

        results = {}
        for sr in ["sr0", "sr1"]:
            results[sr] = self._build_template(signal_spectrum, sr)
        return results

    def _build_template(self, signal_spectrum, sr):
        """
        Build a return signal template for sr0 and sr1
        :return:
        """
        inverse_cdf = interp1d(
            signal_spectrum["coordinate_system"], signal_spectrum["map"]
        )
        samples = np.random.uniform(0, 1, SAMPLE_SIZE)
        ek_samples = inverse_cdf(samples)
        roi_mask = (ek_samples > Ek_MIN) & (ek_samples < Ek_MAX)
        ek_samples = ek_samples[roi_mask]  # Truncate to recoil energy in the ROI

        template = self.base_templates(sr)
        template.histogram = (
            sum(
                [
                    self.interpolators[sr](
                        [
                            ek,
                            np.float64(self.yield_model["ly"](ek)),
                            np.float64(self.yield_model["qy"](ek)),
                        ]
                    )
                    for ek in tqdm(ek_samples, desc="Basis interpolation", leave=False)
                ]
            )
            / SAMPLE_SIZE
            * signal_spectrum["rate"]
        )
        return template

    def interpolator(self, sr):
        """
        Interpolate to get any mono-energetic and yield model template
        :return: Interpolator
        """
        anchors_array = [BASIS_Ek, LY_SWEEP, QY_SWEEP]
        anchors_grid = self.arrays_to_grid(anchors_array)

        extra_dims = [3, 3, 3, 3]  # templates dimension
        anchor_scores = np.zeros(list(anchors_grid.shape)[:-1] + extra_dims)
        for i_e, e in tqdm(enumerate(BASIS_Ek)):
            for i_ly, ly in enumerate(LY_SWEEP):
                for i_qy, qy in enumerate(QY_SWEEP):
                    e = f"{str(e).replace('.', '_')}"
                    file_name = os.path.join(
                        BASIS_PATH,
                        f"template_XENONnT_{sr}_recasting_mono_{e}_cevns_tly_{ly}_tqy_{qy}.h5",
                    )
                    mh = template_to_multihist(file_name)
                    anchor_scores[i_e, i_ly, i_qy, :] = mh["template"]

        itp = RegularGridInterpolator(anchors_array, anchor_scores)
        return lambda *args: itp(*args)[0]

    def base_templates(self, sr):
        """
        Return a base templates with correct bin boundary and axis name,
        lazy solution, load saved templates and overwrite histogram with correct ones later
        :param sr:
        :return:
        """
        template = os.path.join(
            BASIS_PATH,
            f"template_XENONnT_{sr}_recasting_mono_1_0_cevns_tly_7_tqy_7.h5",  # I like the number 7
        )
        return template_to_multihist(template)["template"]

    def default_yield_model(self):
        """
        Return default yield model based on YBe calibration
        :return:
        """
        ly = load_json(os.path.join(JSON_PATH, f"nr_ly_ybe_only_median_sr1_cevns.json"))
        qy = load_json(os.path.join(JSON_PATH, f"nr_qy_ybe_only_median_sr1_cevns.json"))
        ly = interp1d(ly["coordinate_system"], ly["map"])
        qy = interp1d(qy["coordinate_system"], qy["map"])
        return {"ly": ly, "qy": qy}

    def format_custom_signal_spectrum(self, signal_spectrum):
        """
        Format the user input signal spectrum
        :param signal_spectrum:
        :return:
        """
        self.assert_keys_in_dict(signal_spectrum, self.required_keys("signal"))
        # We need cdf finally! Need to convert pdf to cdf if pdf is provided

        assert signal_spectrum["coordinate_name"] in [
            "pdf",
            "cdf",
        ], "coordinate_name must be pdf or cdf"
        if signal_spectrum["coordinate_name"] == "pdf":
            warn(f"Convert signal spectrum from pdf to cdf")
            x, cdf = self.pdf_to_cdf(
                signal_spectrum["coordinate_system"], signal_spectrum["map"]
            )
            signal_spectrum["coordinate_name"] = "cdf"
            signal_spectrum["coordinate_system"] = cdf
            signal_spectrum["map"] = x
        else:
            pass

        return signal_spectrum

    def pdf_to_cdf(self, x, pdf):
        """Convert pdf map to cdf map."""
        norm = integrate_midpoint(x, pdf)
        x, cdf = cumulative_integrate_midpoint(x, pdf)
        cdf /= norm
        return x, cdf

    def format_custom_yield_model(self, yield_model):
        """
        Format the custom yield model. 
        :param yield_model:
        :return:
        """
        yield_model_dict = {}
        for field in ["ly", "qy"]:
            self.assert_keys_in_dict(yield_model[field], self.required_keys("yield"))
            yield_model_dict[field] = interp1d(
                yield_model[field]["coordinate_system"], yield_model[field]["map"]
            )
            # assert the yield model is between 0 and 10
            assert np.all(
                (yield_model_dict[field](yield_model[field]["coordinate_system"]) >= 0)
                & (yield_model_dict[field](yield_model[field]["coordinate_system"]) <= 10)
            ), f"Yield model {field} must be between 0 and 10 quanta/keV"
        return yield_model_dict
    
    def arrays_to_grid(self, arrs):
        """
        Convert a list of n 1-dim arrays to an n+1-dim. array, 
        where last dimension denotes coordinate values at point.
        :param arrs: list of 1-dim arrays
        """
        return np.stack(np.meshgrid(*arrs, indexing='ij'), axis=-1)
