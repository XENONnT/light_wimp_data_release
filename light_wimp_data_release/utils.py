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
from inference_interface import multihist_to_template
import yaml
import pickle
from alea.utils import load_yaml
from alea.models import BlueiceExtendedModel


# Real data for SR0 and SR1 unblinding after all selections
with open(importlib.resources.files("light_wimp_data_release.data") / "real_data.pkl", 'rb') as f:
    REAL_DATA = pickle.load(f) 

E_R_MIN = 0.51  # keV
E_R_MAX = 5.0   # keV
SAMPLE_SIZE = int(2e4)
BASIS_PATH = str(
    importlib.resources.files("light_wimp_data_release.data.orthonormal_basis")
)
JSON_PATH = importlib.resources.files("light_wimp_data_release.data") / "signal"
JSON_PATH = Path(JSON_PATH)

TEMPLATE_PATH = str(importlib.resources.files("light_wimp_data_release.data"))

BASIS_E_R = np.concatenate(
    [[0.51, 0.6, 0.7, 0.8, 0.9], np.arange(1, 2, 0.25), np.arange(2, 5.5, 0.5)]
)
MAX_YIELD = 10 # quanta/keV
MIN_YIELD = 1  # quanta/keV
LY_SWEEP = np.arange(MIN_YIELD, MAX_YIELD+1, 1)
QY_SWEEP = np.arange(MIN_YIELD, MAX_YIELD+1, 1)

def make_spectrum(er, dr_der):
    """Make spectrum object from 1d arrays of recoil energies and differential rates.
    :param er: recoil energies in keV.
    :param dr_der: differential rate in events/keV/tonne/year.
    :return: spectrum object with keys: coordinate_system, coordinate_name, map, rate.
    """
    assert np.max(er) > E_R_MIN, "Maximum recoil energy must be greater than 0.51 keV, \
        to overlap with the search ROI."
    assert np.min(er) < E_R_MAX, "Minimum recoil energy must be less than 5 keV, \
        to overlap with the search ROI."
    assert len(er) == len(dr_der), "Length of er and dr_der must be the same."

    # Get nominal rate from the integral of the differential rate
    nominal_rate = np.trapz(dr_der, er)
    # Normalize the differential rate to 1
    dr_der /= nominal_rate
    cdf = np.cumsum(dr_der) / np.sum(dr_der)

    # Return the spectrum object with the following attributes
    spectrum = {}
    spectrum['coordinate_system'] = cdf.tolist()
    spectrum['coordinate_name'] = 'cdf'
    spectrum['map'] = er.tolist()
    spectrum['rate'] = nominal_rate
    spectrum['annotation'] = {
        'coordinate_system': 'Normalized cumulative distribution function for the spectrum',
        'map': 'Recoil energies in keV',
        'rate': 'Total event rate normalization events/tonne/year'
    }

    return spectrum

def get_yield(t, lower, median, upper, y_max=MAX_YIELD, y_min=MIN_YIELD):
    """
    Get yield based on yield parameter tly or tqy:
        yield = median + (upper - median) * t if t >= 0
        yield = median + (lower - median) * t if t < 0    
    Truncate the yield to be within y_max and y_min
    :param t: yield parameter tly or tqy
    :param lower: lower yield model dict with keys: coordinate_system, map (unit: quanta/keV)
    :param median: median yield model dict with keys: coordinate_system, map (unit: quanta/keV)
    :param upper: upper yield model dict with keys: coordinate_system, map (unit: quanta/keV)
    :param y_max: maximum yield value (unit: quanta/keV), default to MAX_YIELD
    :param y_min: minimum yield value (unit: quanta/keV), default to MIN_YIELD
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

def load_default_yield_model():
    """Load default yield model based on YBe calibration median yields.
    :return: yield model dict with keys: 
        ly_lower, ly_median, ly_upper, qy_lower, qy_median, qy_upper.
    """
    ly_lower = load_json(os.path.join(JSON_PATH, "ly_ybe_model_lower.json"))
    ly_median = load_json(os.path.join(JSON_PATH, "ly_ybe_model_median.json"))
    ly_upper = load_json(os.path.join(JSON_PATH, "ly_ybe_model_upper.json"))
    qy_lower = load_json(os.path.join(JSON_PATH, "qy_ybe_model_lower.json"))
    qy_median = load_json(os.path.join(JSON_PATH, "qy_ybe_model_median.json"))
    qy_upper = load_json(os.path.join(JSON_PATH, "qy_ybe_model_upper.json"))
    return {
        "ly_lower": ly_lower,
        "ly_median": ly_median,
        "ly_upper": ly_upper,
        "qy_lower": qy_lower,
        "qy_median": qy_median,
        "qy_upper": qy_upper
    }

def produce_templates(source_name, output_folder, spectrum, yield_model):
    """Produce templates for a specific source with customized spectrum or yield model, and save them to h5 files.
    :param source_name: source name, eg. "wimp_si_5" means 5 GeV WIMP SI. Can be either new physics model or 'b8'.
    :param spectrum: spectrum object with keys: coordinate_system, coordinate_name, map, rate.
    :param yield_model: yield model dict with keys: ly_lower, ly_median, ly_upper, qy_lower, qy_median, qy_upper.
    :param output_folder: output folder to save the templates.
    """
    # Create output folder if it does not exist.
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Define the tly and tqy values.
    tly_values = [-3, -2, -1, 0, 1, 2, 3]
    tqy_values = [-3, -2, -1, 0, 1, 2, 3]
    srs = ['sr0', 'sr1']
    total_iterations = len(tly_values) * len(tqy_values) * len(srs)

    with tqdm(total=total_iterations, desc=f"Producing Templates for {source_name}") as pbar:
        for tly in tly_values:
            for tqy in tqy_values:
                # Get the customized yield model based on tly and tqy.
                _customized_yield_model = dict()
                _customized_yield_model['ly'] = get_yield(
                    tly, yield_model['ly_lower'], yield_model['ly_median'], yield_model['ly_upper']
                )
                _customized_yield_model['qy'] = get_yield(
                    tqy, yield_model['qy_lower'], yield_model['qy_median'], yield_model['qy_upper']
                )
                template = Template().build_template(spectrum, _customized_yield_model)
                # Save the templates to h5 files for both sr0 and sr1.
                for sr in srs:
                    filename = f"template_XENONnT_{sr}_{source_name}_cevns_tly_{tly}.0_tqy_{tqy}.0.h5"
                    multihist_to_template([template[sr]], os.path.join(output_folder, filename), ["template"])
                    pbar.update(1)

def get_statistical_model(config_path, confidence_level):
    """Get the statistical model based on the config file and confidence level.
    :param config_path: path to the config file.
    :param confidence_level: confidence level for the statistical model.
    :return: alea statistical model.
    """
    config = load_yaml(config_path)
    model = BlueiceExtendedModel(
        parameter_definition = config['parameter_definition'], 
        likelihood_config = config['likelihood_config'],
        confidence_level = confidence_level,
        confidence_interval_kind = "central",
        template_path = TEMPLATE_PATH
    )
    model.data = REAL_DATA

    return model

def produce_model_config(
        signal_source_name, signal_folder_name, b8_source_name=None, b8_folder_name=None
    ):
    """Produce alea model config yaml file for a specific signal model and yield model.
    :param signal_source_name: signal source name, eg. "wimp_si_5" means 5 GeV WIMP SI.
    :param signal_folder_name: folder name containing the signal templates. eg. "wimp_si"
    :param b8_source_name: b8 source name, eg. "b8" means B8 neutrino.
    :param b8_folder_name: folder name containing the b8 templates. eg. "b8_linear".
    """
    assert (b8_source_name is None) == (b8_folder_name is None), \
        "b8 source name and template path must be both None if any of them is None, \
            because both None is the only case that you will use the default YBe yield model"
    
    # Load the base model config.
    with open(importlib.resources.files("light_wimp_data_release.data") / "statistical_model_base.yaml", 'r') as file:
        yaml_data = yaml.safe_load(file)
    # Update the template folder.
    yaml_data['likelihood_config']['template_folder'] = TEMPLATE_PATH

    # Loop over the two science runs.
    for sr in [0, 1]:
        # Update the signal and b8 template filenames.
        signal_template_filename = signal_folder_name + "/template_XENONnT_sr%s_"%(sr) + signal_source_name + "_cevns_tly_{t_ly:.1f}_tqy_{t_qy:.1f}.h5"
        yaml_data['likelihood_config']['likelihood_terms'][sr]['sources'][0]['template_filename'] = signal_template_filename
        # Update the b8 template filename if b8_source_name is provided, which means you are overriding the yield model.
        if b8_source_name is not None:
            b8_template_filename = b8_folder_name + "/template_XENONnT_sr%s_"%(sr) + b8_source_name + "_cevns_tly_{t_ly:.1f}_tqy_{t_qy:.1f}.h5"
            yaml_data['likelihood_config']['likelihood_terms'][sr]['sources'][1]['template_filename'] = b8_template_filename

    # Use signal folder name to decide the output file name.
    output_file = str(importlib.resources.files("light_wimp_data_release.data") / f"{signal_folder_name}_statistical_model.yaml")
    with open(output_file, 'w') as file:
        yaml.safe_dump(yaml_data, file)


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
                "coordinate_system",  # pdf or cdf distrubution, normalized to 1
                "coordinate_name",  # pdf or cdf
                "map",  # recoil energies in keV
                "rate",  # total event rate normalization events/tonne/year liquid xenon
            ]
        elif type == "yield":
            return [
                "coordinate_system",  # pdf or cdf distrubution, normalized to 1
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
        er_samples = inverse_cdf(samples)
        roi_mask = (er_samples > E_R_MIN) & (er_samples < E_R_MAX)
        er_samples = er_samples[roi_mask]  # Truncate to recoil energy in the ROI

        template = self.base_templates(sr)
        template.histogram = (
            sum(
                [
                    self.interpolators[sr](
                        [
                            er,
                            np.float64(self.yield_model["ly"](er)),
                            np.float64(self.yield_model["qy"](er)),
                        ]
                    )
                    for er in er_samples
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
        anchors_array = [BASIS_E_R, LY_SWEEP, QY_SWEEP]
        anchors_grid = self.arrays_to_grid(anchors_array)

        extra_dims = [3, 3, 3, 3]  # Templates dimension
        anchor_scores = np.zeros(list(anchors_grid.shape)[:-1] + extra_dims)
        for i_e, e in enumerate(BASIS_E_R):
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
        :param sr: sr0 or sr1
        :return: base template
        """
        template = os.path.join(
            BASIS_PATH,
            f"template_XENONnT_{sr}_recasting_mono_1_0_cevns_tly_7_tqy_7.h5",  # I like the number 7
        )
        return template_to_multihist(template)["template"]

    def default_yield_model(self):
        """
        Return default yield model based on YBe calibration median yields.
        :return:
        """
        ly = load_json(os.path.join(JSON_PATH, f"ly_ybe_model_median.json"))
        qy = load_json(os.path.join(JSON_PATH, f"qy_ybe_model_median.json"))
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
