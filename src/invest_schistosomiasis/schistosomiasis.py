"""NatCap InVEST Plugin for modeling Schistosomiasis risk.

Adapted from Walz et al. 2015 paper
https://journals.plos.org/plosntds/article?id=10.1371/journal.pntd.0004217

Contributors:
Doug Denu, Andrew Chamberlin, Giulio De Leo, Lisa Mandle, Emily Soth,
    Dave Fisher
"""

import logging
import os
import tempfile
import shutil
import subprocess
import json

import numpy
import pygeoprocessing
import pygeoprocessing.kernels
import taskgraph
from osgeo import gdal
from osgeo import osr
from osgeo_utils import gdal2tiles

import matplotlib.pyplot as plt

from natcap.invest import spec
from natcap.invest import utils
from natcap.invest import validation
from natcap.invest.unit_registry import u

gdal.UseExceptions()

LOGGER = logging.getLogger(__name__)

logging.getLogger('taskgraph').setLevel('DEBUG')
# Was seeing a lot of font related logging
# https://stackoverflow.com/questions/56618739/matplotlib-throws-warning-message-because-of-findfont-python
logging.getLogger('matplotlib.font_manager').disabled = True

FLOAT32_NODATA = float(numpy.finfo(numpy.float32).min)
BYTE_NODATA = 255

# Coloramps for styling output tiles for companion notebook
# Blues
POP_RISK = {
    '0%': '247 251 255',
    '20%': '209 226 243',
    '40%': '154 200 224',
    '60%': '82 157 204',
    '80%': '29 108 177',
    '100%': '8 48 107',
    'nv': '1 1 1 0'
}
# Red organge yellow
GENERIC_RISK = {
    '0%': '255 255 178',
    '25%': '254 204 92',
    '50%': '253 141 60',
    '75%': '240 59 32',
    '100%': '189 0 38',
    'nv': '0 0 0 0'
}

# About text for defining a trapezoid function
TRAPEZOID_DEFINITION = (
    "Trapezoid is defined by a line followed by a plateau followed by a second line."
    " All y values before xa have value xa and all y values after xz have value yz.")

# Input definitions for our custom functions
SPEC_FUNC_DEFINITIONS = {
    'linear': {
        "xa": {"type": "number", "about": 
               "First x coordinate of a line defined by two points: (xa, ya), (xz, yz)."},
        "ya": {"type": "number", "about": 
               "First y coordinate of a line defined by two points: (xa, ya), (xz, yz)."},
        "xz": {"type": "number", "about": 
               "Second x coordinate of a line defined by two points: (xa, ya), (xz, yz)."},
        "yz": {"type": "number", "about": 
               "Second y coordinate of a line defined by two points: (xa, ya), (xz, yz)."},
    },
    'trapezoid': {
        "xa": {"type": "number", "about": 
               "First x coordinate of trapezoids first line defined by two points: (xa, ya), (xb, yb). "
               f"{TRAPEZOID_DEFINITION}" },
        "ya": { "type": "number", "about": 
               "First y coordinate of trapezoids first line defined by two points: (xa, ya), (xb, yb). "
               f"{TRAPEZOID_DEFINITION}" },
        "xb": {"type": "number", "about": 
               "Second x coordinate of trapezoids first line defined by two points: (xa, ya), (xb, yb). "
               f"{TRAPEZOID_DEFINITION}" },
        "yb": {"type": "number", "about":
               "Second y coordinate of trapezoids first line defined by two points: (xa, ya), (xb, yb). "
               f"{TRAPEZOID_DEFINITION}" },
        "xc": {"type": "number", "about":
               "First x coordinate of trapezoids second line defined by two points: (xc, yc), (xz, yz). "
               f"{TRAPEZOID_DEFINITION}" },
        "yc": {"type": "number", "about":
               "First y coordinate of trapezoids second line defined by two points: (xc, yc), (xz, yz). "
               f"{TRAPEZOID_DEFINITION}" },
        "xz": {"type": "number", "about":
               "Second x coordinate of trapezoids second line defined by two points: (xc, yc), (xz, yz). "
               f"{TRAPEZOID_DEFINITION}" },
        "yz": {"type": "number", "about":
               "Second y coordinate of trapezoids second line defined by two points: (xc, yc), (xz, yz). "
               f"{TRAPEZOID_DEFINITION}" },
    },
    'gaussian': {
        "mean": {"type": "number", "about": "Distribution mean."},
        "std": {"type": "number", "about": "Standard deviation."},
        "lb": {"type": "number", "about": "Lower boundary."},
        "ub": {"type": "number", "about": "Upper boundary."},
    },
    'scurve': {
        "yin": {"type": "number", "about": "Initial y-intercept value."},
        "yfin": { "type": "number", "about": "Value of y at tail."},
        "xmed": {"type": "number", "about": 
                 "X value where curve transitions."},
        "inv_slope": { "type": "number", "about":
                "Defines the sharpness of the curve."},
    },
    'exponential': {
        "yin": {"type": "number", "about": "Initial y-intercept value."},
        "xmed": { "type": "number", "about":
                "First points y coordinate that defines the line."},
        "decay_factor": {"type": "number", "about": "Determines rate of decay."},
        "max_dist": { "type": "number", "about":
                "x value where y decays to 0."},
    }
}

# Convenient list of function keys
FUNC_KEYS = list(SPEC_FUNC_DEFINITIONS.keys())

# Helper dictionary for organizing the possible custom function inputs for
# each suitability input type. Keys should match repsective MODEL_SPEC 
# suitability type input ids.
FUNC_PARAMS = {
    'rural_population': [
        spec.NumberInput(
            id=f'rural_population_{func_name}_param_{param_name}',
            name=f'{param_name}',
            about=param_desc['about'],
            required=f"default_population_suit == False and rural_population_func_type == '{func_name}'",
            allowed=f"default_population_suit == False and rural_population_func_type == '{func_name}'",
            units=None
        )
        for func_name in FUNC_KEYS for param_name, param_desc in SPEC_FUNC_DEFINITIONS[func_name].items()
    ],
    'urbanization_population': [
        spec.NumberInput(
            id=f'urbanization_population_{func_name}_param_{param_name}',
            name=f'{param_name}',
            about=param_desc['about'],
            required=f"default_population_suit == False and urbanization_population_func_type == '{func_name}'",
            allowed=f"default_population_suit == False and urbanization_population_func_type == '{func_name}'",
            units=None
        )
        for func_name in FUNC_KEYS for param_name, param_desc in SPEC_FUNC_DEFINITIONS[func_name].items()
    ],
    'water_velocity': [
        spec.NumberInput(
            id=f'water_velocity_{func_name}_param_{param_name}',
            name=f'{param_name}',
            about=param_desc['about'],
            required=f"calc_water_velocity and water_velocity_func_type == '{func_name}'",
            allowed=f"calc_water_velocity and water_velocity_func_type == '{func_name}'",
            units=None
        )
        for func_name in FUNC_KEYS for param_name, param_desc in SPEC_FUNC_DEFINITIONS[func_name].items()
    ],
    'snail_water_temp': [
        spec.NumberInput(
            id=f'snail_water_temp_{func_name}_param_{param_name}',
            name=f'{param_name}',
            about=param_desc['about'],
            required=f"calc_temperature and snail_water_temp_func_type == '{func_name}'",
            allowed=f"calc_temperature and snail_water_temp_func_type == '{func_name}'",
            units=None
        )
        for func_name in FUNC_KEYS for param_name, param_desc in SPEC_FUNC_DEFINITIONS[func_name].items()
    ],
    'parasite_water_temp': [
        spec.NumberInput(
            id=f'parasite_water_temp_{func_name}_param_{param_name}',
            name=f'{param_name}',
            about=param_desc['about'],
            required=f"calc_temperature and parasite_water_temp_func_type == '{func_name}'",
            allowed=f"calc_temperature and parasite_water_temp_func_type == '{func_name}'",
            units=None
        )
        for func_name in FUNC_KEYS for param_name, param_desc in SPEC_FUNC_DEFINITIONS[func_name].items()
    ],
    'ndvi': [
        spec.NumberInput(
            id=f'ndvi_{func_name}_param_{param_name}',
            name=f'{param_name}',
            about=param_desc['about'],
            required=f"calc_ndvi and ndvi_func_type == '{func_name}'",
            allowed=f"calc_ndvi and ndvi_func_type == '{func_name}'",
            units=None
        )
        for func_name in FUNC_KEYS for param_name, param_desc in SPEC_FUNC_DEFINITIONS[func_name].items()
    ]
}

def _user_suitability_func_params(input_id):
    """Return a list of spec.NumberInputs.

    Given a string input_id create a list of spec.NumberInputs for the different
    function types in SPEC_FUNC_DEFINITIONS.

    Parameters:
        input_id (str) - unique id to differentiate user defined suitability
                         inputs.

    Returns:
        A list of spec.NumberInput.
    """
    return [
        spec.NumberInput(
            id=f'custom_{input_id}_{func_name}_param_{param_name}',
            name=f'{param_name}',
            about=param_desc['about'],
            required=f"calc_custom_{input_id} and custom_{input_id}_func_type == '{func_name}'",
            allowed=f"calc_custom_{input_id} and custom_{input_id}_func_type == '{func_name}'",
            units=None
        )
        for func_name in FUNC_KEYS for param_name, param_desc in SPEC_FUNC_DEFINITIONS[func_name].items()
    ]

FUNC_PARAMS_USER = _user_suitability_func_params

SUITABILITY_FUNCTION_OPTIONS = [
    spec.Option(key="linear", display_name="linear"),
    spec.Option(key="exponential", display_name="exponential"),
    spec.Option(key="scurve", display_name="scurve"),
    spec.Option(key="trapezoid", display_name="trapezoid"),
    spec.Option(key="gaussian", display_name="gaussian")]

MODEL_SPEC = spec.ModelSpec(
    model_id='schistosomiasis',
    model_title="Schistosomiasis",
    userguide="https://github.com/natcap/invest-schistosomiasis",
    validate_spatial_overlap=True,
    different_projections_ok=False,
    module_name=__name__,
    input_field_order=[
        ['workspace_dir', 'results_suffix'],
        ['aoi_path'],
        ['decay_distance'],
        ["water_presence_path"],
        ["population_count_path", "default_population_suit",
         "rural_population_max", "urbanization_population_max",
         "rural_population_func_type",
         {"Rural parameters": [key.id for key in FUNC_PARAMS['rural_population']]},
         "urbanization_population_func_type",
         {"Urbanization parameters": [key.id for key in FUNC_PARAMS['urbanization_population']]}],
        ["calc_water_depth", "water_depth_weight"],
        ["calc_temperature", "water_temp_dry_path", "water_temp_wet_path",
        "snail_water_temp_dry_weight", "snail_water_temp_wet_weight", "snail_water_temp_func_type", 
         {"Snail temperature parameters": [key.id for key in FUNC_PARAMS['snail_water_temp']]},
        "parasite_water_temp_dry_weight", "parasite_water_temp_wet_weight", "parasite_water_temp_func_type", 
         {"Parasite temperature parameters": [key.id for key in FUNC_PARAMS['parasite_water_temp']]}],
        ["calc_ndvi", "ndvi_func_type",
         "ndvi_dry_path", "ndvi_dry_weight",
         "ndvi_wet_path", "ndvi_wet_weight",
         {"NDVI parameters": [key.id for key in FUNC_PARAMS['ndvi']]}],
        ["calc_water_velocity", "water_velocity_func_type",
         "dem_path", "water_velocity_weight",
         {"Water velocity parameters": [key.id for key in FUNC_PARAMS['water_velocity']]}],
        ["calc_custom_one", "custom_one_func_type",
         "custom_one_path", "custom_one_weight",
         {"Input parameters": [key.id for key in FUNC_PARAMS_USER('one')]}],
        ["calc_custom_two", "custom_two_func_type",
         "custom_two_path", "custom_two_weight",
         {"Input parameters": [key.id for key in FUNC_PARAMS_USER('two')]}],
        ["calc_custom_three", "custom_three_func_type",
         "custom_three_path", "custom_three_weight",
         {"Input parameters": [key.id for key in FUNC_PARAMS_USER('three')]}],
    ],
    inputs=[
        spec.WORKSPACE,
        spec.SUFFIX,
        spec.N_WORKERS,
        spec.NumberInput(
            id="decay_distance",
            name="max decay distance",
            about="Maximum threat distance from water risk.",
            units=u.meter,
        ),
        spec.AOI,
        spec.SingleBandRasterInput(
            id='population_count_path',
            name='population raster',
            about="A raster representing the number of inhabitants per pixel.",
            data_type=float,
            units=u.meter,
        ),
        spec.BooleanInput(
            id="default_population_suit",
            name="Use default poppulation suitability.",
            about=("Linear increase in risk to rural max and an s-curve"
                   "decrease in risk from rural max to urbanization max."),
            required=False,
        ),
        spec.NumberInput(
            id="rural_population_max",
            name="Rural population max",
            about="The rural population at which risk is highest.",
            required="default_population_suit",
            allowed="default_population_suit",
            units=None,
        ),
        spec.NumberInput(
            id="urbanization_population_max",
            name="Urbanization population max",
            about="The urbanization population at which risk is 0.",
            required="default_population_suit",
            allowed="default_population_suit",
            units=None,
        ),
        *FUNC_PARAMS['rural_population'],
        spec.OptionStringInput(
            id="rural_population_func_type",
            name="Rural Suitability function type",
            about="The function type to apply to the suitability factor.",
            required="default_population_suit == False",
            allowed="default_population_suit == False",
            options=[
                spec.Option(key="linear", display_name="Linear"),
                spec.Option(key="exponential", display_name="exponential"),
                spec.Option(key="scurve", display_name="scurve"),
                spec.Option(key="trapezoid", display_name="trapezoid"),
                spec.Option(key="gaussian", display_name="gaussian"),
            ]),
        *FUNC_PARAMS['urbanization_population'],
        spec.OptionStringInput(
            id="urbanization_population_func_type",
            name="Urbanization Suitability function type",
            about="The function type to apply to the suitability factor.",
            required="False",
            allowed="default_population_suit == False",
            options=[
                spec.Option(key="None", display_name="[Select option]"),
                spec.Option(key="linear", display_name="Linear"),
                spec.Option(key="exponential", display_name="exponential"),
                spec.Option(key="scurve", display_name="scurve"),
                spec.Option(key="trapezoid", display_name="trapezoid"),
                spec.Option(key="gaussian", display_name="gaussian"),
            ]),
        spec.BooleanInput(
            id="calc_water_depth",
            name="calculate water depth",
            about=("Calculate water depth. Using the water presence raster"
                   " input, uses a water distance from shore as a proxy for"
                   " depth."),
            required=False
        ),
        spec.RatioInput(
            id="water_depth_weight",
            name="water depth risk weight",
            about="The weight this factor should have on overall risk.",
            required="calc_water_depth",
            allowed="calc_water_depth"
        ),
        spec.SingleBandRasterInput(
            id="water_presence_path",
            name='water presence',
            about="A raster indicating presence of water.",
            data_type=int,
            units=None,
        ),
        spec.BooleanInput(
            id="calc_water_velocity",
            name="calculate water velocity",
            about="Calculate water velocity.",
            required=False
        ),
        spec.OptionStringInput(
            id="water_velocity_func_type",
            name="Suitability function type",
            about="The function type to apply to the suitability factor.",
            required="calc_water_velocity",
            allowed="calc_water_velocity",
            options=[
                spec.Option(key="default", display_name="Default used in paper."),
                *SUITABILITY_FUNCTION_OPTIONS]
        ),
        *FUNC_PARAMS['water_velocity'],
        spec.DEM.model_copy(update=dict(
            required="calc_water_velocity",
            allowed="calc_water_velocity")
        ),
        spec.RatioInput(
            id="water_velocity_weight",
            about="The weight this factor should have on overall risk.",
            name="water velocity risk weight",
            required="calc_water_velocity",
            allowed="calc_water_velocity"
        ),
        spec.BooleanInput(
            id="calc_temperature",
            about="Calculate water temperature.",
            name="calculate water temperature",
            required=False
        ),
        spec.SingleBandRasterInput(
            id='water_temp_dry_path',
            name='dry season temperature raster',
            data_type=float,
            units=u.celsius,
            projected=True,
            projection_units=u.meter,
            about="A raster representing the water temp for dry season.",
            required="calc_temperature",
            allowed="calc_temperature"
        ),
        spec.SingleBandRasterInput(
            id='water_temp_wet_path',
            name='wet season temperature raster',
            data_type=float,
            units=u.celsius,
            projected=True,
            projection_units=u.meter,
            about="A raster representing the water temp for wet season.",
            required="calc_temperature",
            allowed="calc_temperature"
        ),
        spec.OptionStringInput(
            id=f'snail_water_temp_func_type',
            name="Snail suitability function type",
            about="The function type to apply to the suitability factor.",
            required="calc_temperature",
            allowed="calc_temperature",
            options=[
                spec.Option(key="bt", display_name="Default: Bulinus truncatus."),
                spec.Option(key="bg", display_name="Default: Biomphalaria."),
                *SUITABILITY_FUNCTION_OPTIONS]
        ),
        *FUNC_PARAMS['snail_water_temp'],
        spec.OptionStringInput(
            id=f'parasite_water_temp_func_type',
            name="Parasite suitability function type",
            about="The function type to apply to the suitability factor.",
            required="calc_temperature",
            allowed="calc_temperature",
            options=[
                spec.Option(key="sh", display_name="Default: S. haematobium."),
                spec.Option(key="sm", display_name="Default: S. mansoni."),
                *SUITABILITY_FUNCTION_OPTIONS]
        ),
        *FUNC_PARAMS['parasite_water_temp'],
        spec.RatioInput(
            id="snail_water_temp_dry_weight",
            about="The weight this factor should have on overall risk.",
            name="snail water temp dry risk weight",
            required="calc_temperature",
            allowed="calc_temperature",
        ),
        spec.RatioInput(
            id="snail_water_temp_wet_weight",
            about="The weight this factor should have on overall risk.",
            name="snail water temp wet risk weight",
            required="calc_temperature",
            allowed="calc_temperature",
        ),
        spec.RatioInput(
            id="parasite_water_temp_dry_weight",
            about="The weight this factor should have on overall risk.",
            name="parasite water temp dry risk weight",
            required="calc_temperature",
            allowed="calc_temperature",
        ),
        spec.RatioInput(
            id="parasite_water_temp_wet_weight",
            about="The weight this factor should have on overall risk.",
            name="parasite water temp wet risk weight",
            required="calc_temperature",
            allowed="calc_temperature",
        ),
        spec.BooleanInput(
            id="calc_ndvi",
            about="Calculate NDVI.",
            name="calculate NDVI",
            required=False
        ),
        spec.OptionStringInput(
            id="ndvi_func_type",
            name="Suitability function type",
            about="The function type to apply to the suitability factor.",
            required="calc_ndvi",
            allowed="calc_ndvi",
            options=[
                spec.Option(key="default", display_name="Default used in paper."),
                *SUITABILITY_FUNCTION_OPTIONS]
        ),
        *FUNC_PARAMS['ndvi'],
        spec.SingleBandRasterInput(
            id="ndvi_dry_path",
            name='ndvi dry raster',
            units=None,
            projected=True,
            projection_units=u.meter,
            about= "A raster representing the ndvi for dry season.",
            required="calc_ndvi",
            allowed="calc_ndvi"
        ),
        spec.RatioInput(
            id="ndvi_dry_weight",
            about="The weight this factor should have on overall risk.",
            name="ndvi dry risk weight",
            required="calc_ndvi",
            allowed="calc_ndvi"
        ),
        spec.SingleBandRasterInput(
            id="ndvi_wet_path",
            name='ndvi wet raster',
            units=None,
            projected=True,
            projection_units=u.meter,
            about="A raster representing the ndvi for wet season.",
            required="calc_ndvi",
            allowed="calc_ndvi"
        ),
        spec.RatioInput(
            id="ndvi_wet_weight",
            about="The weight this factor should have on overall risk.",
            name="ndvi wet risk weight",
            required="calc_ndvi",
            allowed="calc_ndvi"
        ),
        spec.BooleanInput(
            id="calc_custom_one",
            required=False,
            about="User defined suitability function.",
            name="Additional user defined suitability input."
        ),
        spec.OptionStringInput(
            id="custom_one_func_type",
            name="Suitability function type",
            about="The function type to apply to the suitability factor.",
            required="calc_custom_one",
            allowed="calc_custom_one",
            options=SUITABILITY_FUNCTION_OPTIONS),
        *FUNC_PARAMS_USER('one'),
        spec.SingleBandRasterInput(
            id='custom_one_path',
            name='custom raster',
            units=None,
            projected=True,
            projection_units=u.meter,
            about="A raster representing the user suitability.",
            required="calc_custom_one",
            allowed="calc_custom_one"
        ),
        spec.RatioInput(
            id="custom_one_weight",
            about="The weight this factor should have on overall risk.",
            name="User risk weight",
            required="calc_custom_one",
            allowed="calc_custom_one"
        ),
        spec.BooleanInput(
            id="calc_custom_two",
            required=False,
            about="User defined suitability function.",
            name="Additional user defined suitability input."
        ),
        spec.OptionStringInput(
            id="custom_two_func_type",
            name="Suitability function type",
            about="The function type to apply to the suitability factor.",
            required="calc_custom_two",
            allowed="calc_custom_two",
            options=SUITABILITY_FUNCTION_OPTIONS),
        *FUNC_PARAMS_USER('two'),
        spec.SingleBandRasterInput(
            id='custom_two_path',
            name='custom raster',
            units=None,
            projected=True,
            projection_units=u.meter,
            about="A raster representing the user suitability.",
            required="calc_custom_two",
            allowed="calc_custom_two"
        ),
        spec.RatioInput(
            id="custom_two_weight",
            about="The weight this factor should have on overall risk.",
            name="User risk weight",
            required="calc_custom_two",
            allowed="calc_custom_two"
        ),
        spec.BooleanInput(
            id="calc_custom_three",
            required=False,
            about="User defined suitability function.",
            name="Additional user defined suitability input."
        ),
        spec.OptionStringInput(
            id="custom_three_func_type",
            name="Suitability function type",
            about="The function type to apply to the suitability factor.",
            required="calc_custom_three",
            allowed="calc_custom_three",
            options=SUITABILITY_FUNCTION_OPTIONS),
        *FUNC_PARAMS_USER('three'),
        spec.SingleBandRasterInput(
            id='custom_three_path',
            name='custom raster',
            units=None,
            projected=True,
            projection_units=u.meter,
            about="A raster representing the user suitability.",
            required="calc_custom_three",
            allowed="calc_custom_three"
        ),
        spec.RatioInput(
            id="custom_three_weight",
            about="The weight this factor should have on overall risk.",
            name="User risk weight",
            required="calc_custom_three",
            allowed="calc_custom_three"
        ),
    ],
    outputs=[
        spec.SingleBandRasterOutput(
            id='snail_water_temp_suit_wet',
            path='snail_water_temp_suit_wet.tif',
            about="",
            data_typ=float,
            units=None
            ),
        spec.SingleBandRasterOutput(
            id='parasite_water_temp_suit_dry',
            path='parasite_water_temp_suit_dry.tif',
            about="",
            data_type=float,
            units=None
            ),
        spec.SingleBandRasterOutput(
            id='parasite_water_temp_suit_wet',
            path='parasite_water_temp_suit_wet.tif',
            about="",
            data_type=float,
            units=None
            ),
        spec.SingleBandRasterOutput(
            id='ndvi_suit_dry',
            path='ndvi_suit_dry.tif',
            about="",
            data_type=float,
            units=None
            ),
        spec.SingleBandRasterOutput(
            id='ndvi_suit_wet',
            path='ndvi_suit_wet.tif',
            about="",
            data_type=float,
            units=None
            ),
        spec.SingleBandRasterOutput(
            id='water_velocity_suit',
            path='water_velocity_suit.tif',
            about="",
            data_type=float,
            units=None
            ),
        spec.SingleBandRasterOutput(
            id='water_depth_suit',
            path='water_depth_suit.tif',
            about="",
            data_type=float,
            units=None
            ),
        spec.SingleBandRasterOutput(
            id='population_suitability',
            path='population_suitability.tif',
            about="",
            data_type=float,
            units=None
            ),
        spec.SingleBandRasterOutput(
            id='habitat_stability_suit',
            path='habitat_stability_suit.tif',
            about="",
            data_type=float,
            units=None
            ),
        spec.SingleBandRasterOutput(
            id='habitat_suit_weighted_mean',
            path='habitat_suit_weighted_mean.tif',
            about="",
            data_type=float,
            units=None
            ),
        spec.SingleBandRasterOutput(
            id='custom_suit_one',
            path='custom_suit_one.tif',
            about="",
            data_type=float,
            units=None
            ),
        spec.SingleBandRasterOutput(
            id='custom_suit_two',
            path='custom_suit_two.tif',
            about="",
            data_type=float,
            units=None
            ),
        spec.SingleBandRasterOutput(
            id='custom_suit_three',
            path='custom_suit_three.tif',
            about="",
            data_type=float,
            units=None
            ),
        spec.SingleBandRasterOutput(
            id='normalized_convolved_risk',
            path='normalized_convolved_risk.tif',
            about="",
            data_type=float,
            units=None
            ),
        spec.SingleBandRasterOutput(
            id='population_suit_sqkm',
            path='population_suit_sqkm.tif',
            about="",
            data_type=float,
            units=None
            ),
        spec.SingleBandRasterOutput(
            id='snail_water_temp_suit_dry',
            path='snail_water_temp_suit_dry.tif',
            about=(
                "Suitability risk."
            ),
            data_type=float,
            units=None
        ),
        spec.TASKGRAPH_CACHE,
        spec.SingleBandRasterOutput(
            id='aligned_pop_count',
            path='intermediate/aligned_population_count.tif',
            about="",
            data_type=float,
            units=None
            ),
        spec.SingleBandRasterOutput(
            id='aligned_pop_density',
            path='intermediate/aligned_pop_density.tif',
            about="",
            data_type=float,
            units=None
            ),
        spec.SingleBandRasterOutput(
            id='aligned_water_temp_dry',
            path='intermediate/aligned_water_temp_dry.tif',
            about="",
            data_type=float,
            units=None
            ),
        spec.SingleBandRasterOutput(
            id='aligned_water_temp_wet',
            path='intermediate/aligned_water_temp_wet.tif',
            about="",
            data_type=float,
            units=None
            ),
        spec.SingleBandRasterOutput(
            id='aligned_ndvi_dry',
            path='intermediate/aligned_ndvi_dry.tif',
            about="",
            data_type=float,
            units=None
            ),
        spec.SingleBandRasterOutput(
            id='aligned_ndvi_wet',
            path='intermediate/aligned_ndvi_wet.tif',
            about="",
            data_type=float,
            units=None
            ),
        spec.SingleBandRasterOutput(
            id='aligned_mask',
            path='intermediate/aligned_valid_pixels_mask.tif',
            about="",
            data_type=float,
            units=None
            ),
        spec.SingleBandRasterOutput(
            id='aligned_dem',
            path='intermediate/aligned_dem.tif',
            about="",
            data_type=float,
            units=None
            ),
        spec.SingleBandRasterOutput(
            id='aligned_water_presence',
            path='intermediate/aligned_water_presence.tif',
            about="",
            data_type=float,
            units=None
            ),
        spec.SingleBandRasterOutput(
            id='aligned_lulc',
            path='intermediate/aligned_lulc.tif',
            about="",
            data_type=float,
            units=None
            ),
        spec.SingleBandRasterOutput(
            id='aligned_custom_one',
            path='intermediate/aligned_custom_one.tif',
            about="",
            data_type=float,
            units=None
            ),
        spec.SingleBandRasterOutput(
            id='aligned_custom_two',
            path='intermediate/aligned_custom_two.tif',
            about="",
            data_type=float,
            units=None
            ),
        spec.SingleBandRasterOutput(
            id='aligned_custom_three',
            path='intermediate/aligned_custom_three.tif',
            about="",
            data_type=float,
            units=None
            ),
        spec.SingleBandRasterOutput(
            id='masked_population',
            path='intermediate/masked_population.tif',
            about="",
            data_type=float,
            units=None
            ),
        spec.SingleBandRasterOutput(
            id='slope',
            path='intermediate/slope.tif',
            about="",
            data_type=float,
            units=None
            ),
        spec.SingleBandRasterOutput(
            id='masked_lulc',
            path='intermediate/masked_lulc.tif',
            about="",
            data_type=float,
            units=None
            ),
        spec.VectorOutput(
            id='reprojected_admin_boundaries',
            path='reprojected_admin_boundaries.gpkg',
            about="",
            fields=[],
            geometry_types=["POLYGON", "MULTIPOLYGON"],
            ),
        spec.SingleBandRasterOutput(
            id='distance',
            path='intermediate/distance.tif',
            about="",
            data_type=float,
            units=None
            ),
        spec.SingleBandRasterOutput(
            id='inverse_water_mask',
            path='intermediate/inverse_water_mask.tif',
            about="",
            data_type=float,
            units=None
            ),
        spec.SingleBandRasterOutput(
            id='distance_from_shore',
            path='intermediate/distance_from_shore.tif',
            about="",
            data_type=float,
            units=None
            ),
        spec.FileOutput(
            id='water_velocity_suit_plot',
            path='water_vel_suit_plot.png',
            ),
        spec.FileOutput(
            id='water_temp_suit_dry_plot',
            path='water_temp_suit_dry_plot.png',
            ),
        spec.SingleBandRasterOutput(
            id='unmasked_water_depth_suit',
            path='intermediate/unmasked_water_depth_suit.tif',
            about="",
            data_type=float,
            units=None
            ),
        spec.FileOutput(
            id='water_depth_suit_plot',
            path='water_depth_suit_plot.png',
            ),
        spec.SingleBandRasterOutput(
            id='rural_population_suit',
            path='intermediate/rural_population_suit.tif',
            about="",
            data_type=float,
            units=None
            ),
        spec.SingleBandRasterOutput(
            id='urbanization_population_suit',
            path='intermediate/urbanization_population_suit.tif',
            about="",
            data_type=float,
            units=None
            ),
        spec.SingleBandRasterOutput(
            id='kernel',
            path='intermediate/kernel.tif',
            about="",
            data_type=float,
            units=None
            ),
        spec.SingleBandRasterOutput(
            id='unmasked_convolved_hab_risk',
            path='unmasked_convolved_hab_risk.tif',
            about="",
            data_type=float,
            units=None
            ),
        spec.SingleBandRasterOutput(
            id='convolved_hab_risk',
            path='convolved_hab_risk.tif',
            about="",
            data_type=float,
            units=None
            ),
        spec.SingleBandRasterOutput(
            id='risk_to_pop_abs',
            path='risk_to_pop_abs.tif',
            about="",
            data_type=float,
            units=None
            ),
        spec.SingleBandRasterOutput(
            id='risk_to_pop_rel',
            path='risk_to_pop_rel.tif',
            about="",
            data_type=float,
            units=None
            ),
        spec.SingleBandRasterOutput(
            id='risk_to_pop_count_abs',
            path='risk_to_pop_count_abs.tif',
            about="",
            data_type=float,
            units=None
            ),
        spec.SingleBandRasterOutput(
            id='risk_to_pop_count_rel',
            path='risk_to_pop_count_rel.tif',
            about="",
            data_type=float,
            units=None
            ),
        spec.VectorOutput(
            id='aoi_geojson',
            path='aoi_geojson.geojson',
            about="",
            fields=[],
            geometry_types=["POLYGON", "MULTIPOLYGON"],
            units=None
            ),
        spec.FileOutput(
            id='[SUIT_KEY]_[FUNC_NAME]',
            path='plot-previews/[SUIT_KEY]_[FUNC_NAME].png',
            ),
        spec.FileOutput(
            id='custom_suit_one_plot',
            path='custom_suit_one_plot.png',
            ),
        spec.FileOutput(
            id='custom_suit_two_plot',
            path='custom_suit_two_plot.png',
            ),
        spec.FileOutput(
            id='custom_suit_three_plot',
            path='custom_suit_three_plot.png',
            ),
        spec.FileOutput(
            id='generic_risk_style',
            path='color-profiles/generic_risk_style.txt',
            ),
        spec.FileOutput(
            id='generic_pop_risk_style',
            path='color-profiles/generic_pop_risk_style.txt',
            ),
        spec.FileOutput(
            id='nb-json-config',
            path='nb-json-config.json',
            ),
    ]
)


def execute(args):
    """Schistosomiasis.
        
        The user can define a function for each suitability metric:
        'rural_population', 'urbanization_population', 'water_velocity', 
        'snail_water_temp', 'parasite_water_temp', 'ndvi', and
        'custom_[one|two|three]'. The functions availabe are 'linear', 
        'exponential', 'scurve', 'guassian', 'trapezoid'. Each of these
        functions are defined using a set of input parameters. For docstring
        brevity 'rural_population' is used as an example of how the function
        parameters are expected in 'args' for each function type.

        args['rural_population_linear_param_xa'] (number): First x coordinate
            of a line defined by two points: (xa, ya), (xz, yz).
        args['rural_population_linear_param_ya'] (number): First y coordinate
            of a line defined by two points: (xa, ya), (xz, yz).
        args['rural_population_linear_param_xz'] (number): Second x
            coordinate of a line defined by two points: (xa, ya), (xz, yz).
        args['rural_population_linear_param_yz'] (number): Second y coordinate
            of a line defined by two points: (xa, ya), (xz, yz).
        args['rural_population_trapezoid_param_xa'] (number): First x
            coordinate of trapezoids first line defined by two points:
            (xa, ya), (xb, yb). Trapezoid is defined by a line followed by a
            plateau followed by a second line. All y values before xa have value
            xa and all y values after xz have value yz.
        args['rural_population_trapezoid_param_ya'] (number): First y coordinate
            of trapezoids first line defined by two points: (xa, ya), (xb, yb).
        args['rural_population_trapezoid_param_xb'] (number): Second x
            coordinate of trapezoids first line defined by two points:
            (xa, ya), (xb, yb).
        args['rural_population_trapezoid_param_yb'] (number): Second y
            coordinate of trapezoids first line defined by two points:
            (xa, ya), (xb, yb). 
        args['rural_population_trapezoid_param_xc'] (number): First x coordinate
            of trapezoids second line defined by two points: (xc, yc), (xz, yz).
        args['rural_population_trapezoid_param_yc'] (number): First y coordinate
            of trapezoids second line defined by two points: (xc, yc), (xz, yz).
        args['rural_population_trapezoid_param_xz'] (number): Second x
            coordinate of trapezoids second line defined by two points:
            (xc, yc), (xz, yz).
        args['rural_population_trapezoid_param_yz'] (number): Second y
            coordinate of trapezoids second line defined by two points:
            (xc, yc), (xz, yz).
        args['rural_population_gaussian_param_mean'] (number): Distribution mean.
        args['rural_population_gaussian_param_std'] (number): Standard deviation.
        args['rural_population_gaussian_param_lb'] (number): Lower boundary.
        args['rural_population_gaussian_param_ub'] (number): Upper boundary.
        args['rural_population_scurve_param_yin'] (number): Initial y-intercept value.
        args['rural_population_scurve_param_yfin'] (number): Value of y at tail.
        args['rural_population_scurve_param_xmed'] (number): X value where
            curve transitions.
        args['rural_population_scurve_param_inv_slope'] (number): Defines the
            sharpness of the curve.
        args['rural_population_exponential_param_yin'] (number): Initial
            y-intercept value.
        args['rural_population_exponential_param_xmed'] (number): First points y
            coordinate that defines the line.
        args['rural_population_exponential_param_decay_factor'] (number):
            Determines rate of decay.

    Args:
        args['workspace_dir'] (string): The folder where all the model's output
            files will be written. If this folder does not exist, it will
            be created. If data already exists in the folder, it will be
            overwritten.
        args['results_suffix'] (string): Suffix that will be appended to
            all output file names. Useful to differentiate between model runs.
        args['n_workers'] (number): The n_workers parameter to provide to
            taskgraph. -1 will cause all jobs to run synchronously.
            0 will run all jobs in the same process, but scheduling will take
            place asynchronously. Any other positive integer will cause that
            many processes to be spawned to execute tasks.
        args['decay_distance'] (number): Maximum threat distance from water
            risk.
        args['aoi_path'] (string): A map of areas over which to aggregate and
            summarize the final results.
        args['population_count_path'] (string): A raster representing the
            number of inhabitants per pixel.
        args['default_population_suit'] (boolean): Linear increase in risk
            to rural max and an s-curvedecrease in risk from rural max to
            urbanization max.
        args['rural_population_max'] (number): The rural population at which
            risk is highest.
        args['urbanization_population_max'] (number): The urbanization
            population at which risk is 0.
        args['rural_population_func_type'] (string): The function type
            to apply to the suitability factor.
        args['urbanization_population_func_type'] (string): The function
            type to apply to the suitability factor.
        args['calc_water_depth'] (boolean): Calculate water depth. Using the
            water presence raster input, uses a water distance from shore as a
            proxy for depth.
        args['water_depth_weight'] (float): The weight this factor should have
            on overall risk.
        args['water_presence_path'] (string): A raster indicating presence of
        `water.
        args['calc_water_velocity'] (boolean): Calculate water velocity.
        args['water_velocity_func_type'] (string): The function type to
            apply to the suitability factor.
        args['dem_path'] (string): Map of elevation above sea level.
        args['water_velocity_weight'] (float): The weight this factor should
            have on overall risk.
        args['calc_temperature'] (boolean): Calculate water temperature.
        args['water_temp_dry_path'] (string): A raster representing the water
            temp for dry season.
        args['water_temp_wet_path'] (string): A raster representing the water
            temp for wet season.
        args['snail_water_temp_func_type'] (string): The function type
            to apply to the suitability factor.
        args['parasite_water_temp_func_type'] (string): The function
            type to apply to the suitability factor.
        args['snail_water_temp_dry_weight'] (float): The weight this factor
            should have on overall risk.
        args['snail_water_temp_wet_weight'] (float): The weight this factor
            should have on overall risk.
        args['parasite_water_temp_dry_weight'] (float): The weight this factor
            should have on overall risk.
        args['parasite_water_temp_wet_weight'] (float): The weight this factor
            should have on overall risk.
        args['calc_ndvi'] (boolean): Calculate NDVI.
        args['ndvi_func_type'] (string): The function type to apply to
            the suitability factor.
        args['ndvi_dry_path'] (string): A raster representing the ndvi for dry season.
        args['ndvi_dry_weight'] (float): The weight this factor should have on overall risk.
        args['ndvi_wet_path'] (string): A raster representing the ndvi for wet season.
        args['ndvi_wet_weight'] (float): The weight this factor should have on overall risk.
        args['calc_custom_one'] (boolean): User defined suitability function.
        args['custom_one_func_type'] (string): The function type to
            apply to the suitability factor.
        args['custom_one_path'] (string): A raster representing the user suitability.
        args['custom_one_weight'] (float): The weight this factor should have
            on overall risk.
        args['calc_custom_two'] (boolean): User defined suitability function.
        args['custom_two_func_type'] (string): The function type to apply
            to the suitability factor.
        args['custom_two_path'] (string): A raster representing the user suitability.
        args['custom_two_weight'] (float): The weight this factor should have on
            overall risk.
        args['calc_custom_three'] (boolean): User defined suitability function.
        args['custom_three_func_type'] (string): The function type to apply
            to the suitability factor.
        args['custom_three_path'] (string): A raster representing the user suitability.
        args['custom_three_weight'] (float): The weight this factor should have on overall risk.

        Returns:
            File registry dictionary mapping MODEL_SPEC output ids to absolute paths
    """
    LOGGER.info(f"Execute {MODEL_SPEC.model_title}.")
    # Preprocess arguments, create a file registry based on MODEL_SPEC.outputs,
    # and initiate a TaskGraph object for tasks.
    args, file_registry, graph = MODEL_SPEC.setup(args)

    FUNC_TYPES = {
        'trapezoid': _trapezoid_op,
        'linear': _linear_op,
        'exponential': _exponential_decay_op,
        's-curve': _sshape_op,
        'gaussian': _gaussian_op,
        }
    DEFAULT_FUNC_TYPES = {
        'temperature': _water_temp_suit,
        'ndvi': _ndvi,
        'default_population_suit': _population_curve_people_per_sqkm,
        'water_velocity': _water_velocity,
        'water_depth': _water_depth_suit,
        }

    # Write color profiles to text file
    default_color_path = file_registry['generic_risk_style']
    pop_color_path = file_registry['generic_pop_risk_style']
    color_path_list = [default_color_path, pop_color_path]
    for color_profile, profile_path in zip(
            [GENERIC_RISK, POP_RISK], color_path_list):
        with open(profile_path, 'w') as f:
            for break_key, rgb_val in color_profile.items():
                f.write(break_key + ' ' + rgb_val + '\n')

    # Set up dictionary to capture parameters necessary for Jupyter Notebook
    # companion as JSON.
    nb_json_config_path = file_registry['nb-json-config']
    nb_json_config = {}

    # Dictionary mapping function and parameters to suitability input.
    suit_func_to_use = {}
    
    # TODO: determine whether to display population, urbanization, or 
    # something else.
    # Mapping of which suitability factors were selected so we know what to
    # operate on.
    suitability_keys = [
        ('ndvi', args['calc_ndvi']),
        ('default_population_suit', args['default_population_suit']),
        ('rural_population', not args['default_population_suit']),
        ('urbanization_population', (
            not args['default_population_suit'] and 
            (args['urbanization_population_func_type'] != 'None'))),
        ('water_velocity', args['calc_water_velocity']),
        ('water_depth', args['calc_water_depth']),
        ('custom_one', args['calc_custom_one']),
        ('custom_two', args['calc_custom_two']),
        ('custom_three', args['calc_custom_three'])]
    # Read chosen function parameters
    for suit_key, calc_suit in suitability_keys:
        # Skip non selected suitability metrics
        if not calc_suit:
            continue
        # Default population suitability and water depth have static functions
        if suit_key in ['default_population_suit', 'water_depth']:
            func_type = 'default'
        else:
            func_type = args[f'{suit_key}_func_type']
        if func_type != 'default':
            func_params = {}
            for key in SPEC_FUNC_DEFINITIONS[func_type].keys():
                LOGGER.info(f'{suit_key}_{func_type}_param_{key}')
                func_params[key] = float(args[f'{suit_key}_{func_type}_param_{key}'])
            user_func = FUNC_TYPES[func_type]
        else:
            func_params = None
            if suit_key == 'default_population_suit':
                func_params = {
                    'rural_population_max': float(args['rural_population_max']),
                    'urbanization_population_max': float(args['urbanization_population_max'])
                    }
            user_func = DEFAULT_FUNC_TYPES[suit_key]

        suit_func_to_use[suit_key] = {
            'func_name':user_func,
            'func_params':func_params
        }
    
    # Handle Temperature separately because of snail, parasite pairing
    temperature_suit_keys = ['snail_water_temp', 'parasite_water_temp']
    # Skip if temperature is not selected
    if args['calc_temperature']:
        for suit_key in temperature_suit_keys:
            func_type = args[f'{suit_key}_func_type']
            if func_type in ['sh', 'sm', 'bg', 'bt']:
                func_params = {'op_key': func_type}
                user_func = DEFAULT_FUNC_TYPES['temperature']
            else:
                func_params = {}
                for key in SPEC_FUNC_DEFINITIONS[func_type].keys():
                    func_params[key] = float(args[f'{suit_key}_{func_type}_param_{key}'])
                user_func = FUNC_TYPES[func_type]

            suit_func_to_use[suit_key] = {
                'func_name':user_func,
                'func_params':func_params,
            }

    ### Align and set up datasets
    # Use the water presence raster for resolution and aligning
    squared_default_pixel_size = _square_off_pixels(
        args['water_presence_path'])

    # Build up a list of provided optional rasters to align
    raster_input_list = [args['water_presence_path']]
    aligned_input_list = [file_registry['aligned_water_presence']]
    conditional_list = [
        (args['calc_temperature'], ['water_temp_dry_path', 'water_temp_wet_path']),
        (args['calc_ndvi'], ['ndvi_dry_path', 'ndvi_wet_path']),
        (args['calc_water_velocity'], ['dem_path']),
        (args['calc_custom_one'], ['custom_one_path']),
        (args['calc_custom_two'], ['custom_two_path']),
        (args['calc_custom_three'], ['custom_three_path']),
    ]
    for conditional, key_list in conditional_list:
        if conditional:
            temporary_paths = [args[path_key] for path_key in key_list]
            raster_input_list += temporary_paths
            temporary_align_paths = [
                file_registry[f'aligned_{path_key[:-5]}'] for path_key in key_list]
            aligned_input_list += temporary_align_paths 

    align_task = graph.add_task(
        pygeoprocessing.align_and_resize_raster_stack,
        kwargs={
            'base_raster_path_list': raster_input_list,
            'base_vector_path_list': [args['aoi_path']],
            'target_raster_path_list': aligned_input_list,
            'resample_method_list': ['near']*len(raster_input_list),
            'target_pixel_size': squared_default_pixel_size,
            'bounding_box_mode': 'intersection',
        },
        target_path_list=aligned_input_list,
        task_name='Align and resize input rasters'
    )
    align_task.join()

    raster_info = pygeoprocessing.get_raster_info(
        file_registry['aligned_water_presence'])
    default_bb = raster_info['bounding_box']
    default_wkt = raster_info['projection_wkt']
    default_pixel_size = raster_info['pixel_size']

    # NOTE: Need to handle population differently in case of scaling
    # Returns population count aligned to other inputs
    population_align_task = graph.add_task(
        _resample_population_raster,
        kwargs={
            'source_population_raster_path': args['population_count_path'],
            'target_pop_count_raster_path': file_registry['aligned_pop_count'],
            'target_pop_density_raster_path': file_registry['aligned_pop_density'],
            'lulc_pixel_size': squared_default_pixel_size,
            'lulc_bb': default_bb,
            'lulc_projection_wkt': default_wkt,
            'working_dir': args['workspace_dir'],
        },
        target_path_list=[
            file_registry['aligned_pop_count'],
            file_registry['aligned_pop_density']],
        task_name='Align and resize population'
    )

    ### Production functions ###
    suitability_tasks = []
    habitat_suit_risk_paths = []
    habitat_suit_risk_weights = []
    outputs_to_tile = []

    ### Population suitability risk
    # Population count to density in square km
    population_suit_sqkm_task = graph.add_task(
        func=_population_count_to_square_km,
        kwargs={
            'population_count_path': file_registry['aligned_pop_count'],
            'target_raster_path': file_registry['population_suit_sqkm'],
        },
        target_path_list=[file_registry['population_suit_sqkm']],
        dependent_task_list=[population_align_task],
        task_name=f'Population count to density in sqkm.')
    outputs_to_tile.append((file_registry['population_suit_sqkm'], default_color_path))

    if args['default_population_suit']:
        population_suitability_path = file_registry['population_suitability']
        default_population_suit_task = graph.add_task(
            _population_curve_people_per_sqkm,
            kwargs={
                'pop_density_path':file_registry['population_suit_sqkm'],
                'target_raster_path':file_registry['population_suitability'],
                'rural_population_max':float(args['rural_population_max']),
                'urbanization_population_max':float(args['urbanization_population_max']),
            },
            dependent_task_list=[population_suit_sqkm_task],
            target_path_list=[file_registry['population_suitability']],
            task_name=f'Default population Suit')
        outputs_to_tile.append((file_registry[f'population_suitability'], default_color_path))
    else:
        rural_pop_task = graph.add_task(
            suit_func_to_use['rural_population']['func_name'],
            args=(
                file_registry['population_suit_sqkm'],
                file_registry['rural_population_suit']),
            kwargs=suit_func_to_use['rural_population']['func_params'],
            dependent_task_list=[population_suit_sqkm_task],
            target_path_list=[file_registry['rural_population_suit']],
            task_name=f'Rural Population Suit')
        
        if args['urbanization_popualation_func_type'] != 'None':
            urbanization_task = graph.add_task(
                suit_func_to_use['urbanization_population']['func_name'],
                args=(
                    file_registry['population_suit_sqkm'],
                    file_registry['urbanization_population_suit']),
                kwargs=suit_func_to_use['urbanization_population']['func_params'],
                dependent_task_list=[population_suit_sqkm_task],
                target_path_list=[file_registry['urbanization_population_suit']],
                task_name=f'Urbanization Population Suit')
            
            rural_urbanization_task = graph.add_task(
                _rural_urbanization_combined,
                args=(
                    file_registry['rural_population_suit'],
                    file_registry['urbanization_population_suit'],
                    file_registry['population_suitability'],
                ),
                dependent_task_list=[rural_pop_task, urbanization_task],
                target_path_list=[file_registry['population_suitability']],
                task_name=f'Rural Urbanization Suit')
            outputs_to_tile.append((file_registry[f'population_suitability'], default_color_path))
        else:
            population_suitability_path = file_registry['rural_population_suit']
            outputs_to_tile.append((file_registry[f'rural_population_suit'], default_color_path))

    ### Water velocity
    if args['calc_water_velocity']:
        # calculate slope
        slope_task = graph.add_task(
            func=pygeoprocessing.calculate_slope,
            args=(
                (file_registry['aligned_dem'], 1),
                file_registry['slope']),
            target_path_list=[file_registry['slope']],
            dependent_task_list=[align_task],
            task_name='calculate slope')

        # water velocity risk is actually being calculated over the landscape
        # and not just where water is present. should it be masked to 
        # water presence?
        water_vel_task = graph.add_task(
            suit_func_to_use['water_velocity']['func_name'],
            args=(file_registry[f'slope'], file_registry['water_velocity_suit']),
            kwargs=suit_func_to_use['water_velocity']['func_params'],
            dependent_task_list=[slope_task],
            target_path_list=[file_registry['water_velocity_suit']],
            task_name=f'Water Velocity Suit')
        suitability_tasks.append(water_vel_task)
        habitat_suit_risk_paths.append(file_registry['water_velocity_suit'])
        habitat_suit_risk_weights.append(float(args['water_velocity_weight']))
        outputs_to_tile.append((file_registry['water_velocity_suit'], default_color_path))

    # Temperature and ndvi have different functions for wet and dry seasons.
    for season in ["dry", "wet"]:
        ### Water temperature
        if args['calc_temperature']:
            for temp_key in ['snail_water_temp', 'parasite_water_temp']:
                # NOTE: TODO I'm not sure this if/else is needed anymore if we use the kwargs signature approach
                if 'op_key' in suit_func_to_use[temp_key]['func_params']:
                    water_temp_task = graph.add_task(
                        suit_func_to_use[temp_key]['func_name'],
                        kwargs={
                            'water_temp_path':file_registry[f'aligned_water_temp_{season}'],
                            'target_raster_path':file_registry[f'{temp_key}_suit_{season}'],
                            **suit_func_to_use[temp_key]['func_params'],
                            },
                        dependent_task_list=[align_task],
                        target_path_list=[file_registry[f'{temp_key}_suit_{season}']],
                        task_name=f'{temp_key} suit for {season}')
                else:
                    water_temp_task = graph.add_task(
                        suit_func_to_use[temp_key]['func_name'],
                        args=(
                            file_registry[f'aligned_water_temp_{season}'],
                            file_registry[f'{temp_key}_suit_{season}'],
                        ),
                        kwargs=suit_func_to_use[temp_key]['func_params'],
                        dependent_task_list=[align_task],
                        target_path_list=[file_registry[f'{temp_key}_suit_{season}']],
                        task_name=f'{temp_key} suit for {season}')
                suitability_tasks.append(water_temp_task)
                habitat_suit_risk_paths.append(file_registry[f'{temp_key}_suit_{season}'])
                habitat_suit_risk_weights.append(float(args[f'{temp_key}_{season}_weight']))
                outputs_to_tile.append((file_registry[f'{temp_key}_suit_{season}'], default_color_path))

        ### Vegetation coverage (NDVI)
        if args['calc_ndvi']:
            ndvi_task = graph.add_task(
                suit_func_to_use['ndvi']['func_name'],
                args=(
                    file_registry[f'aligned_ndvi_{season}'],
                    file_registry[f'ndvi_suit_{season}'],
                ),
                kwargs=suit_func_to_use['ndvi']['func_params'],
                dependent_task_list=[align_task],
                target_path_list=[file_registry[f'ndvi_suit_{season}']],
                task_name=f'NDVI Suit for {season}')
            suitability_tasks.append(ndvi_task)
            habitat_suit_risk_paths.append(file_registry[f'ndvi_suit_{season}'])
            habitat_suit_risk_weights.append(float(args[f'ndvi_{season}_weight']))
            outputs_to_tile.append((file_registry[f'ndvi_suit_{season}'], default_color_path))

    ### Distance from shore, proxy for depth
    if args['calc_water_depth']:
        inverse_water_mask_task = graph.add_task(
            _inverse_water_mask,
            kwargs={
                'input_raster_path': file_registry['aligned_water_presence'],
                'target_raster_path': file_registry['inverse_water_mask'],
                },
            target_path_list=[file_registry['inverse_water_mask']],
            dependent_task_list=[align_task],
            task_name='inverse water mask')

        distance_from_shore_task = graph.add_task(
            func=pygeoprocessing.distance_transform_edt,
            args=(
                (file_registry['inverse_water_mask'], 1),
                file_registry['distance_from_shore'],
                (default_pixel_size[0], default_pixel_size[0])),
            target_path_list=[file_registry['distance_from_shore']],
            dependent_task_list=[inverse_water_mask_task],
            task_name='inverse distance edt')

        water_depth_suit_path = file_registry['unmasked_water_depth_suit']
        water_depth_suit_task = graph.add_task(
            suit_func_to_use['water_depth']['func_name'],
            args=(
                file_registry[f'distance_from_shore'],
                water_depth_suit_path,
            ),
            kwargs=suit_func_to_use['water_depth']['func_params'],
            dependent_task_list=[distance_from_shore_task],
            target_path_list=[water_depth_suit_path],
            task_name=f'Water Depth Suit')

        # Using water presence mask out non water pixels, since risk
        # is tied to water here.
        masked_water_depth_suit_path = file_registry['water_depth_suit']
        mask_water_depth_suit_task = graph.add_task(
            _mask_non_water_values,
            kwargs={
                'input_raster_path': water_depth_suit_path,
                'mask_raster_path': file_registry['aligned_water_presence'],
                'target_raster_path': masked_water_depth_suit_path,
            },
            dependent_task_list=[water_depth_suit_task],
            target_path_list=[masked_water_depth_suit_path],
            task_name=f'Mask Water Depth Suit')

        suitability_tasks.append(mask_water_depth_suit_task)
        habitat_suit_risk_paths.append(masked_water_depth_suit_path)
        habitat_suit_risk_weights.append(float(args['water_depth_weight']))
        outputs_to_tile.append((masked_water_depth_suit_path, default_color_path))

    ### Custom functions provided by user
    for custom_index in ['one', 'two', 'three']:
        if args[f'calc_custom_{custom_index}']:
            target_key = f'custom_suit_{custom_index}'
            custom_task = graph.add_task(
                suit_func_to_use[f'custom_{custom_index}']['func_name'],
                args=(
                    file_registry[f'aligned_custom_{custom_index}'],
                    file_registry[target_key],
                ),
                kwargs=suit_func_to_use[f'custom_{custom_index}']['func_params'],
                dependent_task_list=[align_task],
                target_path_list=[file_registry[target_key]],
                task_name=f'Custom Suit for {custom_index}')
            suitability_tasks.append(custom_task)
            habitat_suit_risk_paths.append(file_registry[target_key])
            habitat_suit_risk_weights.append(float(args[f'custom_{custom_index}_weight']))
            outputs_to_tile.append((file_registry[target_key], default_color_path))

    ### Weighted arithmetic mean of water risks
    weighted_mean_task = graph.add_task(
        _weighted_mean,
        kwargs={
            'raster_list': habitat_suit_risk_paths,
            'weight_value_list': habitat_suit_risk_weights,
            'target_raster_path': file_registry['habitat_suit_weighted_mean'],
            'target_nodata': BYTE_NODATA,
            },
        target_path_list=[file_registry['habitat_suit_weighted_mean']],
        dependent_task_list=suitability_tasks,
        task_name='weighted mean')
    outputs_to_tile.append((file_registry[f'habitat_suit_weighted_mean'], default_color_path))


    ### Convolve habitat suit weighted mean over land

    # TODO: mask out water bodies to nodata and not include in risk
    decay_dist_m = float(args['decay_distance'])

    kernel_path = file_registry['kernel']
    max_dist_pixels = abs(
        decay_dist_m / squared_default_pixel_size[0])
    kernel_func = pygeoprocessing.kernels.create_distance_decay_kernel

    def decay_func(dist_array):
        return _kernel_gaussian(
            dist_array, max_distance=max_dist_pixels)

    kernel_kwargs = dict(
        target_kernel_path=kernel_path,
        distance_decay_function=decay_func,
        max_distance=max_dist_pixels,
        normalize=False)

    kernel_task = graph.add_task(
        kernel_func,
        kwargs=kernel_kwargs,
        task_name=(
            f'Create guassian kernel - {decay_dist_m}m'),
        target_path_list=[kernel_path])

    convolved_hab_risk_path = file_registry['unmasked_convolved_hab_risk']
    convolved_hab_risk_task = graph.add_task(
        _convolve_and_set_lower_bound,
        kwargs={
            'signal_path_band': (file_registry['habitat_suit_weighted_mean'], 1),
            'kernel_path_band': (kernel_path, 1),
            'target_path': convolved_hab_risk_path,
            'working_dir': args['workspace_dir'],
            'normalize': False,
        },
        task_name=f'Convolve hab risk - {decay_dist_m}m',
        target_path_list=[convolved_hab_risk_path],
    )

    # mask convolved output by AOI
    masked_convolved_path = file_registry['convolved_hab_risk']
    mask_aoi_task = graph.add_task(
        pygeoprocessing.mask_raster,
        kwargs={
            'base_raster_path_band': (convolved_hab_risk_path, 1),
            'mask_vector_path': args['aoi_path'],
            'target_mask_raster_path': masked_convolved_path,
        },
        target_path_list=[masked_convolved_path],
        dependent_task_list=[convolved_hab_risk_task],
        task_name='Mask convolved raster by AOI'
    )
    outputs_to_tile.append((masked_convolved_path, default_color_path))
        
    # min-max normalize the absolute risk convolution.
    # min is known to be 0, so we don't misrepresent positive risk values.
    normalize_task = graph.add_task(
        _normalize_raster,
        kwargs={
            'raster_path': masked_convolved_path,
            'target_raster_path': file_registry['normalized_convolved_risk'],
        },
        dependent_task_list=[mask_aoi_task],
        target_path_list=[file_registry['normalized_convolved_risk']],
        task_name=f'Normalize convolved risk')
    outputs_to_tile.append((file_registry['normalized_convolved_risk'], default_color_path))
    
    base_risk_path_list = [masked_convolved_path, file_registry['normalized_convolved_risk']] 
    base_task_list = [mask_aoi_task, normalize_task] 
    # For normalized and unormalized risk (relative, absolute) calculate risk
    # to population.
    for calc_type, base_risk_path, base_task in zip(['abs', 'rel'], base_risk_path_list, base_task_list):
        ### Weight convolved risk by population density
        risk_to_pop_path = file_registry[f'risk_to_pop_{calc_type}']
        risk_to_pop_task = graph.add_task(
            func=pygeoprocessing.raster_map,
            kwargs={
                'op': _multiply_op,
                'rasters': [population_suitability_path, base_risk_path],
                'target_path': risk_to_pop_path,
                #'target_nodata': FLOAT32_NODATA,
                },
            target_path_list=[risk_to_pop_path],
            dependent_task_list=[base_task, population_suit_sqkm_task],
            task_name=f'risk to population {calc_type}')
        outputs_to_tile.append((risk_to_pop_path, pop_color_path))
        
        ### Multiply risk_to_pop by people count
        risk_to_pop_count_path = file_registry[f'risk_to_pop_count_{calc_type}']
        risk_to_pop_count_task = graph.add_task(
            func=pygeoprocessing.raster_map,
            kwargs={
                'op': _multiply_op,
                'rasters': [risk_to_pop_path, file_registry['aligned_pop_count']],
                'target_path': risk_to_pop_count_path,
                #'target_nodata': FLOAT32_NODATA,
                },
            target_path_list=[risk_to_pop_count_path],
            dependent_task_list=[risk_to_pop_task],
            task_name=f'risk to pop_count {calc_type}')
        outputs_to_tile.append((risk_to_pop_count_path, pop_color_path))

    # Get the extents and center of the AOI for notebook companion
    aoi_info = pygeoprocessing.get_vector_info(args['aoi_path'])
    # WGS84 WKT
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    wgs84_wkt = srs.ExportToWkt()
    wgs84_bb = pygeoprocessing.geoprocessing.transform_bounding_box(
            aoi_info['bounding_box'], aoi_info['projection_wkt'], wgs84_wkt)
    aoi_center = (
        ((wgs84_bb[1] + wgs84_bb[3]) / 2), 
        ((wgs84_bb[0] + wgs84_bb[2]) / 2))
    nb_json_config['aoi_center'] = aoi_center

    # Save AOI as GeoJSON for companion notebook
    aoi_geojson_path = file_registry[f'aoi_geojson']

    aoi_geojson_task = graph.add_task(
        func=pygeoprocessing.reproject_vector,
        kwargs={
            'base_vector_path': args['aoi_path'],
            'target_projection_wkt': wgs84_wkt,
            'target_path': aoi_geojson_path,
            'driver_name': 'GeoJSON',
            'copy_fields': False,
            },
        target_path_list=[aoi_geojson_path],
        task_name=f'reproject aoi to geojson')
    nb_json_config['aoi_geojson'] = os.path.basename(aoi_geojson_path)

    # For the notebook to be able to display only the currently selected
    # risk layers over the http server, write to json config
    nb_json_config['layers'] = []
    for raster_path, _ in outputs_to_tile:
        base_name = os.path.splitext(os.path.basename(raster_path))[0]
        nb_json_config['layers'].append(base_name)

    graph.close()
    graph.join()
    
    ### Save plots of function choices
    # Read func params from table
    # TODO: determine whether to display population, urbanization, or 
    # something else.
    # Store plot path locations to display in Jupyter Notebook
    nb_json_config['plot_paths'] = []

    suitability_keys = [
        ('ndvi', args['calc_ndvi'], args['ndvi_dry_path']),
        ('default_population_suit', args['default_population_suit'],
         file_registry['population_suit_sqkm']),
        ('rural_population', not args['default_population_suit'],
         file_registry['rural_population_suit']),
        ('urbanization_population', (
            not args['default_population_suit'] and
            (args['urbanization_population_func_type'] != 'None')),
         file_registry['urbanization_population_suit']),
        ('water_velocity', args['calc_water_velocity'], file_registry[f'slope']),
        ('water_depth', args['calc_water_depth'], file_registry[f'distance_from_shore']),
        ('custom_one', args['calc_custom_one'], args['custom_one_path']),
        ('custom_two', args['calc_custom_two'], args['custom_two_path']),
        ('custom_three', args['calc_custom_three'], args['custom_three_path']),
        ('snail_water_temp', args['calc_temperature'], args['water_temp_dry_path']),
        ('parasite_water_temp', args['calc_temperature'], args['water_temp_dry_path'])]

    for suit_key, calc_suit, raster_path in suitability_keys:
        # Skip non selected suitability metrics
        if not calc_suit:
            continue

        user_func = suit_func_to_use[suit_key]['func_name']
        func_params = suit_func_to_use[suit_key]['func_params']
        # Default population and water depth have static functions
        if suit_key in ['default_population_suit', 'water_depth']:
            func_type = 'default'
        else:
            func_type = args[f'{suit_key}_func_type']
        
        # Use input raster range to plot against function
        plot_png_name = f"{suit_key}-{func_type}.png"
        plot_raster = gdal.OpenEx(raster_path)
        plot_band = plot_raster.GetRasterBand(1)
        min_max_val = plot_band.ComputeRasterMinMax(True)
        plot_band = None
        plot_raster = None
        LOGGER.debug(
            f"finished computing min/max for {suit_key}: {min_max_val}")

        results = _generic_func_values(
            user_func, min_max_val, args['workspace_dir'], func_params)

        plot_path = file_registry['[SUIT_KEY]_[FUNC_NAME]', suit_key, func_type]
        _plotter(
            results[0], results[1], save_path=plot_path,
            label_x=suit_key, label_y=func_type,
            title=f'{suit_key}--{func_type}', xticks=None, yticks=None)
        # Track the current plots in the NB json config
        nb_json_config['plot_paths'].append(plot_png_name)

    # Write out the notebook json config
    with open(nb_json_config_path, 'w', encoding='utf-8') as f:
        json.dump(nb_json_config, f, ensure_ascii=False, indent=4)

    ### Tile outputs
    #tile_task = graph.add_task(
    #    _tile_raster,
    #    kwargs={
    #        'raster_path': file_registry['water_temp_suit_wet_sm'],
    #        'color_relief_path': color_relief_path,
    #    },
    #    task_name=f'Tile temperature',
    #    dependent_task_list=suitability_tasks)
    for raster_path, color_path in outputs_to_tile:
        _tile_raster(raster_path, color_path)


    return file_registry.registry
    LOGGER.info("Model completed")


def _water_depth_suit(shore_distance_path, target_raster_path):
    """Distance from shore risk as a proxy for water depth risk.

    Args:
        shore_distance_path (string): a path to a raster that has distance
            values from shore in meters. non water pixels should be 0 or NoData.
        target_raster_path (string): a path to write the resulting raster.

    Returns:
        Nothing
    """
    # Risk function is based on:
    #'y = y1 - (y2 - y1)/(x2-x1)  * x1 + (y2 - y1)/(x2-x1) * x 
    raster_info = pygeoprocessing.get_raster_info(shore_distance_path)
    raster_nodata = raster_info['nodata'][0]
    
    # Need to define the shape of the function
    # Taken from the shared google sheet
    xa = 0 
    ya = 1 
    xb = 210  
    yb = 0.097
    xc = 211
    yc = 0.076184 
    xd = 2000
    yd = 0

    slope_one = (yb - ya) / (xb - xa)
    slope_two = (yc - yb) / (xc - xb)
    slope_three = (yc - yd) / (xc - xd)
    y_intercept_two = yb - (slope_two * xb)
    y_intercept_three = yc - (slope_three * xc)

    # Pixel stack operation computed on the raster blocks
    def op(raster_array):
        # Define the output array
        output = numpy.full(
            raster_array.shape, FLOAT32_NODATA, dtype=numpy.float32)
        # Do not compute values where NoData
        valid_pixels = ~pygeoprocessing.array_equals_nodata(raster_array, raster_nodata)

        # First line
        mask_one = valid_pixels & (raster_array <= xb)
        output[mask_one] = (slope_one * raster_array[mask_one]) + ya
        
        # Second line
        mask_two = valid_pixels & (raster_array > xb) & (raster_array <= xc)
        output[mask_two] = (slope_two * raster_array[mask_two]) + y_intercept_two
        
        # Third line
        mask_three = valid_pixels & (raster_array > xc) & (raster_array <= xd)
        output[mask_three] = (slope_three * raster_array[mask_three]) + y_intercept_three

        # Everything greater than xd is 0
        mask_final = valid_pixels & (raster_array > xd)
        output[mask_final] = 0

        return output

    pygeoprocessing.raster_calculator(
        [(shore_distance_path, 1)], op, target_raster_path, gdal.GDT_Float32,
        FLOAT32_NODATA)


def _inverse_water_mask(input_raster_path, target_raster_path):
    """Set values of 1 to 0 and values of 0 to 1.

    Args:
        input_raster_path (string): a path to a raster of 1s and 0s.
        target_raster_path (string): a path to write the resulting raster.

    Returns:
        Nothing
    """
    input_info = pygeoprocessing.get_raster_info(input_raster_path)
    input_nodata = input_info['nodata'][0]
    input_datatype = input_info['datatype']

    def _inverse_op(input_array):
        output = numpy.full(input_array.shape, input_nodata)
        nodata_mask = pygeoprocessing.array_equals_nodata(input_array, input_nodata)
        water_mask = input_array == 1

        output[water_mask] = 0
        output[~water_mask] = 1
        output[nodata_mask] = input_nodata

        return output
    
    pygeoprocessing.raster_calculator(
        [(input_raster_path, 1)], _inverse_op, target_raster_path,
        input_datatype, input_nodata)


def _mask_non_water_values(input_raster_path, mask_raster_path, target_raster_path):
    """Set values to nodata if not covered by provided mask.

    Args:
        input_raster_path (string): path to raster to be masked
        mask_raster_path (string): path to the mask raster where values of 1 indicate
            valid values to keep
        target_raster_path (string): path to the output raster

    Returns:
        Nothing.
    """
    input_info = pygeoprocessing.get_raster_info(input_raster_path)
    input_nodata = input_info['nodata'][0]
    mask_info = pygeoprocessing.get_raster_info(mask_raster_path)
    mask_nodata = mask_info['nodata'][0]

    def _mask_op(input_array, mask_array):
        output = numpy.full(input_array.shape, input_nodata)
        valid_mask = (
                ~pygeoprocessing.array_equals_nodata(input_array, input_nodata) &
                ~pygeoprocessing.array_equals_nodata(mask_array, mask_nodata))

        water_mask = (mask_array == 1) & valid_mask
        output[water_mask] = input_array[water_mask]

        return output
    
    pygeoprocessing.raster_calculator(
        [(input_raster_path, 1), (mask_raster_path, 1)],
        _mask_op, target_raster_path, gdal.GDT_Float32, input_nodata)


def _weighted_mean(raster_list, weight_value_list, target_raster_path, target_nodata):
    """Weighted arithmetic mean wrapper.

    Args:
        raster_list (list of strings): a list of raster paths.
        weight_value_list (list of floats): a list of weights corresponding
            to the entries in raster_list.
        target_raster_path (string): a path to write weighted mean raster.
        target_nodata (float): value to use as nodata for output.

    Returns:
        Nothing.
    """

    def _weighted_mean_op(*arrays):
        """
        raster_map op for weighted arithmetic mean of habitat suitablity risk layers.
        `arrays` is expected to be a list of numpy arrays
        """

        return numpy.average(arrays, axis=0, weights=weight_value_list)

    pygeoprocessing.raster_map(
        op=_weighted_mean_op,
        rasters=raster_list,
        target_path=target_raster_path,
        target_nodata=target_nodata,
    )


def _normalize_raster(raster_path, target_raster_path):
    """Min-Max normalization with mininum fixed to 0."""

    raster = gdal.OpenEx(raster_path) 
    band = raster.GetRasterBand(1)
    # returns (min, max, mean, std)
    stats = band.GetStatistics(False, True)
    max_val = stats[1]
    min_val = 0

    def _normalize_op(array):
        """raster_map op for normalization."""

        return (array - min_val) / (max_val - min_val)

    pygeoprocessing.raster_map(
        op=_normalize_op,
        rasters=[raster_path],
        target_path=target_raster_path,
    )


def _rural_urbanization_combined(rural_raster_path, urbanization_raster_path, target_raster_path):
    """Combine the rural and urbanization functions.

    Takes a waterfall approach writing rural values first and then
    urbanization values, which could overwrite rural values.

    Args:
        rural_raster_path (string): a path to the rural raster.
        urbanization_raster_path (string): a path to the urbanization raster.
        target_raster_path (string): a path to write output result.

    Returns:
        Nothing
    """
    rural_info = pygeoprocessing.get_raster_info(rural_raster_path)
    rural_nodata = rural_info['nodata'][0]
    urbanization_info = pygeoprocessing.get_raster_info(urbanization_raster_path)
    urbanization_nodata = urbanization_info['nodata'][0]

    def _rural_urbanization_op(rural_array, urbanization_array):
        output = numpy.full(
            rural_array.shape, BYTE_NODATA, dtype=numpy.float32)
        valid_mask = (
                ~pygeoprocessing.array_equals_nodata(rural_array, rural_nodata) &
                ~pygeoprocessing.array_equals_nodata(urbanization_array, urbanization_nodata) )
        
        # We should be able to waterfall these by setting rural values first
        # and then overwriting any urbanization results.
        output[valid_mask] = rural_array[valid_mask]
        output[valid_mask] = urbanization_array[valid_mask]

        return output
    
    pygeoprocessing.raster_calculator(
        [(rural_raster_path, 1), (urbanization_raster_path, 1)],
        _rural_urbanization_op, target_raster_path, gdal.GDT_Float32, BYTE_NODATA)


def _multiply_op(array_one, array_two): return numpy.multiply(array_one, array_two)


def _tile_raster(raster_path, color_relief_path):
    """Create XYZ tiles for a given raster.

    Args:
        raster_path (string): path to a raster to tile.
        color_relief_path (string): path to a text file with the styling 
            definition to use.

    Returns:
        Nothing
    """
    # Set up directory and paths for outputs
    base_dir = os.path.dirname(raster_path)
    base_name = os.path.splitext(os.path.basename(raster_path))[0]
    rgb_raster_path = os.path.join(base_dir, f'{base_name}_rgb.tif')
    tile_dir = os.path.join(base_dir, f'{base_name}_tiles')

    if not os.path.isdir(tile_dir):
        os.mkdir(tile_dir)
    LOGGER.info(f'Creating stylized raster for {base_name}')
    gdaldem_cmd = f'gdaldem color-relief -q -alpha -co COMPRESS=LZW {raster_path} {color_relief_path} {rgb_raster_path}'
    subprocess.run(gdaldem_cmd, shell=True)
    LOGGER.info(f'Creating tiles for {base_name}')
    tile_cmd = [
        '--verbose', '--xyz', '--resampling=near', '--quiet',
        '--resume', '--zoom=1-12', '--process=4', 
        '--webviewer=leaflet', rgb_raster_path, tile_dir]
    gdal2tiles.main(tile_cmd)

### Water temperature functions ###
def _water_temp_op_sm(temp_array, temp_nodata):
    """Water temperature suitability for S. mansoni."""
    #SmWaterTemp <- function(Temp){ifelse(Temp<16, 0,ifelse(Temp<=35, -0.003 * (268/(Temp - 14.2) - 335) + 0.0336538, 0))}
    #=IFS(TEMP<16, 0, TEMP<=35, -0.003*(268/(TEMP-14.2)-335)+0.0336538, TEMP>35, 0)
    output = numpy.full(
        temp_array.shape, BYTE_NODATA, dtype=numpy.float32)
    nodata_pixels = pygeoprocessing.array_equals_nodata(temp_array, temp_nodata)

    # if temp is less than 16 or higher than 35 set to 0
    valid_range_mask = (temp_array>=16) & (temp_array<=35) & ~nodata_pixels
    output[valid_range_mask] = (
        -0.003 * (268 / (temp_array[valid_range_mask] - 14.2) - 335) + 0.0336538)
    output[~nodata_pixels & (temp_array < 16)] = 0
    output[~nodata_pixels & (temp_array > 35)] = 0
    output[nodata_pixels] = BYTE_NODATA

    return output


def _water_temp_op_sh(temp_array, temp_nodata):
    """Water temperature suitability for S. haematobium."""
    #ShWaterTemp <- function(Temp){ifelse(Temp<17, 0,ifelse(Temp<=33, -0.006 * (295/(Temp - 15.3) - 174) + 0.056, 0))}
    output = numpy.full(
        temp_array.shape, BYTE_NODATA, dtype=numpy.float32)
    nodata_pixels = pygeoprocessing.array_equals_nodata(temp_array, temp_nodata)

    # if temp is less than 16 set to 0
    valid_range_mask = (temp_array>=17) & (temp_array<=33) & ~nodata_pixels
    output[valid_range_mask] = (
        -0.006 * (295 / (temp_array[valid_range_mask] - 15.3) - 174) + 0.056)
    output[~nodata_pixels & (temp_array < 17)] = 0
    output[~nodata_pixels & (temp_array > 33)] = 0
    output[nodata_pixels] = BYTE_NODATA

    return output


def _water_temp_op_bt(temp_array, temp_nodata):
    """Water temperature suitability for Bulinus truncatus."""
    #BtruncatusWaterTempNEW <- function(Temp){ifelse(Temp<17, 0,ifelse(Temp<=33, -48.173 + 8.534e+00 * Temp + -5.568e-01 * Temp^2 + 1.599e-02 * Temp^3 + -1.697e-04 * Temp^4, 0))}
    output = numpy.full(
        temp_array.shape, BYTE_NODATA, dtype=numpy.float32)
    nodata_pixels = pygeoprocessing.array_equals_nodata(temp_array, temp_nodata)

    # if temp is less than 16 set to 0
    valid_range_mask = (temp_array>=17) & (temp_array<=33) & ~nodata_pixels
    output[valid_range_mask] = (
        -48.173 + (8.534 * temp_array[valid_range_mask]) + 
        (-5.568e-01 * numpy.power(temp_array[valid_range_mask], 2)) +
        (1.599e-02 * numpy.power(temp_array[valid_range_mask], 3)) +
        (-1.697e-04 * numpy.power(temp_array[valid_range_mask], 4)))
    output[~nodata_pixels & (temp_array < 17)] = 0
    output[~nodata_pixels & (temp_array > 33)] = 0
    output[nodata_pixels] = BYTE_NODATA

    return output


def _water_temp_op_bg(temp_array, temp_nodata):
    """Water temperature suitability for Biomphalaria."""
    #BglabrataWaterTempNEW <- function(Temp){ifelse(Temp<16, 0,ifelse(Temp<=35, -29.9111 + 5.015e+00 * Temp + -3.107e-01 * Temp^2 +8.560e-03 * Temp^3 + -8.769e-05 * Temp^4, 0))}
    output = numpy.full(
        temp_array.shape, BYTE_NODATA, dtype=numpy.float32)
    nodata_pixels = pygeoprocessing.array_equals_nodata(temp_array, temp_nodata)

    # if temp is less than 16 set to 0
    valid_range_mask = (temp_array>=16) & (temp_array<=35) & ~nodata_pixels
    output[valid_range_mask] = (
        -29.9111 + (5.015 * temp_array[valid_range_mask]) + 
        (-3.107e-01 * numpy.power(temp_array[valid_range_mask], 2)) +
        (8.560e-03 * numpy.power(temp_array[valid_range_mask], 3)) +
        (-8.769e-05 * numpy.power(temp_array[valid_range_mask], 4)))
    output[~nodata_pixels & (temp_array < 16)] = 0
    output[~nodata_pixels & (temp_array > 35)] = 0
    output[nodata_pixels] = BYTE_NODATA

    return output


def _water_temp_suit(water_temp_path, target_raster_path, op_key):
    """Default functions for temperature risk.

        Args:
            water_temp_path (string): path to water temperature raster.
            target_raster_path (string): path to write output results.
            op_key (string): string for which default function to use:
                'sh' | 'sm' | 'bt' | 'bg'

        Returns:
            Nothing.
    """
    TEMP_OP_MAP = {
        "sh": _water_temp_op_sh, 
        "sm": _water_temp_op_sm, 
        "bg": _water_temp_op_bg, 
        "bt": _water_temp_op_bt, 
    }
    water_temp_info = pygeoprocessing.get_raster_info(water_temp_path)
    water_temp_nodata = water_temp_info['nodata'][0]

    pygeoprocessing.raster_calculator(
        [(water_temp_path, 1), (water_temp_nodata, "raw")], TEMP_OP_MAP[op_key],
        target_raster_path, gdal.GDT_Float32, BYTE_NODATA)

### End water temperature functions ###

def _ndvi(ndvi_raster_path, target_raster_path):
    """Suitability risk function for vegetation coverage using NDVI.

    Args:
        ndvi_raster_path (string): path to ndvi raster.
        target_raster_path (string): path to write result output.

    Returns:
        Nothing.
    """
    #VegCoverage <- function(V){ifelse(V<0,0,ifelse(V<=0.3,3.33*V,1))}
    ndvi_info = pygeoprocessing.get_raster_info(ndvi_raster_path)
    ndvi_nodata = ndvi_info['nodata'][0]
    def op(ndvi_array):
        output = numpy.full(
            ndvi_array.shape, BYTE_NODATA, dtype=numpy.float32)
        valid_pixels = (~pygeoprocessing.array_equals_nodata(ndvi_array, ndvi_nodata))

        # if temp is less than 0 set to 0
        mask = valid_pixels & (ndvi_array>=0) & (ndvi_array<=0.3)
        output[mask] = (3.33 * ndvi_array[mask])
        output[valid_pixels & (ndvi_array < 0)] = 0
        output[valid_pixels & (ndvi_array > 0.3)] = 1
        output[~valid_pixels] = BYTE_NODATA

        return output

    pygeoprocessing.raster_calculator(
        [(ndvi_raster_path, 1)], op, target_raster_path, gdal.GDT_Float32,
        BYTE_NODATA)


def _population_curve_people_per_sqkm(
        pop_density_path, target_raster_path, rural_population_max,
        urbanization_population_max):
    """Risk based on population per square km.

    Risk function is defined as:
        - 0 if people per sq km < 2
        - linear if people per sq km >=2 and <=2000
        - sigmoidal decreasing to 0 if people per sq km >2000 to 200,000

    Args:
        pop_density_path (string): path to population density in people per sq km.
        target_raster_path (string): output path for population risk output

    Returns:
        Nothing
    """
    population_info = pygeoprocessing.get_raster_info(pop_density_path)
    population_nodata = population_info['nodata'][0]
    
    # Linear component definition
    slope = (1 - 0) / (rural_population_max - 2)
    intercept = 0 - (slope * 2)

    scurve_midpoint = (urbanization_population_max - rural_population_max) / 2

    def op(pop_density_array):
        """Population density risk curve.
            - 0 if people per sq km < 2
            - linear if people per sq km >=2 and <=2000
            - sigmoidal decreasing to 0 if people per sq km >2000 to 200,000
        Args:
            pop_density_path (string): path to population density in people per sq km.

        Returns:
            output (numpy array): population density risk.
        """


        output = numpy.full(
            pop_density_array.shape, BYTE_NODATA, dtype=numpy.float32)

        valid_pixels = (~pygeoprocessing.array_equals_nodata(pop_density_array, population_nodata))

        less_than_two_mask = (pop_density_array < 2) & (valid_pixels)
        output[less_than_two_mask] = 0

        linear_mask = (
            (pop_density_array >= 2) & (pop_density_array <= rural_population_max) & valid_pixels)
        output[linear_mask] = (slope * pop_density_array[linear_mask]) + intercept

        sigmoidal_mask = (pop_density_array > rural_population_max) & (valid_pixels)
        
        output[sigmoidal_mask] = (
            1 + (0 - 1) / (1 + numpy.exp(
                -1 * (pop_density_array[sigmoidal_mask] - scurve_midpoint) / urbanization_population_max)))

        return output

    pygeoprocessing.raster_calculator(
        [(pop_density_path, 1)], op, target_raster_path, gdal.GDT_Float32,
        BYTE_NODATA)


def _water_velocity(slope_raster_path, target_raster_path):
    """Suitability risk for water velocity based on DEM slope.

    Args:
        slope_raster_path (string): path to a raster of percent slope.
        target_raster_path (string): path to write output raster.

    Returns:
        Nothing
    """
    #WaterVel <- function(S){ifelse(S<=0.00014,-5714.3 * S + 1,-0.0029*S+0.2)}
    slope_info = pygeoprocessing.get_raster_info(slope_raster_path)
    slope_nodata = slope_info['nodata'][0]

    def op(slope_array):
        degree = numpy.full(
            slope_array.shape, FLOAT32_NODATA, dtype=numpy.float32)
        output = numpy.full(
            slope_array.shape, FLOAT32_NODATA, dtype=numpy.float32)
        valid_pixels = ~pygeoprocessing.array_equals_nodata(slope_array, slope_nodata)

        # percent slope to degrees
        # https://support.esri.com/en-us/knowledge-base/how-to-convert-the-slope-unit-from-percent-to-degree-in-000022558
        degree[valid_pixels] = numpy.degrees(numpy.arctan(slope_array[valid_pixels] / 100.0))
        mask_lt = valid_pixels & (degree <= 0.00014)
        output[mask_lt] = -5714.3 * degree[mask_lt] + 1
        mask_gt = valid_pixels & (degree > 0.00014)
        output[mask_gt] = -0.0029 * degree[mask_gt] + 0.2

        return output

    pygeoprocessing.raster_calculator(
        [(slope_raster_path, 1)], op, target_raster_path, gdal.GDT_Float32,
        FLOAT32_NODATA)


def _trapezoid_op(raster_path, target_raster_path, 
                  xa=12, ya=0, xb=20, yb=1, xc=30, yc=1, xz=40, yz=0):
    """Trapezoid function definition.
    
    Args:
        raster_path (string): path to a raster to apply function to
        target_raster_path (string): path to write output raster.
        xa (float): first x coordinate of trapezoids first line defined by
            two points: (xa, ya), (xb, yb)
        ya (float): first y coordinate of trapezoids first line defined by
            two points: (xa, ya), (xb, yb)
        xb (float): second x coordinate of trapezoids first line defined by
            two points: (xa, ya), (xb, yb)
        yb (float): second y coordinate of trapezoids first line defined by
            two points: (xa, ya), (xb, yb)
        xc (float): first x coordinate of trapezoids second line defined by
            two points: (xc, yc), (xz, yz)
        yc (float): first y coordinate of trapezoids second line defined by
            two points: (xc, yc), (xz, yz)
        xz (float): second x coordinate of trapezoids second line defined by
            two points: (xc, yc), (xz, yz)
        yz (float): second y coordinate of trapezoids second line defined by
            two points: (xc, yc), (xz, yz)

    Returns:
        Nothing.
    """
    #'y = y1 - (y2 - y1)/(x2-x1)  * x1 + (y2 - y1)/(x2-x1) * x 
    raster_info = pygeoprocessing.get_raster_info(raster_path)
    raster_nodata = raster_info['nodata'][0]

    slope_inc = (yb - ya) / (xb - xa)
    slope_dec = (yc - yz) / (xc - xz)
    intercept_inc = ya - (slope_inc * xa)
    intercept_dec = yc - (slope_dec * xc)

    def op(raster_array):
        output = numpy.full(
            raster_array.shape, FLOAT32_NODATA, dtype=numpy.float32)
        valid_pixels = (~pygeoprocessing.array_equals_nodata(raster_array, raster_nodata))

        # First plateau
        mask_one = valid_pixels & (raster_array <= xa)
        output[mask_one] = ya
        # First slope
        mask_linear_inc = valid_pixels & (raster_array > xa) & (raster_array < xb)
        output[mask_linear_inc] = (slope_inc * raster_array[mask_linear_inc]) + intercept_inc
        # Second plateau
        mask_three = valid_pixels & (raster_array >= xb) & (raster_array <= xc)
        output[mask_three] = yb
        # Second slope
        mask_linear_dec = valid_pixels & (raster_array > xc) & (raster_array < xz)
        output[mask_linear_dec] = (slope_dec * raster_array[mask_linear_dec]) + intercept_dec
        # Third plateau
        mask_four = valid_pixels & (raster_array >= xz)
        output[mask_four] = yz

        return output

    pygeoprocessing.raster_calculator(
        [(raster_path, 1)], op, target_raster_path, gdal.GDT_Float32,
        FLOAT32_NODATA)


def _gaussian_op(raster_path, target_raster_path, mean=0, std=1, lb=0, ub=40):
    """Gaussian function definition.
    
    Args:
        raster_path (string): path to a raster to apply function to
        target_raster_path (string): path to write resulting raster.
        mean (float): mean for gaussian distribution
        std (float): standard deviation for gaussian distribution
        lb (float): lower bound where values become zero
        ub (float): upper bound where values become zero

    Returns:
        Nothing.
    """
    raster_info = pygeoprocessing.get_raster_info(raster_path)
    raster_nodata = raster_info['nodata'][0]

    rv = scipy.stats.norm(loc=mean, scale=std)

    def op(raster_array):
        output = numpy.full(
            raster_array.shape, FLOAT32_NODATA, dtype=numpy.float32)
        valid_pixels = (~pygeoprocessing.array_equals_nodata(raster_array, raster_nodata))

        output[valid_pixels] = rv.pdf(raster_array[valid_pixels])
        bounds_mask = valid_pixels & (raster_array <= lb) & (raster_array >= ub)
        output[bounds_mask] = 0

        return output

    pygeoprocessing.raster_calculator(
        [(raster_path, 1)], op, target_raster_path, gdal.GDT_Float32,
        FLOAT32_NODATA)


def _sshape_op(
        raster_path, target_raster_path, yin=1, yfin=0, xmed=15, inv_slope=3):
    """S-curve, sigmoidal function definition.
    
    Args:
        raster_path (string): path to a raster to apply function to
        target_raster_path (string): path to write resulting raster.
        yin (float): initial y-intercept value
        yfin (float): value of y at tail
        xmed (float): x value where curve transitions
        inv_slope (float): defines the sharpness of the curve

    Returns:
        Nothing.
    """
    #y = yin + (yfin - yin)/(1 + exp(-(x - xmed)/invSlope)))
    raster_info = pygeoprocessing.get_raster_info(raster_path)
    raster_nodata = raster_info['nodata'][0]

    def op(raster_array):
        output = numpy.full(
            raster_array.shape, FLOAT32_NODATA, dtype=numpy.float32)
        valid_pixels = (~pygeoprocessing.array_equals_nodata(raster_array, raster_nodata))

        output[valid_pixels] = yin + (yfin - yin) / (1 + numpy.exp(-1 * ((raster_array[valid_pixels]) - xmed) / inv_slope))

        return output

    pygeoprocessing.raster_calculator(
        [(raster_path, 1)], op, target_raster_path, gdal.GDT_Float32,
        FLOAT32_NODATA)


def _exponential_decay_op(
        raster_path, target_raster_path, yin=1, xmed=1, decay_factor=0.982,
        max_dist=1000):
    """Exponential decay function definition.
    
    Args:
        raster_path (string): path to a raster to apply function to
        target_raster_path (string): path to write resulting raster.
        yin (float): initial y-intercept value
        xmed (float): the first points y coordinate that defines the line
        decay_factor (float): determines rate of decay
        max_dist (float): x value where y decays to 0

    Returns:
        Nothing.
    """
    raster_info = pygeoprocessing.get_raster_info(raster_path)
    raster_nodata = raster_info['nodata'][0]

    def op(raster_array):
        output = numpy.full(
            raster_array.shape, FLOAT32_NODATA, dtype=numpy.float32)
        valid_pixels = (~pygeoprocessing.array_equals_nodata(raster_array, raster_nodata))

        output[valid_pixels & (raster_array < xmed)] = yin
        exp_mask = valid_pixels & (raster_array >= xmed)
        output[exp_mask] = yin * (decay_factor**raster_array[exp_mask])

        return output

    pygeoprocessing.raster_calculator(
        [(raster_path, 1)], op, target_raster_path, gdal.GDT_Float32,
        FLOAT32_NODATA)


def _linear_op(raster_path, target_raster_path, xa, ya, xz, yz):
    """Linear function definition.
    
    Args:
        raster_path (string): path to a raster to apply function to
        target_raster_path (string): path to write resulting raster.
        xa (float): first x coordinate of a line defined by two points: (xa, ya), (xz, yz).
        ya (float): first y coordinate of a line defined by two points: (xa, ya), (xz, yz).
        xz (float): second x coordinate of a line defined by two points: (xa, ya), (xz, yz).
        yz (float): second y coordinate of a line defined by two points: (xa, ya), (xz, yz).

    Returns:
        Nothing.
    """
    raster_info = pygeoprocessing.get_raster_info(raster_path)
    raster_nodata = raster_info['nodata'][0]

    slope = (yz - ya) / (xz - xa)
    intercept = ya - (slope * xa)

    def op(raster_array):
        output = numpy.full(
            raster_array.shape, FLOAT32_NODATA, dtype=numpy.float32)
        valid_pixels = (~pygeoprocessing.array_equals_nodata(raster_array, raster_nodata))

        # First plateau
        mask_one = valid_pixels & (raster_array <= xa)
        output[mask_one] = ya
        # Line
        mask_linear_inc = valid_pixels & (raster_array > xa) & (raster_array < xz)
        output[mask_linear_inc] = (slope * raster_array[mask_linear_inc]) + intercept
        # Second plateau
        mask_two = valid_pixels & (raster_array >= xz)
        output[mask_two] = yz

        return output

    pygeoprocessing.raster_calculator(
        [(raster_path, 1)], op, target_raster_path, gdal.GDT_Float32,
        FLOAT32_NODATA)


def _generic_func_values(func_op, xrange, working_dir, kwargs):
    """Call a raster based function on a generic range of values.

    The point of this function is to be able to plot values in ``xrange``
    against ``func_op(x)``. Since ``func_op`` expects a raster to operate on
    we create one with the values of ``xrange`` to pass in.

    Args:
        func_op (string): the function to pass a created raster path to
        xrange (string): the range of values to use in creating raster
        working_dir (string): directory to save temporary files
        kwargs (dict): additional parameters to be passed along to func_op 

    Returns:
        values_x (numpy array): numpy array of x values
        numpy_values_y (numpy array): numpy array of func_op(x) values
    """
    # Generic spatial reference
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(26910)  # NAD83 / UTM zone 11N
    srs_wkt = srs.ExportToWkt()
    origin = (463250, 4929700)
    pixel_size = (30, -30)

    values_x = numpy.linspace(xrange[0], xrange[1], 100).reshape(10,10)
    nodata_x = -999

    tmp_working_dir = tempfile.mkdtemp(dir=working_dir)

    func_input_path = os.path.join(tmp_working_dir, f'temp-{func_op.__name__}.tif')
    pygeoprocessing.numpy_array_to_raster(
        values_x, nodata_x, pixel_size, origin, srs_wkt, func_input_path)
    func_result_path = os.path.join(tmp_working_dir, f'temp-{func_op.__name__}-result.tif')

    LOGGER.debug(f"func kwargs: {kwargs}")
    if kwargs:
        func_op(func_input_path, func_result_path, **kwargs)
    else:
        func_op(func_input_path, func_result_path)

    numpy_values_y = pygeoprocessing.raster_to_numpy_array(func_result_path)

    shutil.rmtree(tmp_working_dir, ignore_errors=True)

    return (values_x, numpy_values_y)


def _plotter(values_x, values_y, save_path=None, label_x=None, label_y=None,
             title=None, xticks=None, yticks=None):
    """Generic a plot of values.
    
    Args:
        values_x (numpy array): numpy array of x values to plot
        values_y (numpy array): numpy array of y values to plot
        save_path (string): path to save the plot as a png. optional, default=None.
        label_x (string): text for labelling the x-axis. optional, default=None.
        label_y (string): text for labelling the y-axis. optional, default=None.
        title (string): text for plot title. optional, default=None.
        xticks (): not currently implemented
        yticks ():  not currently implemented

    Returns:
        Nothing
    """
    flattened_x_array = values_x.flatten()
    flattened_y_array = values_y.flatten()
    xmin=numpy.min(flattened_x_array)
    xmax=numpy.max(flattened_x_array)
    ymin=numpy.min(flattened_y_array)
    ymax=numpy.max(flattened_y_array)

    # plot
    #plt.style.use('_mpl-gallery')
    fig, ax = plt.subplots()

    ax.plot(flattened_x_array, flattened_y_array, linewidth=2.0)

    #ax.set(xlim=(xmin, xmax), xticks=numpy.arange(xmin + 10, xmax, 10),
    #        ylim=(ymin-.1, ymax+.1), yticks=numpy.arange(ymin, ymax + .25, .25))
    ax.set(xlim=(xmin, xmax), ylim=(ymin-.1, ymax+.1))

    plt.xlabel(label_x)
    plt.ylabel(label_y)
    plt.title(title)
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()


def _square_off_pixels(raster_path):
    """Create square pixels from the provided raster.

    The pixel dimensions produced will respect the sign of the original pixel
    dimensions and will be the mean of the absolute source pixel dimensions.

    Args:
        raster_path (string): The path to a raster on disk.

    Returns:
        A 2-tuple of ``(pixel_width, pixel_height)``, in projected units.
    """
    raster_info = pygeoprocessing.get_raster_info(raster_path)
    pixel_width, pixel_height = raster_info['pixel_size']

    if abs(pixel_width) == abs(pixel_height):
        return (pixel_width, pixel_height)

    pixel_tuple = ()
    average_absolute_size = (abs(pixel_width) + abs(pixel_height)) / 2
    for pixel_dimension_size in (pixel_width, pixel_height):
        # This loop allows either or both pixel dimension(s) to be negative
        sign_factor = 1
        if pixel_dimension_size < 0:
            sign_factor = -1

        pixel_tuple += (average_absolute_size * sign_factor,)

    return pixel_tuple


def _resample_population_raster(
        source_population_raster_path, target_pop_count_raster_path,
        target_pop_density_raster_path,
        lulc_pixel_size, lulc_bb, lulc_projection_wkt, working_dir):
    """Resample a population raster without losing or gaining people.

    Population rasters are an interesting special case where the data are
    neither continuous nor categorical, and the total population count
    typically matters. Common resampling methods for continuous
    (interpolation) and categorical (nearest-neighbor) datasets leave room for
    the total population of a resampled raster to significantly change. This
    function resamples a population raster with the following steps:

        1. Convert a population count raster to population density per pixel
        2. Warp the population density raster to the target spatial reference
           and pixel size using bilinear interpolation.
        3. Convert the warped density raster back to population counts.

    Args:
        source_population_raster_path (string): The source population raster.
            Pixel values represent the number of people occupying the pixel.
            Must be linearly projected in meters.
        target_population_raster_path (string): The path to where the target,
            warped population raster will live on disk.
        lulc_pixel_size (tuple): A tuple of the pixel size for the target
            raster.  Passed directly to ``pygeoprocessing.warp_raster``.
        lulc_bb (tuple): A tuple of the bounding box for the target raster.
            Passed directly to ``pygeoprocessing.warp_raster``.
        lulc_projection_wkt (string): The Well-Known Text of the target
            spatial reference fro the target raster.  Passed directly to
            ``pygeoprocessing.warp_raster``.  Assumed to be a linear projection
            in meters.
        working_dir (string): The path to a directory on disk.  A new directory
            is created within this directory for the storage of temporary files
            and then deleted upon successful completion of the function.

    Returns:
        ``None``
    """
    if not os.path.isdir(working_dir):
        os.makedirs(working_dir)
    tmp_working_dir = tempfile.mkdtemp(dir=working_dir)
    population_raster_info = pygeoprocessing.get_raster_info(
        source_population_raster_path)
    pixel_area = abs(numpy.multiply(*population_raster_info['pixel_size']))
    population_nodata = population_raster_info['nodata'][0]

    population_srs = osr.SpatialReference()
    population_srs.ImportFromWkt(population_raster_info['projection_wkt'])

    # Convert population pixel area to square km
    population_pixel_area = (
        pixel_area * population_srs.GetLinearUnits()) / 1e6

    def _convert_population_to_density(population):
        """Convert population counts to population per square km.

        Args:
            population (numpy.array): A numpy array where pixel values
                represent the number of people who reside in a pixel.

        Returns:
            """
        out_array = numpy.full(
            population.shape, FLOAT32_NODATA, dtype=numpy.float32)
        valid_mask = ~pygeoprocessing.array_equals_nodata(population, population_nodata)
        out_array[valid_mask] = population[valid_mask] / population_pixel_area
        return out_array

    # Step 1: convert the population raster to population density per sq. km
    density_raster_path = os.path.join(tmp_working_dir, 'pop_density.tif')
    pygeoprocessing.raster_calculator(
        [(source_population_raster_path, 1)],
        _convert_population_to_density,
        density_raster_path, gdal.GDT_Float32, FLOAT32_NODATA)

    # Step 2: align to the LULC
    pygeoprocessing.warp_raster(
        density_raster_path,
        target_pixel_size=lulc_pixel_size,
        target_raster_path=target_pop_density_raster_path,
        resample_method='bilinear',
        target_bb=lulc_bb,
        target_projection_wkt=lulc_projection_wkt)

    # Step 3: convert the warped population raster back from density to the
    # population per pixel
    target_srs = osr.SpatialReference()
    target_srs.ImportFromWkt(lulc_projection_wkt)
    # Calculate target pixel area in km to match above
    target_pixel_area = abs((
        numpy.multiply(*lulc_pixel_size) * target_srs.GetLinearUnits()) / 1e6)

    def _convert_density_to_population(density):
        """Convert a population density raster back to population counts.

        Args:
            density (numpy.array): An array of the population density per
                square kilometer.

        Returns:
            A ``numpy.array`` of the population counts given the target pixel
            size of the output raster."""
        # We're using a float32 array here because doing these unit
        # conversions is likely to end up with partial people spread out
        # between multiple pixels.  So it's preserving an unrealistic degree of
        # precision, but that's probably OK because pixels are imprecise
        # measures anyways.
        out_array = numpy.full(
            density.shape, FLOAT32_NODATA, dtype=numpy.float32)

        # We already know that the nodata value is FLOAT32_NODATA
        valid_mask = ~pygeoprocessing.array_equals_nodata(density, FLOAT32_NODATA)
        out_array[valid_mask] = density[valid_mask] * target_pixel_area
        return out_array

    pygeoprocessing.raster_calculator(
        [(target_pop_density_raster_path, 1)],
        _convert_density_to_population,
        target_pop_count_raster_path, gdal.GDT_Float32, FLOAT32_NODATA)

    shutil.rmtree(tmp_working_dir, ignore_errors=True)


def _population_count_to_square_km(population_count_path, target_raster_path):
    """Population count to population density in square km.

    Args:
        population_count_path (str): path to population count raster
        target_raster_path (str): path to save density raster

    Returns:
        Nothing.
    """
    population_raster_info = pygeoprocessing.get_raster_info(population_count_path)
    population_pixel_area = abs(numpy.multiply(*population_raster_info['pixel_size']))

    # 1,000,000 square meters equals 1 square km.
    kwargs={
        'op': lambda x: (x / population_pixel_area) * 1000000,  
        'rasters': [population_count_path],
        'target_path': target_raster_path,
        'target_nodata': -1,
    }

    pygeoprocessing.raster_map(**kwargs)


def _kernel_gaussian(distance, max_distance):
    """Create a gaussian kernel.

    Args:
        distance (numpy.array): An array of euclidean distances (in pixels)
            from the center of the kernel.
        max_distance (float): The maximum distance of the kernel.  Pixels that
            are more than this number of pixels will have a value of 0.

    Returns:
        ``numpy.array`` with dtype of numpy.float32 and same shape as
        ``distance.
    """
    kernel = numpy.zeros(distance.shape, dtype=numpy.float32)
    pixels_in_radius = (distance <= max_distance)
    kernel[pixels_in_radius] = (
        (numpy.e ** (-0.5 * ((distance[pixels_in_radius] / max_distance) ** 2))
         - numpy.e ** (-0.5)) / (1 - numpy.e ** (-0.5)))
    return kernel


def _convolve_and_set_lower_bound(
        signal_path_band, kernel_path_band, target_path, working_dir, normalize):
    """Convolve a raster and set all values below 0 to 0.

    Args:
        signal_path_band (tuple): A 2-tuple of (signal_raster_path, band_index)
            to use as the signal raster in the convolution.
        kernel_path_band (tuple): A 2-tuple of (kernel_raster_path, band_index)
            to use as the kernel raster in the convolution.  This kernel should
            be non-normalized.
        target_path (string): Where the target raster should be written.
        working_dir (string): The working directory that
            ``pygeoprocessing.convolve_2d`` may use for its intermediate files.
        normalize (bool): whether to normalize the kernel

    Returns:
        ``None``
    """
    pygeoprocessing.convolve_2d(
        signal_path_band=signal_path_band,
        kernel_path_band=kernel_path_band,
        target_path=target_path,
        working_dir=working_dir,
        #ignore_nodata_and_edges=True,
        ignore_nodata_and_edges=False,
        mask_nodata=False,
        normalize_kernel=normalize
        )

    # Sometimes there are negative values that should have been clamped to 0 in
    # the convolution but weren't, so let's clamp them to avoid support issues
    # later on.
    target_raster = gdal.OpenEx(target_path, gdal.GA_Update)
    target_band = target_raster.GetRasterBand(1)
    target_nodata = target_band.GetNoDataValue()
    for block_data, block in pygeoprocessing.iterblocks(
            (target_path, 1)):
        valid_pixels = ~pygeoprocessing.array_equals_nodata(block, target_nodata)
        block[(block < 0) & valid_pixels] = 0
        target_band.WriteArray(
            block, xoff=block_data['xoff'], yoff=block_data['yoff'])

    target_band = None
    target_raster = None


@validation.invest_validator
def validate(args, limit_to=None):
    return validation.validate(args, MODEL_SPEC)
