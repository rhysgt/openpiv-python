import pathlib
from dataclasses import dataclass
from importlib_resources import files
from typing import Optional, Tuple, Union
import numpy as np

@dataclass
class PIVSettings:
    """ All the settings for PIV. Default settings are set at the initiation"""
    
    "Data related settings"

    # Input directory (containing images to be processed)
    filepath_images: Union[pathlib.Path, str] = files('openpiv') / "data" / "test1"  # type: ignore
    
    # Format and Image Sequence
    frame_pattern_a: str = 'exp1_001_a.bmp'
    frame_pattern_b: str = 'exp1_001_b.bmp'

    # Output directory
    save_directory: pathlib.Path = filepath_images.parent

    # Output filename
    save_filename: str = 'test1'

    "Region of interest"

    # (xmin,xmax,ymin,ymax) or 'full' for full image
    roi: Union[Tuple[int, int, int, int], str] = "full"

    "Image preprocessing"

    # Every image would be processed separately and the
    # average mask is applied to both A, B, but it's varying
    # for the frames sequence
    #: None for no masking
    #: 'edges' for edges masking, 
    #: 'intensity' for intensity masking
    dynamic_masking_method: Optional[str] = None # ['edge','intensity']
    dynamic_masking_threshold: float = 0.005
    dynamic_masking_filter_size: int = 7

    # Static masking applied to all images, A,B
    static_mask: Optional[np.ndarray] = None # or a boolean matrix of image shape

    "Processing Parameters"

    correlation_method: str="circular"  # ['circular', 'linear']
    normalized_correlation: bool = False

    # add the interroagtion window size for each pass.
    # For the moment, it should be a power of 2
    windowsizes: Tuple[int, ...] = (64,32,16)
    
    # The overlap of the interroagtion window for each pass.
    overlap: Tuple[int, ...] = (32, 16, 8)  # This is 50% overlap

    # Has to be a value with base two. In general window size/2 is a good
    # choice.

    num_iterations: int = len(windowsizes)  # select the number of PIV passes

    # methode used for subpixel interpolation:
    # 'gaussian','centroid','parabolic'
    subpixel_method: str = "gaussian"
    # use vectorized sig2noise and subpixel approximation functions
    use_vectorized: bool = True
    # 'symmetric' or 'second image', 'symmetric' splits the deformation
    # both images, while 'second image' does only deform the second image.
    deformation_method: str = 'second image'  # 'symmetric' or 'second image'
    # order of the image interpolation for the window deformation
    interpolation_order: int = 3
    scaling_factor: float = 1.0  # scaling factor pixel/meter
    dt: float = 1.0  # time between to frames (in seconds)

    # Signal to noise ratio:
    # we can decide to estimate it or not at every vector position
    # we can decided if we use it for validation or only store it for 
    # later post-processing
    # plus we need some parameters for threshold validation and for the 
    # calculations:

    sig2noise_method: Optional[str] = None # or "peak2peak" or "None"
    # select the width of the masked to masked out pixels next to the main
    # peak
    sig2noise_mask: int = 2
    # If extract_sig2noise::False the values in the signal to noise ratio
    # output column are set to NaN
    
    # "Validation based on the signal to noise ratio"
    # Note: only available when extract_sig2noise::True and only for the
    # last pass of the interrogation
    # Enable the signal to noise ratio validation. Options: True or False
    # sig2noise_validate: False  # This is time consuming
    # minmum signal to noise ratio that is need for a valid vector
    sig2noise_threshold: float = 1.0
    sig2noise_validate: bool = False # when it's False we can save time by not
    #estimating sig2noise ratio at all, so we can set both sig2noise_method to None 


    "vector validation options"

    # choose if you want to do validation of the first pass: True or False
    validation_first_pass: bool = False
    # only effecting the first pass of the interrogation the following
    # passes in the multipass will be validated

    "Validation Parameters"

    # The validation is done at each iteration based on three filters.
    # The first filter is based on the min/max ranges. Observe that these
    # values are defined in
    # terms of minimum and maximum displacement in pixel/frames.
    min_max_validate: bool = False
    min_max_u_disp: Tuple = (-10000, 10000)
    min_max_v_disp: Tuple = (-10000, 10000)

    # The second filter is based on the global STD threshold
    std_validate: bool = False
    std_threshold: int = 7  # threshold of the std validation
    
    # The third filter is the median test (not normalized at the moment)
    median_validate: bool = False
    median_threshold: int = 3  # threshold of the median validation
    # On the last iteration, an additional validation can be done based on
    # the S/N.
    median_size: int = 1  # defines the size of the local median
    
    "Outlier replacement or Smoothing options"

    # Replacment options for vectors which are masked as invalid by the
    # validation
    # Choose: True or False
    replace_vectors: bool = True  # Enable the replacement.
    smoothn: bool = False  # Enables smoothing of the displacement field
    smoothn_p: float = 0.5  # This is a smoothing parameter
    # select a method to replace the outliers:
    # 'localmean', 'disk', 'distance'
    filter_method: str = "localmean"
    # maximum iterations performed to replace the outliers
    max_filter_iteration: int = 3
    filter_kernel_size: int = 2  # kernel size for the localmean method
    
    "Output options"

    # Select if you want to save the plotted vectorfield: True or False
    save_plot: bool = False
    # Choose wether you want to see the vectorfield or not:True or False
    show_plot: bool = False
    scale_plot: int = 100  # select a value to scale the quiver plot of
    # the vectorfield run the script with the given settings

    show_all_plots: bool=False

    invert: bool = False  # for the test_invert

    fmt: str = "%.4e"
    
    "CPU/GPU settings"

    num_cpus: int = 1
    max_array_size: int = None