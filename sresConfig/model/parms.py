#!/usr/bin/env python
# coding: utf-8
import os
import sys
import argparse  # system libraries
from datetime import datetime
from pathlib import Path
import csv


# -----------------------------------------------------------------------------
# class context
#
# This class is a serializable context for orchestration.
# -----------------------------------------------------------------------------
class parms(object):
    # Custom name for current run
    BATCH_NAME = 'batch_name'
    CAT_ID = 'catid'

    # Super Resolution parameters
    SRES_BATCH = 'sres_batch'
    SRES_DEVICE = 'sres_device'
    SRES_DATASET = 'sres_dataset'
    SRES_MODEL = 'sres_model'
    SRES_PIPELINE = 'sres_pipeline'
    SRES_PLATFORM = 'sres_platform'
    SRES_TASK = 'sres_task'
    SRES_VAR = 'sres_var'
    DIR_CONFIG = 'sres_config_dir'

    DIR_TOA = 'dir_toa'
    DIR_TARGET = 'dir_target'
    DIR_CLOUDMASK = 'dir_cloudmask'
    DIR_OUTPUT = 'dir_out'
    DIR_OUTPUT_CSV = 'dir_out_csv'
    DIR_OUTPUT_ERROR = 'dir_out_err'
    DIR_OUTPUT_WARP = 'dir_out_warp'

    # File names
    FN_DEST = 'fn_dest'
    FN_SRC = 'fn_src'
    FN_LIST = 'fn_list'
    FN_REPROJECTION_LIST = 'fn_reprojection_list'
    DS_LIST = 'ds_list'
    DS_WARP_LIST = 'ds_warp_list'
    DS_INTERSECTION_LIST = 'ds_intersection_list'
    DS_WARP_CLOUD_LIST = 'ds_warp_cloud_list'
    MA_LIST = 'ma_list'
    MA_WARP_LIST = 'ma_warp_list'
    MA_WARP_CLOUD_LIST = 'ma_warp_cloud_list'
    MA_WARP_VALID_LIST = 'ma_warp_valid_list'
    MA_WARP_MASKED_LIST = 'ma_warp_masked_list'
    PRED_LIST = 'pred_list'
    METRICS_LIST = 'metrics_list'
    ERROR_LIST = 'error_list'
    COMMON_MASK_LIST = 'common_mask_list'
    COMMON_MASK = 'common_mask'

    FN_TOA = 'fn_toa'
    FN_TOA_DOWNSCALE = 'fn_toa_downscale'
    DS_TOA_DOWNSCALE = 'ds_toa_downscale'
    MA_TOA_DOWNSCALE = 'ds_toa_downscale'
    FN_TARGET = 'fn_target'
    FN_TARGET_DOWNSCALE = 'fn_target_downscale'
    DS_TARGET_DOWNSCALE = 'ds_target_downscale'
    MA_TARGET_DOWNSCALE = 'ma_target_downscale'
    FN_CLOUDMASK = 'fn_cloudmask'
    FN_CLOUDMASK_DOWNSCALE = 'fn_cloudmask_downscale'
    DS_CLOUDMASK_DOWNSCALE = 'ds_cloudmask_downscale'
    MA_CLOUDMASK_DOWNSCALE = 'ma_cloudmask_downscale'
    FN_PREFIX = 'fn_prefix'
    FN_COG = 'fn_cog'
    FN_COG_8BAND = 'fn_cog_8band'
    FN_SUFFIX = 'fn_suffix'
    GEOM_TOA = 'geom_toa'

    # File name suffixes
    FN_TOA_SUFFIX = 'fn_toa_suffix'
    FN_TOA_DOWNSCALE_SUFFIX = '_toa_30m.tif'
    FN_TARGET_SUFFIX = 'fn_target_suffix'
    FN_TARGET_DOWNSCALE_SUFFIX = '_target_30m.tif'
    FN_CLOUDMASK_SUFFIX = 'fn_cloudmask_suffix'
    FN_CLOUDMASK_DOWNSCALE_SUFFIX = '_toa_clouds_30m.tif'
    FN_SRLITE_NONCOG_SUFFIX = '_noncog.tif'
    FN_SRLITE_SUFFIX = '_sr_02m.tif'
    FN_WARP_SUFFIX = '_warp.tif'

    # Band pairs
    LIST_BAND_PAIRS = 'list_band_pairs'
    LIST_BAND_PAIR_INDICES = 'list_band_pairs_indices'
    LIST_TOA_BANDS = 'list_toa_bands'
    LIST_TARGET_BANDS = 'list_target_bands'
    BAND_NUM = 'band_num'
    BAND_DESCRIPTION_LIST = 'band_description_list'

    # Index of data arrays FN_LIST, MA_LIST
    LIST_INDEX_TOA = 'list_index_toa'
    LIST_INDEX_TARGET = 'list_index_target'
    LIST_INDEX_CLOUDMASK = 'list_index_cloudmask'
    LIST_INDEX_THRESHOLD = 'list_index_threshold'

    # Target vars and defaults
    TARGET_GEO_TRANSFORM = 'target_geo_transform'
    TARGET_EXTENT = 'target_extent'
    TARGET_FN = 'target_fn'
    TARGET_XRES = 'target_xres'
    TARGET_YRES = 'target_yres'
    TARGET_PRJ = 'target_prj'
    TARGET_SRS = 'target_srs'
    TARGET_RASTERX_SIZE = 'target_rasterX_size'
    TARGET_RASTERY_SIZE = 'target_rasterY_size'
    TARGET_RASTER_COUNT = 'target_raster_count'
    TARGET_DRIVER = 'target_driver'
    TARGET_OUTPUT_TYPE = 'target_output_type'
    TARGET_DTYPE = 'target_dtype'
    TARGET_NODATA_VALUE = 'target_nodata_value'
    TARGET_SAMPLING_METHOD = 'target_sampling_method'

    # Default values
    DEFAULT_TOA_SUFFIX = 'toa.tif'
    DEFAULT_TARGET_SUFFIX = 'ccdc.tif'
    DEFAULT_CLOUDMASK_SUFFIX = 'toa.cloudmask.v1.2.tif'

    # Suffixs modified as per PM - 05/19/24
    DEFAULT_ERROR_REPORT_SUFFIX = 'sr_errors.csv'
    DEFAULT_STATISTICS_REPORT_SUFFIX = '_sr_stats.csv'
    DEFAULT_XRES = 30
    DEFAULT_YRES = 30
    DEFAULT_NODATA_VALUE = -9999
    DEFAULT_SAMPLING_METHOD = 'average'

    # Regression algorithms
    REGRESSION_MODEL = 'regressor'
    REGRESSOR_MODEL_OLS = 'ols'
    REGRESSOR_MODEL_HUBER = 'huber'
    REGRESSOR_MODEL_RMA = 'rma'

    # Storage type
    STORAGE_TYPE = 'storage'
    STORAGE_TYPE_MEMORY = 'memory'
    STORAGE_TYPE_FILE = 'file'

    # Debug & log values
    DEBUG_NONE_VALUE = 0
    DEBUG_TRACE_VALUE = 1
    DEBUG_VIZ_VALUE = 2
    DEBUG_LEVEL = 'debug_level'
    LOG_FLAG = 'log_flag'
    CLEAN_FLAG = 'clean_flag'
    VALIDATE_FLAG = 'validate_flag'
    NONCOG_FLAG = 'noncog_flag'
    CSV_FLAG = 'csv_flag'
    ERROR_REPORT_FLAG = 'error_report_flag'
    BAND8_FLAG = 'band8_flag'
    CSV_WRITER = 'csv_writer'

    # Quality flag and list of values
    QUALITY_MASK_FLAG = 'qf_mask_flag'
    LIST_QUALITY_MASK = 'list_quality_mask'
    POSITIVE_MASK_FLAG = 'positive_mask_flag'

    # Cloud mask flag
    CLOUD_MASK_FLAG = 'cloud_mask_flag'

    # Threshold flag
    THRESHOLD_MASK_FLAG = 'threshold_mask_flag'
    THRESHOLD_MIN = 'threshold_min'
    THRESHOLD_MAX = 'threshold_max'

    # Global instance variables
    context_dict = {}
    plotLib = None
    debug_level = 0
    writer = None

    # -------------------------------------------------------------------------
    # trace()
    #
    # Print trace debug (cus
    # -------------------------------------------------------------------------
    def trace(self, value):
        if (self._debug_level > 0):
            print(value)

    # -------------------------------------------------------------------------
    # __init__
    # -------------------------------------------------------------------------
    def __init__(self, debug_level):
        self._debug_level = debug_level
        args = self._getParser()
        # Initialize serializable context for orchestration
        try:
            self.context_dict[parms.DEBUG_LEVEL] = int(args.debug_level)
            self.context_dict[parms.VALIDATE_FLAG] = str(args.validatebool)
            # if eval(self.context_dict[parms.VALIDATE_FLAG]):
            config_location = str(args.sres_config_dir)
            if (os.path.isdir(config_location)):
                self.context_dict[parms.DIR_CONFIG] = config_location
            else:
                raise FileNotFoundError("Config directory not found: {}".format(config_location))
                
            #     self._create_logfile(self.context_dict[parms.REGRESSION_MODEL],
            #                          self.context_dict[parms.DIR_OUTPUT])
            # if (int(self.context_dict[parms.DEBUG_LEVEL]) >= int(self.DEBUG_TRACE_VALUE)):
            #     print(sys.path)

            self.context_dict[parms.SRES_BATCH] = str(args.sres_batch)
            self.context_dict[parms.SRES_DEVICE] = str(args.sres_device)
            self.context_dict[parms.SRES_VAR] = str(args.sres_var)            
            self.context_dict[parms.SRES_TASK] = str(args.sres_task)            
            self.context_dict[parms.SRES_DATASET] = str(args.sres_dataset)            
            self.context_dict[parms.SRES_MODEL] = str(args.sres_model)            
            self.context_dict[parms.SRES_PIPELINE] = str(args.sres_pipeline)            
            self.context_dict[parms.SRES_PLATFORM] = str(args.sres_platform)            

            # toa_location = str(args.toa_dir)

            # # If a TOA file is specified, create a list of one TOA files
            # if (os.path.isfile(toa_location)):
            #     self.context_dict[parms.DIR_TOA] = toa_location
            # else:
            #     self.context_dict[parms.DIR_TOA] = str(args.toa_dir)
            # self.context_dict[parms.DIR_TARGET] = str(args.target_dir)
            # self.context_dict[parms.DIR_CLOUDMASK] = str(args.cloudmask_dir)

            # # Manage output paths
            # self.context_dict[parms.DIR_OUTPUT] = str(args.out_dir)
            # try:
            #     os.makedirs(self.context_dict[parms.DIR_OUTPUT], exist_ok=True)
            # except OSError as error:
            #     print("Directory '%s' can not be created" % self.context_dict[parms.DIR_OUTPUT])

            # if (args.err_dir == "./"):
            #     self.context_dict[parms.DIR_OUTPUT_ERROR] = self.context_dict[parms.DIR_OUTPUT]
            # else:
            #     self.context_dict[parms.DIR_OUTPUT_ERROR] = str(args.err_dir)
            #     try:
            #         os.makedirs(self.context_dict[parms.DIR_OUTPUT_ERROR], exist_ok=True)
            #     except OSError as error:
            #         print("Directory '%s' can not be created" % self.context_dict[parms.DIR_OUTPUT_ERROR])

            # if (args.warp_dir == "./"):
            #     self.context_dict[parms.DIR_OUTPUT_WARP] = self.context_dict[parms.DIR_OUTPUT]
            # else:
            #     self.context_dict[parms.DIR_OUTPUT_WARP] = str(args.warp_dir)
            #     try:
            #         os.makedirs(self.context_dict[parms.DIR_OUTPUT_WARP], exist_ok=True)
            #     except OSError as error:
            #         print("Directory '%s' can not be created" % self.context_dict[parms.DIR_OUTPUT_WARP])

            # self.context_dict[parms.CSV_FLAG] = str(args.errorreportbool)
            # if (args.csv_dir == "./"):
            #     self.context_dict[parms.DIR_OUTPUT_CSV] = self.context_dict[parms.DIR_OUTPUT]
            # elif (args.csv_dir == "None"):
            #    self.context_dict[parms.CSV_FLAG] = str(False)
            #    self.context_dict[parms.DIR_OUTPUT_CSV] = self.context_dict[parms.DIR_OUTPUT]
            # else:
            #     self.context_dict[parms.DIR_OUTPUT_CSV] = str(args.csv_dir)
            #     try:
            #         os.makedirs(self.context_dict[parms.DIR_OUTPUT_CSV], exist_ok=True)
            #     except OSError as error:
            #         print("Directory '%s' can not be created" % self.context_dict[parms.DIR_OUTPUT_CSV])

            # # Parse general configuration parameters
            # self.context_dict[parms.LIST_BAND_PAIRS] = str(args.band_pairs_list)
            # self.context_dict[parms.TARGET_XRES] = int(args.target_xres)
            # self.context_dict[parms.TARGET_YRES] = int(args.target_yres)
            # self.context_dict[parms.TARGET_SAMPLING_METHOD] = str(args.target_sampling_method)

            # self.context_dict[parms.FN_TOA_SUFFIX] = '-' + str(args.toa_suffix)
            # self.context_dict[parms.FN_TARGET_SUFFIX] = '-' + str(args.target_suffix)
            # self.context_dict[parms.FN_CLOUDMASK_SUFFIX] = '-' + str(args.cloudmask_suffix)

            # self.context_dict[parms.REGRESSION_MODEL] = str(args.regressor)
            # self.context_dict[parms.DEBUG_LEVEL] = int(args.debug_level)
            # self.context_dict[parms.CLEAN_FLAG] = str(args.cleanbool)
            # self.context_dict[parms.NONCOG_FLAG] = str(args.noncogbool)
            # self.context_dict[parms.LOG_FLAG] = str(args.logbool)
            # if eval(self.context_dict[parms.LOG_FLAG]):
            #     self._create_logfile(self.context_dict[parms.REGRESSION_MODEL],
            #                          self.context_dict[parms.DIR_OUTPUT])
            # if (int(self.context_dict[parms.DEBUG_LEVEL]) >= int(self.DEBUG_TRACE_VALUE)):
            #     print(sys.path)
            # # self.context_dict[parms.ALGORITHM_CLASS] = str(args.algorithm)
            # self.context_dict[parms.STORAGE_TYPE] = str(args.storage)
            # self.context_dict[parms.CLOUD_MASK_FLAG] = str(args.cmaskbool)
            # self.context_dict[parms.POSITIVE_MASK_FLAG] = str(args.pmaskbool)
            # self.context_dict[parms.ERROR_REPORT_FLAG] = str(args.csvbool)
            # self.context_dict[parms.BAND8_FLAG] = str(args.band8bool)
            # self.context_dict[parms.QUALITY_MASK_FLAG] = str(args.qfmaskbool)
            # self.context_dict[parms.LIST_QUALITY_MASK] = str(args.qfmask_list)

            # self.context_dict[parms.THRESHOLD_MASK_FLAG] = str(args.thmaskbool)
            # threshold_range = (str(args.threshold_range)).partition(",")
            # self.context_dict[parms.THRESHOLD_MIN] = int(threshold_range[0])
            # self.context_dict[parms.THRESHOLD_MAX] = int(threshold_range[2])

        except BaseException as err:
            print('Check arguments: ', err)
            sys.exit(1)

        # Initialize instance variables
#        self.debug_level = int(self.context_dict[parms.DEBUG_LEVEL])

        # Echo input parameter values
    # SRES_VAR = 'sres_var'
    # SRES_TASK = 'sres_task'
    # SRES_DATASET = 'sres_dataset'
    # SRES_PIPELINE = 'sres_pipeline'
    # SRES_PLATFORM = 'sres_platform'

        self.trace(f'Initializing Super Resolution invocation with the following parameters:')
        if (self.context_dict[parms.SRES_BATCH] != None): self.trace(f'Batch:    {self.context_dict[parms.SRES_BATCH]}')
        if (self.context_dict[parms.SRES_DEVICE] != None): self.trace(f'Device:    {self.context_dict[parms.SRES_DEVICE]}')
        self.trace(f'Dataset:    {self.context_dict[parms.SRES_DATASET]}')
        self.trace(f'Model:    {self.context_dict[parms.SRES_MODEL]}')
        self.trace(f'Pipeline:    {self.context_dict[parms.SRES_PIPELINE]}')
        self.trace(f'Platform:    {self.context_dict[parms.SRES_PLATFORM]}')
        self.trace(f'Task:    {self.context_dict[parms.SRES_TASK]}')
        self.trace(f'Var:    {self.context_dict[parms.SRES_VAR]}')

        return

    # -------------------------------------------------------------------------
    # getParser()
    #
    # Print trace debug (cus
    # -------------------------------------------------------------------------
    def _getParser(self):
        """
        :return: argparser object with CLI commands.
        """
        parser = argparse.ArgumentParser()

        # Super Resolution parameters
        # var = "sst"
        # task = "cape_basin",
        # dataset = "LLC4320",
        # pipeline = "sres",
        # platform = "explore_gt"
        
        parser.add_argument(
            "--batch", "--batch", type=str, required=False, dest='sres_batch',
            default=None, help="Specify batch name for run."
        )
        parser.add_argument(
            "--device", "--device", type=str, required=False, dest='sres_device',
            default=None, help="Override default device name for run."
        )
        parser.add_argument(
            "-config_dir", "--config_dir", type=str, required=True, dest='sres_config_dir',
            default=None, help="Specify directory path containing CONFIG files."
        )
        parser.add_argument(
            "-dataset", "--dataset", type=str, required=True, dest='sres_dataset',
            default="LLC4320", help="Specify super-resolution dataset to process."
        )
        parser.add_argument(
            "-model", "--model", type=str, required=True, dest='sres_model',
            default="rcan", help="Specify super-resolution model to process."
        )
        parser.add_argument(
            "-pipeline", "--pipeline", type=str, required=True, dest='sres_pipeline',
            default="sres", help="Specify super-resolution pipeline to process."
        )
        parser.add_argument(
            "-platform", "--platform", type=str, required=True, dest='sres_platform',
            default="explore_gt", help="Specify super-resolution platform to process."
        )
        parser.add_argument(
            "-task", "--task", type=str, required=True, dest='sres_task',
            default="cape_basin", help="Specify super-resolution task to process."
        )
        parser.add_argument(
            "-var", "--var", type=str, required=True, dest='sres_var',
            default="sst", help="Specify super-resolution variable to process."
        )
        parser.add_argument(
            "--debug", "--debug_level", type=int, required=False, dest='debug_level',
            default=parms.DEBUG_NONE_VALUE, help="Specify debug level [0,1,2,3]"
        )
        parser.add_argument(
            "--validate", "--validate", required=False, dest='validatebool',
            action='store_true', help="Validate input parameters at startup."
        )



        ### SRLite-specific below
        parser.add_argument(
            "-target_dir", "--input-target-dir", type=str, required=False, dest='target_dir',
            default=None, help="Specify directory path containing TARGET files."
        )
        parser.add_argument(
            "-cloudmask_dir", "--input-cloudmask-dir", type=str, required=False, dest='cloudmask_dir',
            default=None, help="Specify directory path containing Cloudmask files."
        )
        parser.add_argument(
            "-bandpairs", "--input-list-of-band-pairs", type=str, required=False, dest='band_pairs_list',
            default="[['blue_ccdc', 'BAND-B'], ['green_ccdc', 'BAND-G'], ['red_ccdc', 'BAND-R'], ['nir_ccdc', 'BAND-N']]",
            help="Specify list of band pairs to be processed per TOA."
        )
        parser.add_argument(
            "-output_dir", "--output-directory", type=str, required=False, dest='out_dir',
            default="./", help="Specify output directory."
        )
        parser.add_argument(
            "--err_dir", "--output-err-dir", type=str, required=False, dest='err_dir',
            default="./", help="Specify directory path containing error files (defaults to out_dir)."
        )
        parser.add_argument(
            "--warp_dir", "--interim-warp-dir", type=str, required=False, dest='warp_dir',
            default="./", help="Specify directory path containing interim warped files (defaults to out_dir)."
        )
        parser.add_argument(
            "--csv_dir", "--output-csv-dir", type=str, required=False, dest='csv_dir',
            default="./", help="Specify directory path containing statistics files (defaults to out_dir)."
        )
        parser.add_argument(
            "--xres", "--input-x-resolution", type=str, required=False, dest='target_xres',
            default=parms.DEFAULT_XRES, help="Specify target X resolution (default = 30)."
        )
        parser.add_argument(
            "--yres", "--input-y-resolution", type=str, required=False, dest='target_yres',
            default=parms.DEFAULT_XRES, help="Specify target Y resolution (default = 30)."
        )
        parser.add_argument(
            "--sampling", "--reprojection-sampling-method", type=str, required=False, dest='target_sampling_method',
            default=parms.DEFAULT_SAMPLING_METHOD, help="Specify target warp sampling method (default = 'average'')."
        )
        parser.add_argument(
            "--toa_suffix", "--input-toa-suffix", type=str, required=False, dest='toa_suffix',
            default=parms.DEFAULT_TOA_SUFFIX, help="Specify TOA file suffix (default = -toa.tif')."
        )
        parser.add_argument(
            "--target_suffix", "--input-target-suffix", type=str, required=False, dest='target_suffix',
            default=parms.DEFAULT_TARGET_SUFFIX, help="Specify TARGET file suffix (default = -ccdc.tif')."
        )
        parser.add_argument(
            "--cloudmask_suffix", "--input-cloudmask-suffix", type=str, required=False, dest='cloudmask_suffix',
            default=parms.DEFAULT_CLOUDMASK_SUFFIX,
            help="Specify CLOUDMASK file suffix (default = -toa.cloudmask.v1.2.tif')."
        )
        # parser.add_argument(
        #     "--debug", "--debug_level", type=int, required=False, dest='debug_level',
        #     default=parms.DEBUG_NONE_VALUE, help="Specify debug level [0,1,2,3]"
        # )
        # parser.add_argument(
        #     "--clean", "--clean", required=False, dest='cleanbool',
        #     action='store_true', help="Force cleaning of generated artifacts prior to run (e.g, warp files)."
        # )
        #NOTE:  As per MC (3/19/24) regarding noncog flag: "Given your previous testing results that showed no real down side to COG,
        # I don’t see any reason to give the user the choice.  I would rather not offer options that may be confusing unless the 
        # users start to request them."  Functionality exists but should not be advertised to users in documentation.
        parser.add_argument(
            "--noncog", "--noncog", required=False, dest='noncogbool',
            action='store_true', help="Disable Cloud-optimized Geotiff format."
        )
        parser.add_argument(
            "--log", "--log", required=False, dest='logbool',
            action='store_true', help="Set logging."
        )
        parser.add_argument('--regressor',
                            required=False,
                            dest='regressor',
                            default='robust',
                            choices=['ols', 'huber', 'rma'],
                            help='Choose which regression algorithm to use')

        parser.add_argument('--pmask',
                            required=False,
                            dest='pmaskbool',
                            default=False,
                            action='store_true',
                            help='Suppress negative pixel values in reprojected bands')

        parser.add_argument('--csv',
                            required=False,
                            dest='csvbool',
                            default=True,
                            action='store_true',
                            help='Generate CSV file with runtime history')

        parser.add_argument('--err',
                            required=False,
                            dest='errorreportbool',
                            default=True,
                            action='store_true',
                            help='Generate error report')

        parser.add_argument('--band8',
                            required=False,
                            dest='band8bool',
                            default=False,
                            action='store_true',
                            help='Generate missing spectral bands [Coastal,Yellow,Rededge,NIR2] ' \
                                 + 'when using CCDC as the target  - use only when input = [Blue|Green|Red|NIR]')

        parser.add_argument('--storage',
                            required=False,
                            dest='storage',
                            default='memory',
                            choices=['memory', 'file'],
                            help='Choose which storage model to use')

        parser.add_argument('--cloudmask',
                            required=False,
                            dest='cmaskbool',
                            default=False,
                            action='store_true',
                            help='Apply cloud mask values to common mask')

        parser.add_argument('--qfmask',
                            required=False,
                            dest='qfmaskbool',
                            default=False,
                            action='store_true',
                            help='Apply quality flag values to common mask')

        parser.add_argument('--qfmasklist',
                            required=False,
                            dest='qfmask_list',
                            default='0,3,4',
                            type=str,
                            help='Choose quality flag values to mask')

        parser.add_argument('--thmask',
                            required=False,
                            dest='thmaskbool',
                            default=False,
                            action='store_true',
                            help='Apply threshold mask values to common mask')

        parser.add_argument('--thrange',
                            required=False,
                            dest='threshold_range',
                            default='-100, 2000',
                            type=str,
                            help='Choose quality flag values to mask')

        return parser.parse_args()

    # -------------------------------------------------------------------------
    # getDict()
    #
    # Get context dictionary
    # -------------------------------------------------------------------------
    def getDict(self):
        return self.context_dict


    # # -------------------------------------------------------------------------
    # # getDebugLevel()
    # #
    # # Get debug_level
    # # -------------------------------------------------------------------------
    # def getDebugLevel(self):
    #     return self.debug_level

    # -------------------------------------------------------------------------
    # getFileNames()
    #
    # Get input file names
    # -------------------------------------------------------------------------
    def getFileNames(self, prefix, context):
        """
        :param prefix: core TOA file name (must match core target and cloudmask file name)
        :param context: input context object dictionary
        :return: updated context
        """
        context[parms.FN_PREFIX] = str((prefix[1]).split("-toa.tif", 1)[0])
        last_index = context[parms.FN_PREFIX].rindex('_')
        context[parms.CAT_ID] =  context[parms.FN_PREFIX] [last_index+1:]

        # Provide the fully-qualified file name (if provided).  Otherwise assume, list of files
        if os.path.isfile(Path(context[parms.DIR_TOA])):
            context[parms.FN_TOA] = context[parms.DIR_TOA]
        else:
            context[parms.FN_TOA] = os.path.join(context[parms.DIR_TOA] + '/' +
                                                   context[parms.FN_PREFIX] + context[parms.FN_TOA_SUFFIX])

        if os.path.isfile(Path(context[parms.DIR_TARGET])):
            context[parms.FN_TARGET] = context[parms.DIR_TARGET]
        else:
            context[parms.FN_TARGET] = os.path.join(context[parms.DIR_TARGET] + '/' +
                                                      context[parms.FN_PREFIX] + context[parms.FN_TARGET_SUFFIX])

        if os.path.isfile(Path(context[parms.DIR_CLOUDMASK])):
            context[parms.FN_CLOUDMASK] = context[parms.DIR_CLOUDMASK]
        else:
            context[parms.FN_CLOUDMASK] = os.path.join(context[parms.DIR_CLOUDMASK] + '/' +
                                                         context[parms.FN_PREFIX] + context[
                                                             parms.FN_CLOUDMASK_SUFFIX])

        # Name artifacts according to TOA prefix
        context[parms.FN_TOA_DOWNSCALE] = os.path.join(context[parms.DIR_OUTPUT] + '/' +
                                                         context[parms.FN_PREFIX] + self.FN_TOA_DOWNSCALE_SUFFIX)
        context[parms.FN_TARGET_DOWNSCALE] = os.path.join(context[parms.DIR_OUTPUT] + '/' +
                                                            context[
                                                                parms.FN_PREFIX] + self.FN_TARGET_DOWNSCALE_SUFFIX)
        context[parms.FN_CLOUDMASK_DOWNSCALE] = os.path.join(context[parms.DIR_OUTPUT] + '/' +
                                                               context[
                                                                   parms.FN_PREFIX] + self.FN_CLOUDMASK_DOWNSCALE_SUFFIX)

                                                    
        if (eval(self.context_dict[parms.NONCOG_FLAG])):
            context[parms.FN_COG] = os.path.join(context[parms.DIR_OUTPUT] + '/' +
                                               context[parms.FN_PREFIX] + self.FN_SRLITE_NONCOG_SUFFIX)
        else:
            context[parms.FN_COG] = os.path.join(context[parms.DIR_OUTPUT] + '/' +
                                               context[parms.FN_PREFIX] + self.FN_SRLITE_SUFFIX)


        if not (os.path.exists(context[parms.FN_TOA])):
            raise FileNotFoundError("TOA File not found: {}".format(context[parms.FN_TOA]))
        if not (os.path.exists(context[parms.FN_TARGET])):
            self.plot_lib.trace("Processing: " + context[parms.FN_TOA])
            raise FileNotFoundError("TARGET File not found: {}".format(context[parms.FN_TARGET]))
        if (eval(self.context_dict[parms.CLOUD_MASK_FLAG])):
            if not (os.path.exists(context[parms.FN_CLOUDMASK])):
                self.plot_lib.trace("Processing: " + context[parms.FN_TOA])
                raise FileNotFoundError("Cloudmask File not found: {}".format(context[parms.FN_CLOUDMASK]))

        return context

    # -------------------------------------------------------------------------
    # _create_logfile()
    #
    # Print trace debug (cus
    # -------------------------------------------------------------------------
    def _create_logfile(self, model, logdir='results'):
        """
        :param args: argparser object
        :param logdir: log directory to store log file
        :return: logfile instance, stdour and stderr being logged to file
        """
        logfile = os.path.join(logdir, '{}_log_{}_model.txt'.format(
            datetime.now().strftime("%Y%m%d-%H%M%S"), model))
        print('See ', logfile)
        so = se = open(logfile, 'w')  # open our log file
        sys.stdout = os.fdopen(sys.stdout.fileno(), 'w')  # stdout buffering
        os.dup2(so.fileno(), sys.stdout.fileno())  # redirect to the log file
        os.dup2(se.fileno(), sys.stderr.fileno())
        return logfile
